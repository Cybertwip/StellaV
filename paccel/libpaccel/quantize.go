package libpaccel

import (
	"fmt"
	"math"
)

// QuantizedTensor is the in-memory result of block-wise quantization. It mirrors
// the reference accelerator::QuantizedTensor structure: a shape, the bit width,
// the block size, one fp32 scale per block, and the packed low-bit code stream.
type QuantizedTensor struct {
	Shape       []uint64
	BitWidth    uint32
	BlockSize   uint32
	Layout      uint32 // PackedLayoutLinear or PackedLayoutOutputTileInt4
	TileOutputs uint32 // tile width for the v6 int4 tiled layout (else 0)
	Scales      []float32
	Packed      []byte
}

// roundHalfAwayFromZero reproduces the reference round_to_long: add 0.5 toward
// the value's sign, then truncate. This is the rounding rule baked into every
// PAccel code, so dequantization parity depends on matching it exactly.
func roundHalfAwayFromZero(value float64) int64 {
	const maxI32 = float64(math.MaxInt32)
	const minI32 = float64(math.MinInt32)
	if value >= maxI32 {
		return math.MaxInt32
	}
	if value <= minI32 {
		return math.MinInt32
	}
	if value >= 0 {
		return int64(math.Trunc(value + 0.5))
	}
	return int64(math.Trunc(value - 0.5))
}

// TurboQuantize performs symmetric, mid-rise, block-wise affine quantization.
//
// The value stream is partitioned into contiguous blocks of BlockSize. For each
// block it computes maxAbs = max|v|, derives a single scale s = maxAbs/(C) where
// C = floor((2^b - 1)/2) is the positive code budget, and encodes every element
// as q = clamp(round(v/s) + C, 0, 2^b - 1). The constant C is also the integer
// representation of zero, so the format is exactly symmetric with an implicit
// zero point. Scales are stored per block; codes are bit-packed (see pack.go).
//
// name and preferOutputTile select the v6 AVX-friendly int4 tile layout for
// eligible rank-2 4-bit weights; pass "" and false for the plain linear layout.
func TurboQuantize(values []float32, shape []uint64, bitWidth, blockSize uint32, name string, preferOutputTile bool) (QuantizedTensor, error) {
	if bitWidth < 2 || bitWidth > 8 {
		return QuantizedTensor{}, fmt.Errorf("turbo quantize: bit width must be 2..8 (got %d)", bitWidth)
	}
	if blockSize == 0 {
		return QuantizedTensor{}, fmt.Errorf("turbo quantize: block size must be positive")
	}
	shape = NormalizedShape(shape, len(values))
	maxCode := uint32(1<<bitWidth) - 1
	center := int64(maxCode / 2)

	layout, tileOutputs := resolveLayout(name, shape, bitWidth, blockSize, preferOutputTile)
	packedBytes, err := packedByteCount(len(values), shape, bitWidth, layout, tileOutputs)
	if err != nil {
		return QuantizedTensor{}, err
	}
	blockCount := (len(values) + int(blockSize) - 1) / int(blockSize)

	q := QuantizedTensor{
		Shape:       append([]uint64(nil), shape...),
		BitWidth:    bitWidth,
		BlockSize:   blockSize,
		Layout:      layout,
		TileOutputs: tileOutputs,
		Scales:      make([]float32, blockCount),
		Packed:      make([]byte, packedBytes),
	}

	for block := 0; block < blockCount; block++ {
		begin := block * int(blockSize)
		end := begin + int(blockSize)
		if end > len(values) {
			end = len(values)
		}
		maxAbs := float32(0)
		for _, v := range values[begin:end] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return QuantizedTensor{}, fmt.Errorf("turbo quantize: non-finite input value")
			}
			if a := float32(math.Abs(float64(v))); a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(1)
		if maxAbs > 0 {
			scale = maxAbs / float32(maxCode/2)
		}
		q.Scales[block] = scale
		for i := begin; i < end; i++ {
			code := roundHalfAwayFromZero(float64(values[i]/scale)) + center
			if code < 0 {
				code = 0
			} else if code > int64(maxCode) {
				code = int64(maxCode)
			}
			setPackedCode(&q, i, byte(code))
		}
	}
	return q, nil
}

// TurboDequantize inverts TurboQuantize: it unpacks the codes and applies
// v = (code - C) * scale[block] for each element, where C = floor((2^b-1)/2).
func TurboDequantize(q QuantizedTensor) []float32 {
	valueCount := 1
	for _, dim := range q.Shape {
		valueCount *= int(dim)
	}
	if valueCount < 0 {
		return nil
	}
	maxCode := uint32(1<<q.BitWidth) - 1
	center := int(maxCode / 2)
	out := make([]float32, valueCount)
	for i := 0; i < valueCount; i++ {
		block := i / int(q.BlockSize)
		scale := float32(1)
		if block < len(q.Scales) {
			scale = q.Scales[block]
		}
		out[i] = float32(int(getPackedCode(q, i, valueCount))-center) * scale
	}
	return out
}

// QuantizedValueCount returns the logical element count described by q.Shape.
// It returns 0 if the shape is empty, invalid, or too large for this runtime.
func QuantizedValueCount(q QuantizedTensor) int {
	valueCount := 1
	maxInt := int(^uint(0) >> 1)
	for _, dim := range q.Shape {
		if dim == 0 || dim > uint64(maxInt/valueCount) {
			return 0
		}
		valueCount *= int(dim)
	}
	return valueCount
}

// DequantizeValue returns one logical element from a quantized tensor without
// expanding the whole tensor. It is the scalar primitive used by direct packed
// runtimes that want PAccel to remain a resident weight format.
func DequantizeValue(q QuantizedTensor, index int) float32 {
	valueCount := QuantizedValueCount(q)
	if index < 0 || index >= valueCount || q.BitWidth == 0 || q.BitWidth > 8 || q.BlockSize == 0 {
		return 0
	}
	maxCode := uint32(1<<q.BitWidth) - 1
	center := int(maxCode / 2)
	block := index / int(q.BlockSize)
	scale := float32(1)
	if block >= 0 && block < len(q.Scales) {
		scale = q.Scales[block]
	}
	return float32(int(getPackedCode(q, index, valueCount))-center) * scale
}

// DequantizeSpan expands a contiguous logical span into out. The span is still
// addressed in canonical row-major tensor order; packed layouts decide how each
// logical element maps to bytes internally.
func DequantizeSpan(q QuantizedTensor, begin int, out []float32) {
	for i := range out {
		out[i] = DequantizeValue(q, begin+i)
	}
}

// NormalizedShape validates that the product of dims equals valueCount; if dims
// are empty or inconsistent it falls back to a flat 1-D shape, matching the
// reference normalizer used by every front-end loader.
func NormalizedShape(dims []uint64, valueCount int) []uint64 {
	if len(dims) == 0 {
		return []uint64{uint64(valueCount)}
	}
	product := uint64(1)
	for _, dim := range dims {
		if dim == 0 || product > math.MaxUint64/dim {
			return []uint64{uint64(valueCount)}
		}
		product *= dim
	}
	if product != uint64(valueCount) {
		return []uint64{uint64(valueCount)}
	}
	return append([]uint64(nil), dims...)
}
