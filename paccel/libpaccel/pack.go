package libpaccel

import (
	"fmt"
	"math"
	"os"
	"strings"
)

// Packed layouts. Linear is the universal LSB-first bit stream. OutputTileInt4
// is the v6 layout that re-orders 4-bit codes into AVX-friendly output-column
// tiles of 16 lanes so a SIMD matmul can load a tile per instruction.
const (
	PackedLayoutLinear         = 0
	PackedLayoutOutputTileInt4 = 1
	PackedTileInt4OutputCols   = 16
)

// PackLowBits serializes one code per element into a contiguous little-endian
// bit stream: element i occupies bits [i*b, i*b+b), least-significant bit first,
// spanning byte boundaries. This is the canonical linear packing.
func PackLowBits(codes []byte, bitWidth uint32) []byte {
	bitCount := len(codes) * int(bitWidth)
	packed := make([]byte, (bitCount+7)/8)
	bitPos := 0
	mask := uint32(1<<bitWidth) - 1
	for _, code := range codes {
		value := uint32(code) & mask
		for b := uint32(0); b < bitWidth; b++ {
			if value&(1<<b) != 0 {
				packed[bitPos/8] |= byte(1 << (bitPos % 8))
			}
			bitPos++
		}
	}
	return packed
}

// UnpackLowBits is the inverse of PackLowBits for the linear layout.
func UnpackLowBits(packed []byte, valueCount int, bitWidth uint32) []byte {
	codes := make([]byte, valueCount)
	bitPos := 0
	for i := 0; i < valueCount; i++ {
		var value uint32
		for b := uint32(0); b < bitWidth; b++ {
			if bitPos/8 < len(packed) && packed[bitPos/8]&(1<<(bitPos%8)) != 0 {
				value |= 1 << b
			}
			bitPos++
		}
		codes[i] = byte(value)
	}
	return codes
}

// resolveLayout decides whether a tensor is eligible for, and should use, the
// v6 output-tiled int4 layout. Eligibility requires 4-bit, rank-2, and a block
// size that is a positive multiple of the 16-lane tile that also divides the
// inner dimension.
func resolveLayout(name string, shape []uint64, bitWidth, blockSize uint32, preferOutputTile bool) (uint32, uint32) {
	if !outputTileInt4Eligible(shape, bitWidth, blockSize) {
		return PackedLayoutLinear, 0
	}
	if preferOutputTile || forceV6TileAll() || nameLooksOutputTiled(name) {
		return PackedLayoutOutputTileInt4, PackedTileInt4OutputCols
	}
	return PackedLayoutLinear, 0
}

func outputTileInt4Eligible(shape []uint64, bitWidth, blockSize uint32) bool {
	if bitWidth != 4 || len(shape) != 2 || shape[0] == 0 || shape[1] == 0 {
		return false
	}
	if blockSize < PackedTileInt4OutputCols || blockSize%PackedTileInt4OutputCols != 0 {
		return false
	}
	if shape[1]%uint64(blockSize) != 0 {
		return false
	}
	return shape[0] <= math.MaxInt && shape[1] <= math.MaxInt
}

func forceV6TileAll() bool {
	v := strings.TrimSpace(os.Getenv("POWER_PACCEL_V6_TILE_ALL"))
	return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes") || strings.EqualFold(v, "on")
}

func nameLooksOutputTiled(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "power_t") || strings.Contains(lower, "power_5f_t")
}

// packedByteCount returns the exact byte length of the packed code stream for a
// given layout. For the tiled layout each (tile, column) pair stores tileOutputs
// nibbles, i.e. tileOutputs/2 bytes.
func packedByteCount(valueCount int, shape []uint64, bitWidth, layout, tileOutputs uint32) (int, error) {
	if layout == PackedLayoutOutputTileInt4 {
		if tileOutputs == 0 || len(shape) != 2 {
			return 0, fmt.Errorf("turbo quantize: invalid tiled int4 layout")
		}
		cols := shape[0]
		outputs := shape[1]
		tileCount := (outputs + uint64(tileOutputs) - 1) / uint64(tileOutputs)
		bytes := tileCount * cols * uint64(tileOutputs/2)
		if bytes > math.MaxInt {
			return 0, fmt.Errorf("turbo quantize: tiled packed payload too large")
		}
		return int(bytes), nil
	}
	bits := uint64(valueCount) * uint64(bitWidth)
	bytes := (bits + 7) / 8
	if bytes > math.MaxInt {
		return 0, fmt.Errorf("turbo quantize: packed payload too large")
	}
	return int(bytes), nil
}

// setPackedCode writes a single code into the packed stream at logical element
// index, honoring the tensor's layout and bit width. Fast paths exist for the
// byte-aligned 8/4/2-bit cases; arbitrary widths fall back to bit-by-bit.
func setPackedCode(q *QuantizedTensor, index int, code byte) {
	if q.Layout == PackedLayoutOutputTileInt4 {
		byteIndex, highNibble := tileInt4ByteIndex(q, index)
		if highNibble {
			q.Packed[byteIndex] |= (code & 0x0f) << 4
		} else {
			q.Packed[byteIndex] |= code & 0x0f
		}
		return
	}
	switch q.BitWidth {
	case 8:
		q.Packed[index] = code
	case 4:
		byteIndex := index / 2
		if index&1 == 0 {
			q.Packed[byteIndex] |= code & 0x0f
		} else {
			q.Packed[byteIndex] |= (code & 0x0f) << 4
		}
	case 2:
		byteIndex := index / 4
		shift := uint((index & 3) * 2)
		q.Packed[byteIndex] |= (code & 0x03) << shift
	default:
		bitPos := index * int(q.BitWidth)
		value := uint32(code) & ((1 << q.BitWidth) - 1)
		for b := uint32(0); b < q.BitWidth; b++ {
			if value&(1<<b) != 0 {
				q.Packed[bitPos/8] |= byte(1 << (bitPos % 8))
			}
			bitPos++
		}
	}
}

// getPackedCode reads a single code back out; it is the exact inverse of
// setPackedCode for every supported layout and bit width.
func getPackedCode(q QuantizedTensor, index, valueCount int) byte {
	if q.Layout == PackedLayoutOutputTileInt4 {
		byteIndex, highNibble := tileInt4ByteIndex(&q, index)
		if byteIndex >= len(q.Packed) {
			return 0
		}
		if highNibble {
			return (q.Packed[byteIndex] >> 4) & 0x0f
		}
		return q.Packed[byteIndex] & 0x0f
	}
	switch q.BitWidth {
	case 8:
		if index < len(q.Packed) {
			return q.Packed[index]
		}
		return 0
	case 4:
		byteIndex := index / 2
		if byteIndex >= len(q.Packed) {
			return 0
		}
		if index&1 == 0 {
			return q.Packed[byteIndex] & 0x0f
		}
		return (q.Packed[byteIndex] >> 4) & 0x0f
	case 2:
		byteIndex := index / 4
		if byteIndex >= len(q.Packed) {
			return 0
		}
		shift := uint((index & 3) * 2)
		return (q.Packed[byteIndex] >> shift) & 0x03
	default:
		bitPos := index * int(q.BitWidth)
		var value uint32
		for b := uint32(0); b < q.BitWidth; b++ {
			if bitPos/8 < len(q.Packed) && q.Packed[bitPos/8]&(1<<(bitPos%8)) != 0 {
				value |= 1 << b
			}
			bitPos++
		}
		return byte(value)
	}
}

// tileInt4ByteIndex maps a row-major element index (column-major weight, i.e.
// index = col*outputs + output) to its byte position and nibble within the v6
// output-tiled int4 layout, and reports whether the high nibble is used.
func tileInt4ByteIndex(q *QuantizedTensor, index int) (int, bool) {
	outputs := int(q.Shape[1])
	cols := int(q.Shape[0])
	col := index / outputs
	output := index - col*outputs
	tile := output / int(q.TileOutputs)
	lane := output - tile*int(q.TileOutputs)
	byteIndex := (tile*cols+col)*int(q.TileOutputs/2) + lane/2
	return byteIndex, lane&1 == 1
}
