package libpaccel

import (
	"encoding/binary"
	"math"
	"strings"
)

// Quantization defaults, matching the reference compressor.
const (
	DefaultBits          = 4  // compact transformer-block profile
	DefaultBlockSize     = 32 // values per scale
	SensitiveMinBits     = 8  // floor for quality-sensitive tensors
	RecordOverheadBytes  = 96 // amortized per-record container overhead
	LargeDimThreshold    = 8192
)

// DecodeRawTensor expands a raw little-endian payload of the given source dtype
// into float32. Half and bfloat16 are widened through the bit-exact converters.
func DecodeRawTensor(raw []byte, dataType uint32) []float32 {
	switch dataType {
	case WireFloat:
		if len(raw)%4 != 0 {
			return nil
		}
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out
	case WireFloat16:
		if len(raw)%2 != 0 {
			return nil
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = Fp16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out
	case WireBFloat16:
		if len(raw)%2 != 0 {
			return nil
		}
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = BFloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out
	case WireDouble:
		if len(raw)%8 != 0 {
			return nil
		}
		out := make([]float32, len(raw)/8)
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out
	default:
		return nil
	}
}

// ResolveMixedBitWidth raises quality-sensitive tensors to at least
// SensitiveMinBits while leaving ordinary transformer-block weights at the
// requested width — the "smart mixed precision" policy.
func ResolveMixedBitWidth(name string, shape []uint64, requested uint32) uint32 {
	if IsQualitySensitiveWeight(name, shape) {
		if requested > SensitiveMinBits {
			return requested
		}
		return SensitiveMinBits
	}
	return requested
}

// IsQualitySensitiveWeight flags tensors whose quantization error disproportionately
// damages output quality: token embeddings, the LM head, normalization and bias
// vectors, rotary/positional caches, and any tensor with a very large dimension.
func IsQualitySensitiveWeight(name string, shape []uint64) bool {
	decoded := strings.ToLower(name)
	decoded = strings.ReplaceAll(decoded, "_2e", ".")
	decoded = strings.ReplaceAll(decoded, "_5f", "_")
	for strings.Contains(decoded, "__") {
		decoded = strings.ReplaceAll(decoded, "__", "_")
	}
	for _, marker := range []string{
		"embed", "lm_head", "word_embeddings", "tok_embeddings", "wte",
		"per_layer", "rope", "sin_cache", "cos_cache", "norm", "bias",
	} {
		if strings.Contains(decoded, marker) {
			return true
		}
	}
	for _, dim := range shape {
		if dim > LargeDimThreshold {
			return true
		}
	}
	return false
}

// EstimatedQuantizedBytes is the packed+scale size a tensor would occupy if
// compressed at the given width/block — used to decide whether compression is
// worthwhile versus storing the tensor raw.
func EstimatedQuantizedBytes(valueCount uint64, bitWidth, blockSize uint32) uint64 {
	if valueCount == 0 || bitWidth == 0 || blockSize == 0 {
		return 0
	}
	packed := (valueCount*uint64(bitWidth) + 7) / 8
	scales := ((valueCount + uint64(blockSize) - 1) / uint64(blockSize)) * 2 // fp16
	return packed + scales
}

// ShapeValueCount returns the element count for a shape, or 0 on overflow/zero dim.
func ShapeValueCount(shape []uint64) uint64 {
	if len(shape) == 0 {
		return 1
	}
	total := uint64(1)
	for _, dim := range shape {
		if dim == 0 || total > math.MaxUint64/dim {
			return 0
		}
		total *= dim
	}
	return total
}
