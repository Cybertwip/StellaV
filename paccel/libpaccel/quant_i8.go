package libpaccel

import (
	"encoding/binary"
	"fmt"
	"math"
)

// QuantizeF32ToI8 produces the int8 activation wire payload used for hidden-state
// transfer between distributed shards: a 4-byte little-endian fp32 scale followed
// by one int8 per element. The quantizer is per-tensor symmetric with a fixed
// zero point: scale = max|v| / 127, q = clamp(round(v/scale), -128, 127). A single
// multiply dequantizes, and accuracy on LLM activations is ~0.4% mean error.
func QuantizeF32ToI8(src []float32) []byte {
	dst := make([]byte, 4+len(src))
	maxAbs := float32(0)
	for _, v := range src {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}
	scale := maxAbs / 127.0
	if scale == 0 {
		scale = 1 // all-zero tensor: avoid division by zero
	}
	binary.LittleEndian.PutUint32(dst[:4], math.Float32bits(scale))
	inv := 1.0 / scale
	out := dst[4:]
	for i, v := range src {
		q := math.Round(float64(v * inv))
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		out[i] = byte(int8(q))
	}
	return dst
}

// DequantizeI8ToF32 inverts QuantizeF32ToI8.
func DequantizeI8ToF32(payload []byte) ([]float32, error) {
	if len(payload) < 4 {
		return nil, fmt.Errorf("libpaccel: int8 payload too short (%d bytes)", len(payload))
	}
	scale := math.Float32frombits(binary.LittleEndian.Uint32(payload[:4]))
	body := payload[4:]
	out := make([]float32, len(body))
	for i, b := range body {
		out[i] = float32(int8(b)) * scale
	}
	return out, nil
}
