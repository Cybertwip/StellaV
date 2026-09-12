package mmengine

import (
	"math"
	"math/rand"
)

// GaussianLatent returns n unit-variance Gaussian samples for the initial noise.
func GaussianLatent(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

// TextCond is a deterministic, dependency-free stand-in for a text encoder
// (T5Gemma / CLIP): it hashes the prompt into a smooth pooled conditioning vector
// of width dim. A production engine replaces this with the real encoder output.
func TextCond(prompt string, dim int) []float32 {
	out := make([]float32, dim)
	if dim == 0 {
		return out
	}
	h := uint64(1469598103934665603)
	for i := 0; i < len(prompt); i++ {
		h ^= uint64(prompt[i])
		h *= 1099511628211
	}
	r := rand.New(rand.NewSource(int64(h)))
	for i := range out {
		out[i] = float32(r.NormFloat64() * 0.1)
	}
	return out
}

// NormalizeAudio scales samples into [-1, 1] by peak, the final step before PCM.
func NormalizeAudio(samples []float32) []float32 {
	peak := float32(0)
	for _, v := range samples {
		a := float32(math.Abs(float64(v)))
		if a > peak {
			peak = a
		}
	}
	if peak == 0 {
		return samples
	}
	inv := 1.0 / peak
	out := make([]float32, len(samples))
	for i, v := range samples {
		out[i] = v * inv
	}
	return out
}
