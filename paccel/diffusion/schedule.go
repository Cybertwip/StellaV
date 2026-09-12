// Package diffusion is a dependency-free port of the noise schedules and
// samplers the PowerEngine native image, video, and audio engines run on. It
// implements the exact algebra of the scaled-linear beta schedule, the
// Karras sigma schedule, the Euler v-prediction sampler (SVD/video/audio), the
// LCM epsilon sampler (few-step text-to-image), classifier-free guidance, and a
// generic latent-diffusion loop that composes them with a backbone and a decoder.
package diffusion

import "math"

// ScaledLinearBetas returns the diffusers "scaled_linear" beta schedule:
// betas = linspace(sqrt(betaStart), sqrt(betaEnd), T)^2.
func ScaledLinearBetas(betaStart, betaEnd float64, T int) []float64 {
	betas := make([]float64, T)
	bs, be := math.Sqrt(betaStart), math.Sqrt(betaEnd)
	for i := 0; i < T; i++ {
		t := 0.0
		if T > 1 {
			t = float64(i) / float64(T-1)
		}
		v := bs + (be-bs)*t
		betas[i] = v * v
	}
	return betas
}

// AlphasCumprod returns the cumulative product of (1 - beta_i).
func AlphasCumprod(betas []float64) []float64 {
	out := make([]float64, len(betas))
	acc := 1.0
	for i, b := range betas {
		acc *= 1.0 - b
		out[i] = acc
	}
	return out
}

// TrainSigmas returns the per-train-step sigma = sqrt((1-alphaCumprod)/alphaCumprod).
func TrainSigmas(alphasCumprod []float64) []float64 {
	out := make([]float64, len(alphasCumprod))
	for i, a := range alphasCumprod {
		out[i] = math.Sqrt((1.0 - a) / a)
	}
	return out
}

// KarrasSigmas builds the rho=7 Karras sigma schedule between sigmaMax and
// sigmaMin over n inference steps.
func KarrasSigmas(n int, sigmaMax, sigmaMin float64) []float64 {
	const rho = 7.0
	out := make([]float64, n)
	minInv := math.Pow(sigmaMin, 1.0/rho)
	maxInv := math.Pow(sigmaMax, 1.0/rho)
	for i := 0; i < n; i++ {
		t := 0.0
		if n > 1 {
			t = float64(i) / float64(n-1)
		}
		out[i] = math.Pow(maxInv+(minInv-maxInv)*t, rho)
	}
	return out
}

// Interpolate1D does clamped linear interpolation of values at fractional index x.
func Interpolate1D(values []float64, x float64) float64 {
	if x <= 0 {
		return values[0]
	}
	last := len(values) - 1
	if x >= float64(last) {
		return values[last]
	}
	i := int(math.Floor(x))
	t := x - float64(i)
	return values[i]*(1-t) + values[i+1]*t
}
