package diffusion

import "math"

// EulerScheduler is the Karras/Euler ancestral-free sampler with v-prediction,
// as used by Stable Video Diffusion and the native audio/video engines. It holds
// steps+1 sigmas (the trailing 0 lets the final step land on x0) and the matching
// continuous timesteps.
type EulerScheduler struct {
	sigmas    []float64 // length steps+1, sigmas[steps] == 0
	timesteps []float64 // length steps
}

// NewEulerKarras builds an Euler scheduler with a Karras sigma schedule derived
// from a scaled-linear beta schedule, the standard SVD configuration.
func NewEulerKarras(steps, trainT int, betaStart, betaEnd, sigmaMin, sigmaMax float64) *EulerScheduler {
	if steps <= 0 {
		steps = 25
	}
	if trainT <= 0 {
		trainT = 1000
	}
	betas := ScaledLinearBetas(betaStart, betaEnd, trainT)
	trainSigmas := TrainSigmas(AlphasCumprod(betas))
	if sigmaMin <= 0 {
		sigmaMin = trainSigmas[0]
	}
	if sigmaMax <= 0 {
		sigmaMax = trainSigmas[len(trainSigmas)-1]
	}
	sigmas := KarrasSigmas(steps, sigmaMax, sigmaMin)
	// Continuous v-prediction timesteps: t = 0.25 * ln(sigma).
	timesteps := make([]float64, steps)
	for i, s := range sigmas {
		timesteps[i] = 0.25 * math.Log(math.Max(s, 1e-12))
	}
	sigmas = append(sigmas, 0)
	return &EulerScheduler{sigmas: sigmas, timesteps: timesteps}
}

func (s *EulerScheduler) Steps() int { return len(s.sigmas) - 1 }

// InitSigma is the standard deviation the initial latent noise is scaled by.
func (s *EulerScheduler) InitSigma() float64 {
	if len(s.sigmas) == 0 {
		return 1
	}
	return math.Sqrt(s.sigmas[0]*s.sigmas[0] + 1)
}

func (s *EulerScheduler) Timestep(step int) float64 { return s.timesteps[step] }

// ScaleInput returns x / sqrt(sigma^2 + 1), the preconditioning the backbone
// expects at this step.
func (s *EulerScheduler) ScaleInput(x []float32, step int) []float32 {
	sigma := s.sigmas[step]
	scale := float32(1.0 / math.Sqrt(sigma*sigma+1.0))
	out := make([]float32, len(x))
	for i := range x {
		out[i] = x[i] * scale
	}
	return out
}

// Step performs one Euler update with a v-prediction model output, advancing the
// sample from sigma[step] to sigma[step+1]. On the final step (next sigma = 0) it
// returns the predicted clean sample x0.
func (s *EulerScheduler) Step(modelOutput, sample []float32, step int) []float32 {
	sigma := s.sigmas[step]
	nextSigma := s.sigmas[step+1]
	denom := sigma*sigma + 1.0
	sqrtDenom := math.Sqrt(denom)
	out := make([]float32, len(sample))
	for i := range sample {
		// v-prediction -> predicted original sample x0.
		predOriginal := float64(modelOutput[i])*(-sigma/sqrtDenom) + float64(sample[i])/denom
		derivative := (float64(sample[i]) - predOriginal) / sigma
		out[i] = float32(float64(sample[i]) + derivative*(nextSigma-sigma))
	}
	return out
}
