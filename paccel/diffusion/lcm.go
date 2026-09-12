package diffusion

import "math"

// LCMScheduler is the few-step Latent Consistency Model sampler with
// epsilon-prediction, used by the native text-to-image engine (1-8 steps). It
// sub-samples the original DDIM grid and jumps directly between selected
// timesteps, deterministically (the stochastic re-noising term is omitted).
type LCMScheduler struct {
	timesteps     []int     // descending selected timestep indices
	alphasCumprod []float64 // length trainT
}

// NewLCM builds an LCM scheduler from a scaled-linear beta schedule.
func NewLCM(numSteps, trainT, originalInferenceSteps, stepsOffset int, betaStart, betaEnd float64) *LCMScheduler {
	if trainT <= 0 {
		trainT = 1000
	}
	if numSteps <= 0 {
		numSteps = 4
	}
	if originalInferenceSteps <= 0 {
		originalInferenceSteps = 50
	}
	cumprod := AlphasCumprod(ScaledLinearBetas(betaStart, betaEnd, trainT))

	stepRatio := trainT / originalInferenceSteps
	if stepRatio <= 0 {
		stepRatio = 1
	}
	skip := originalInferenceSteps / numSteps
	if skip <= 0 {
		skip = 1
	}
	timesteps := make([]int, 0, numSteps)
	for i := 0; i < numSteps; i++ {
		idx := originalInferenceSteps - 1 - i*skip
		if idx < 0 {
			idx = 0
		}
		ts := idx*stepRatio + (stepRatio - 1) + stepsOffset
		if ts < 0 {
			ts = 0
		}
		if ts >= trainT {
			ts = trainT - 1
		}
		timesteps = append(timesteps, ts)
	}
	return &LCMScheduler{timesteps: timesteps, alphasCumprod: cumprod}
}

func (s *LCMScheduler) Steps() int            { return len(s.timesteps) }
func (s *LCMScheduler) InitSigma() float64    { return 1 }
func (s *LCMScheduler) Timestep(step int) float64 { return float64(s.timesteps[step]) }

// ScaleInput is the identity for LCM (the sample is already x_t).
func (s *LCMScheduler) ScaleInput(x []float32, step int) []float32 {
	out := make([]float32, len(x))
	copy(out, x)
	return out
}

// Step performs one LCM update. modelOutput is the epsilon prediction. It
// computes the predicted clean sample x0 = (x_t - sqrt(1-a_t)*eps)/sqrt(a_t) and
// jumps to the previous selected timestep: x_{prev} = sqrt(a_prev) * x0. On the
// final step a_prev = 1, so it returns x0.
func (s *LCMScheduler) Step(modelOutput, sample []float32, step int) []float32 {
	t := s.timesteps[step]
	alphaProd := s.alphasCumprod[t]
	betaProd := 1 - alphaProd
	sqAlpha := math.Sqrt(alphaProd)
	sqBeta := math.Sqrt(betaProd)

	predOrig := make([]float32, len(sample))
	for i := range sample {
		predOrig[i] = float32((float64(sample[i]) - sqBeta*float64(modelOutput[i])) / sqAlpha)
	}
	prevT := -1
	for _, ts := range s.timesteps {
		if ts < t {
			prevT = ts
			break
		}
	}
	alphaPrev := 1.0
	if prevT >= 0 {
		alphaPrev = s.alphasCumprod[prevT]
	}
	sqAlphaPrev := math.Sqrt(alphaPrev)
	out := make([]float32, len(sample))
	for i := range sample {
		out[i] = float32(sqAlphaPrev * float64(predOrig[i]))
	}
	return out
}

// GuidanceScaleEmbedding builds the sinusoidal timestep_cond a distilled LCM
// UNet expects: w=(scale-1)*1000, then concat(sin, cos) over a log-spaced grid.
func GuidanceScaleEmbedding(guidanceScale float64, dim int) []float32 {
	w := (guidanceScale - 1) * 1000
	half := dim / 2
	if half < 1 {
		half = 1
	}
	scale := math.Log(10000) / float64(half-1)
	out := make([]float32, dim)
	for i := 0; i < half; i++ {
		x := w * math.Exp(-scale*float64(i))
		out[i] = float32(math.Sin(x))
		if half+i < dim {
			out[half+i] = float32(math.Cos(x))
		}
	}
	return out
}
