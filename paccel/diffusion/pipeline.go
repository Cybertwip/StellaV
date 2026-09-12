package diffusion

import "fmt"

// Scheduler is the sampler interface the pipeline drives. EulerScheduler and
// LCMScheduler both implement it, so image, video, and audio share one loop.
type Scheduler interface {
	Steps() int
	InitSigma() float64
	Timestep(step int) float64
	ScaleInput(x []float32, step int) []float32
	Step(modelOutput, sample []float32, step int) []float32
}

// Backbone predicts the model output (epsilon or v) for a noisy latent at a
// timestep, conditioned on a context vector. The concrete backbone — a Stable
// Audio DiT, an SVD UNet, or a CogVideoX transformer — owns its own shapes and
// runs its matmuls/attention on the compute engine; the pipeline only needs this
// one method.
type Backbone interface {
	Denoise(latent []float32, timestep float64, cond []float32) ([]float32, error)
}

// Decoder turns a denoised latent into output samples (audio PCM or pixels),
// i.e. the VAE / pretransform decoder.
type Decoder interface {
	Decode(latent []float32) ([]float32, error)
}

// CFG combines unconditional and conditional model outputs:
// out = uncond + scale*(cond - uncond). With scale <= 1 it returns cond.
func CFG(uncond, cond []float32, scale float64) []float32 {
	if scale <= 1 || uncond == nil {
		return cond
	}
	s := float32(scale)
	out := make([]float32, len(cond))
	for i := range cond {
		out[i] = uncond[i] + s*(cond[i]-uncond[i])
	}
	return out
}

// Pipeline runs the standard latent-diffusion denoising loop.
type Pipeline struct {
	Scheduler     Scheduler
	Backbone      Backbone
	Decoder       Decoder
	GuidanceScale float64
}

// Generate runs the loop from an initial unit-variance latent: it scales the
// latent by the scheduler's init sigma, then for each step preconditions the
// sample, runs the backbone (twice for classifier-free guidance), applies the
// scheduler update, and finally decodes the clean latent.
func (p *Pipeline) Generate(latent, cond, uncond []float32) ([]float32, error) {
	if p.Scheduler == nil || p.Backbone == nil {
		return nil, fmt.Errorf("diffusion: pipeline needs a scheduler and a backbone")
	}
	x := make([]float32, len(latent))
	init := float32(p.Scheduler.InitSigma())
	for i := range latent {
		x[i] = latent[i] * init
	}
	for step := 0; step < p.Scheduler.Steps(); step++ {
		in := p.Scheduler.ScaleInput(x, step)
		t := p.Scheduler.Timestep(step)
		condOut, err := p.Backbone.Denoise(in, t, cond)
		if err != nil {
			return nil, fmt.Errorf("diffusion: backbone step %d: %w", step, err)
		}
		modelOut := condOut
		if p.GuidanceScale > 1 && uncond != nil {
			uncondOut, err := p.Backbone.Denoise(in, t, uncond)
			if err != nil {
				return nil, fmt.Errorf("diffusion: backbone uncond step %d: %w", step, err)
			}
			modelOut = CFG(uncondOut, condOut, p.GuidanceScale)
		}
		x = p.Scheduler.Step(modelOut, x, step)
	}
	if p.Decoder == nil {
		return x, nil // no decoder: return the clean latent
	}
	return p.Decoder.Decode(x)
}
