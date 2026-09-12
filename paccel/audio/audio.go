// Package audio is the standalone Stable Audio 3-style text-to-audio engine. It
// composes the compute engine, the diffusion sampler core, and (a reference for)
// the DiT backbone and VAE decoder. The production pipeline is
// T5Gemma text encoder -> diffusion transformer (DiT) -> pretransform/VAE decoder;
// the heavy modules load from a .paccel via libpaccel and plug into the same
// diffusion.Backbone / diffusion.Decoder seams used here.
package audio

import (
	"math"

	"github.com/powerengine/paccel/compute"
	"github.com/powerengine/paccel/diffusion"
	"github.com/powerengine/paccel/mmengine"
)

// Config describes the audio model and sampler. Defaults follow Stable Audio 3.
type Config struct {
	SampleRate      int     // 44100
	Channels        int     // 2 (stereo)
	ModelDim        int     // DiT / latent channel width
	FF              int     // feed-forward width
	SamplesPerFrame int     // pretransform downsampling ratio (e.g. 4096)
	Steps           int     // sampler steps (8)
	Guidance        float64 // CFG scale (1.0 = off)
	TrainT          int
	BetaStart       float64
	BetaEnd         float64
	SigmaMin        float64
	SigmaMax        float64
}

// Default returns the Stable Audio 3 configuration (reference-sized backbone).
func Default() Config {
	return Config{
		SampleRate: 44100, Channels: 2, ModelDim: 64, FF: 256, SamplesPerFrame: 4096,
		Steps: 8, Guidance: 1.0, TrainT: 1000,
		BetaStart: 0.00085, BetaEnd: 0.012, SigmaMin: 0.002, SigmaMax: 700.0,
	}
}

// Engine is a ready-to-run audio generator.
type Engine struct {
	cfg  Config
	eng  *compute.Engine
	pipe *diffusion.Pipeline
}

// New builds an audio engine with reference weights. Replace the backbone/decoder
// weights via NewWithWeights to run real Stable Audio 3 parameters.
func New(cfg Config) (*Engine, error) {
	return NewWithWeights(cfg, mmengine.RandomWeights(cfg.ModelDim, cfg.FF, cfg.SamplesPerFrame*cfg.Channels, 7))
}

func NewWithWeights(cfg Config, w mmengine.Weights) (*Engine, error) {
	eng, err := compute.NewEngine(compute.Options{DisableGPU: true})
	if err != nil {
		return nil, err
	}
	// Stable Audio uses a v-prediction Euler/Karras sampler.
	sched := diffusion.NewEulerKarras(cfg.Steps, cfg.TrainT, cfg.BetaStart, cfg.BetaEnd, cfg.SigmaMin, cfg.SigmaMax)
	pipe := &diffusion.Pipeline{
		Scheduler:     sched,
		Backbone:      &mmengine.RefBackbone{Eng: eng, W: w},
		Decoder:       &mmengine.LinearDecoder{Eng: eng, W: w},
		GuidanceScale: cfg.Guidance,
	}
	return &Engine{cfg: cfg, eng: eng, pipe: pipe}, nil
}

// LatentFrames is the number of latent frames for a duration: ceil(rate*sec/down).
func (e *Engine) LatentFrames(durationSec float64) int {
	frames := int(math.Ceil(durationSec * float64(e.cfg.SampleRate) / float64(e.cfg.SamplesPerFrame)))
	if frames < 1 {
		frames = 1
	}
	return frames
}

// Generate produces interleaved PCM for a prompt and duration. The returned
// slice has length frames*SamplesPerFrame*Channels, normalized to [-1, 1].
func (e *Engine) Generate(prompt string, durationSec float64, seed int64) ([]float32, error) {
	frames := e.LatentFrames(durationSec)
	latent := mmengine.GaussianLatent(frames*e.cfg.ModelDim, seed)
	cond := mmengine.TextCond(prompt, e.cfg.ModelDim)
	uncond := mmengine.TextCond("", e.cfg.ModelDim)
	out, err := e.pipe.Generate(latent, cond, uncond)
	if err != nil {
		return nil, err
	}
	return mmengine.NormalizeAudio(out), nil
}

// Close releases the compute engine.
func (e *Engine) Close() { e.eng.Close() }
