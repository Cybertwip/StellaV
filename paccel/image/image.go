// Package image is the standalone few-step text-to-image engine. It composes the
// compute engine, the LCM diffusion sampler, and (a reference for) the UNet/DiT
// backbone and VAE decoder. The production pipeline is
// CLIP/T5 text encoder -> latent-diffusion UNet -> VAE decoder; those modules load
// from a .paccel via libpaccel and plug into the diffusion seams used here.
package image

import (
	"github.com/powerengine/paccel/compute"
	"github.com/powerengine/paccel/diffusion"
	"github.com/powerengine/paccel/mmengine"
)

// Config describes the image model and the LCM sampler.
type Config struct {
	Width, Height int     // output pixel dimensions
	Downsample    int     // VAE spatial downsample (8)
	LatentDim     int     // latent channels (model width)
	FF            int     // feed-forward width
	Channels      int     // output channels (3 = RGB)
	Steps         int     // LCM steps (1-8)
	Guidance      float64 // CFG scale
	TrainT        int
	OriginalSteps int
	BetaStart     float64
	BetaEnd       float64
}

// Default returns a 512x512, 4-step LCM configuration (reference-sized backbone).
func Default() Config {
	return Config{
		Width: 512, Height: 512, Downsample: 8, LatentDim: 64, FF: 256, Channels: 3,
		Steps: 4, Guidance: 1.0, TrainT: 1000, OriginalSteps: 50,
		BetaStart: 0.00085, BetaEnd: 0.012,
	}
}

func (c Config) latentTokens() int {
	return (c.Width / c.Downsample) * (c.Height / c.Downsample)
}

func (c Config) patchPixels() int {
	return c.Downsample * c.Downsample * c.Channels
}

// Engine is a ready-to-run text-to-image generator.
type Engine struct {
	cfg  Config
	eng  *compute.Engine
	pipe *diffusion.Pipeline
}

func New(cfg Config) (*Engine, error) {
	return NewWithWeights(cfg, mmengine.RandomWeights(cfg.LatentDim, cfg.FF, cfg.patchPixels(), 11))
}

func NewWithWeights(cfg Config, w mmengine.Weights) (*Engine, error) {
	eng, err := compute.NewEngine(compute.Options{DisableGPU: true})
	if err != nil {
		return nil, err
	}
	// LCM is an epsilon-prediction, few-step sampler.
	sched := diffusion.NewLCM(cfg.Steps, cfg.TrainT, cfg.OriginalSteps, 0, cfg.BetaStart, cfg.BetaEnd)
	pipe := &diffusion.Pipeline{
		Scheduler:     sched,
		Backbone:      &mmengine.RefBackbone{Eng: eng, W: w},
		Decoder:       &mmengine.LinearDecoder{Eng: eng, W: w},
		GuidanceScale: cfg.Guidance,
	}
	return &Engine{cfg: cfg, eng: eng, pipe: pipe}, nil
}

// Generate produces RGB pixels for a prompt. The returned slice has length
// Width*Height*Channels (patch-major from the decoder), in roughly [-1, 1].
func (e *Engine) Generate(prompt string, seed int64) ([]float32, error) {
	latent := mmengine.GaussianLatent(e.cfg.latentTokens()*e.cfg.LatentDim, seed)
	cond := mmengine.TextCond(prompt, e.cfg.LatentDim)
	uncond := mmengine.TextCond("", e.cfg.LatentDim)
	return e.pipe.Generate(latent, cond, uncond)
}

func (e *Engine) Close() { e.eng.Close() }
