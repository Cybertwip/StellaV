// Package video is the standalone CogVideoX-style text-to-video engine. It
// composes the compute engine, the Euler v-prediction diffusion sampler, and (a
// reference for) the spatio-temporal DiT backbone and 3D VAE decoder. The
// production pipeline is T5 text encoder -> 3D diffusion transformer -> 3D VAE
// decoder; those modules load from a .paccel via libpaccel and plug into the
// diffusion seams used here.
package video

import (
	"github.com/powerengine/paccel/compute"
	"github.com/powerengine/paccel/diffusion"
	"github.com/powerengine/paccel/mmengine"
)

// Config describes the video model and the sampler.
type Config struct {
	Width, Height int     // output pixel dimensions per frame
	Frames        int     // output frames
	Downsample    int     // spatial VAE downsample (8)
	TimeDownsample int    // temporal VAE downsample (4)
	LatentDim     int     // latent channels (model width)
	FF            int     // feed-forward width
	Channels      int     // output channels (3 = RGB)
	Steps         int     // sampler steps
	Guidance      float64 // CFG scale
	TrainT        int
	BetaStart     float64
	BetaEnd       float64
	SigmaMin      float64
	SigmaMax      float64
}

// Default returns a short, low-res CogVideoX-style configuration (reference-sized).
func Default() Config {
	return Config{
		Width: 256, Height: 256, Frames: 9, Downsample: 8, TimeDownsample: 4,
		LatentDim: 64, FF: 256, Channels: 3, Steps: 10, Guidance: 6.0, TrainT: 1000,
		BetaStart: 0.00085, BetaEnd: 0.012, SigmaMin: 0.002, SigmaMax: 700.0,
	}
}

func (c Config) latentFrames() int {
	f := c.Frames / c.TimeDownsample
	if f < 1 {
		f = 1
	}
	return f
}

func (c Config) latentTokens() int {
	return c.latentFrames() * (c.Width / c.Downsample) * (c.Height / c.Downsample)
}

func (c Config) patchPixels() int {
	return c.Downsample * c.Downsample * c.Channels
}

// Engine is a ready-to-run text-to-video generator.
type Engine struct {
	cfg  Config
	eng  *compute.Engine
	pipe *diffusion.Pipeline
}

func New(cfg Config) (*Engine, error) {
	return NewWithWeights(cfg, mmengine.RandomWeights(cfg.LatentDim, cfg.FF, cfg.patchPixels(), 23))
}

func NewWithWeights(cfg Config, w mmengine.Weights) (*Engine, error) {
	eng, err := compute.NewEngine(compute.Options{DisableGPU: true})
	if err != nil {
		return nil, err
	}
	sched := diffusion.NewEulerKarras(cfg.Steps, cfg.TrainT, cfg.BetaStart, cfg.BetaEnd, cfg.SigmaMin, cfg.SigmaMax)
	pipe := &diffusion.Pipeline{
		Scheduler:     sched,
		Backbone:      &mmengine.RefBackbone{Eng: eng, W: w},
		Decoder:       &mmengine.LinearDecoder{Eng: eng, W: w},
		GuidanceScale: cfg.Guidance,
	}
	return &Engine{cfg: cfg, eng: eng, pipe: pipe}, nil
}

// Generate produces RGB video pixels for a prompt. The returned slice has length
// latentTokens*patchPixels (frame-major then patch-major from the decoder).
func (e *Engine) Generate(prompt string, seed int64) ([]float32, error) {
	latent := mmengine.GaussianLatent(e.cfg.latentTokens()*e.cfg.LatentDim, seed)
	cond := mmengine.TextCond(prompt, e.cfg.LatentDim)
	uncond := mmengine.TextCond("", e.cfg.LatentDim)
	return e.pipe.Generate(latent, cond, uncond)
}

func (e *Engine) Close() { e.eng.Close() }
