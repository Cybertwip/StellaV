// Package mmengine provides the shared building blocks the standalone audio,
// image, and video engines run on: a reference diffusion-transformer backbone and
// a linear decoder, both implemented on the compute engine. They satisfy the
// diffusion.Backbone and diffusion.Decoder interfaces, so the production backbones
// (Stable Audio 3 DiT, SVD UNet, CogVideoX transformer) and VAE decoders — loaded
// from a .paccel via libpaccel — are drop-in replacements behind the same seams.
package mmengine

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"github.com/powerengine/paccel/compute"
)

// Weights holds the dense parameters of one reference DiT block plus the decoder.
// All matrices are row-major [out, in] (the compute-engine MatMulBias convention).
type Weights struct {
	Dim   int // model / latent channel width
	FF    int // feed-forward inner width
	Wqkv  []float32 // [3*Dim, Dim]
	Wo    []float32 // [Dim, Dim]
	W1    []float32 // [FF, Dim]
	B1    []float32 // [FF]
	W2    []float32 // [Dim, FF]
	B2    []float32 // [Dim]
	WDec  []float32 // decoder [OutDim, Dim]
	BDec  []float32 // [OutDim]
	OutDim int
}

// RandomWeights builds small random weights for the reference backbone/decoder.
// Production weights replace these via libpaccel; the shapes are identical.
func RandomWeights(dim, ff, outDim int, seed int64) Weights {
	r := rand.New(rand.NewSource(seed))
	mk := func(n int, scale float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(r.NormFloat64() * scale)
		}
		return out
	}
	s := 1.0 / math.Sqrt(float64(dim))
	return Weights{
		Dim: dim, FF: ff, OutDim: outDim,
		Wqkv: mk(3*dim*dim, s), Wo: mk(dim*dim, s),
		W1: mk(ff*dim, s), B1: make([]float32, ff),
		W2: mk(dim*ff, 1.0/math.Sqrt(float64(ff))), B2: make([]float32, dim),
		WDec: mk(outDim*dim, s), BDec: make([]float32, outDim),
	}
}

// RefBackbone is a single-block diffusion transformer denoiser: timestep + cross
// conditioning injection, fused self-attention, and a GELU feed-forward, all on
// the compute engine. It predicts a model output (epsilon or v) the same shape as
// the latent.
type RefBackbone struct {
	Eng *compute.Engine
	W   Weights
}

func (b *RefBackbone) Denoise(latent []float32, timestep float64, cond []float32) ([]float32, error) {
	dim := b.W.Dim
	if dim <= 0 || len(latent)%dim != 0 {
		return nil, fmt.Errorf("mmengine: latent length %d not a multiple of dim %d", len(latent), dim)
	}
	tokens := len(latent) / dim
	ctx := context.Background()

	// x = latent + timestep embedding (+ broadcast cross conditioning).
	tEmb := fourierEmbed(timestep, dim)
	x := make([]float32, len(latent))
	for tkn := 0; tkn < tokens; tkn++ {
		for d := 0; d < dim; d++ {
			v := latent[tkn*dim+d] + tEmb[d]
			if len(cond) == dim {
				v += cond[d]
			}
			x[tkn*dim+d] = v
		}
	}

	// Self-attention: project to q|k|v, run fused attention, project out, residual.
	qkv := make([]float32, tokens*3*dim)
	if err := b.Eng.MatMulBias(ctx, qkv, x, b.W.Wqkv, nil, tokens, dim, 3*dim); err != nil {
		return nil, err
	}
	q := make([]float32, tokens*dim)
	k := make([]float32, tokens*dim)
	v := make([]float32, tokens*dim)
	for tkn := 0; tkn < tokens; tkn++ {
		copy(q[tkn*dim:], qkv[tkn*3*dim:tkn*3*dim+dim])
		copy(k[tkn*dim:], qkv[tkn*3*dim+dim:tkn*3*dim+2*dim])
		copy(v[tkn*dim:], qkv[tkn*3*dim+2*dim:tkn*3*dim+3*dim])
	}
	attn := make([]float32, tokens*dim)
	scale := float32(1.0 / math.Sqrt(float64(dim)))
	if err := b.Eng.Attention(ctx, attn, q, k, v, tokens, tokens, dim, scale); err != nil {
		return nil, err
	}
	proj := make([]float32, tokens*dim)
	if err := b.Eng.MatMulBias(ctx, proj, attn, b.W.Wo, nil, tokens, dim, dim); err != nil {
		return nil, err
	}
	if err := b.Eng.Add(ctx, x, x, proj); err != nil {
		return nil, err
	}

	// Feed-forward: GELU(x W1 + b1) W2 + b2, residual.
	h := make([]float32, tokens*b.W.FF)
	if err := b.Eng.MatMulBias(ctx, h, x, b.W.W1, b.W.B1, tokens, dim, b.W.FF); err != nil {
		return nil, err
	}
	gelu(h)
	ff := make([]float32, tokens*dim)
	if err := b.Eng.MatMulBias(ctx, ff, h, b.W.W2, b.W.B2, tokens, b.W.FF, dim); err != nil {
		return nil, err
	}
	if err := b.Eng.Add(ctx, x, x, ff); err != nil {
		return nil, err
	}
	return x, nil
}

// LinearDecoder maps each latent token [Dim] to OutDim output samples through a
// single biased matmul — a stand-in for the VAE / pretransform decoder.
type LinearDecoder struct {
	Eng *compute.Engine
	W   Weights
}

func (d *LinearDecoder) Decode(latent []float32) ([]float32, error) {
	dim := d.W.Dim
	if dim <= 0 || len(latent)%dim != 0 {
		return nil, fmt.Errorf("mmengine: decode latent length %d not a multiple of dim %d", len(latent), dim)
	}
	tokens := len(latent) / dim
	out := make([]float32, tokens*d.W.OutDim)
	if err := d.Eng.MatMulBias(context.Background(), out, latent, d.W.WDec, d.W.BDec, tokens, dim, d.W.OutDim); err != nil {
		return nil, err
	}
	return out, nil
}

// fourierEmbed maps a scalar timestep to a [dim] sinusoidal embedding.
func fourierEmbed(t float64, dim int) []float32 {
	out := make([]float32, dim)
	half := dim / 2
	if half < 1 {
		half = 1
	}
	logStep := math.Log(10000) / float64(half)
	for i := 0; i < half; i++ {
		freq := math.Exp(-logStep * float64(i))
		out[i] = float32(math.Sin(t * freq))
		if half+i < dim {
			out[half+i] = float32(math.Cos(t * freq))
		}
	}
	return out
}

// gelu applies the tanh GELU approximation in place.
func gelu(x []float32) {
	const c = 0.7978845608028654 // sqrt(2/pi)
	for i, v := range x {
		v64 := float64(v)
		x[i] = float32(0.5 * v64 * (1.0 + math.Tanh(c*(v64+0.044715*v64*v64*v64))))
	}
}
