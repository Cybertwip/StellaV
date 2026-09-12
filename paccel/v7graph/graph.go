package v7graph

import (
	"math"

	"github.com/powerengine/paccel/libpaccel"
)

// Model is a bound, ready-to-run V7 transformer decoder.
type Model struct {
	R *Recipe
	W *Weights
}

// Load reads a model.v7.json recipe and its direct-layout .paccel package and
// binds a runnable model. recipePath and paccelPath are the two files a V7
// "direct-only" snapshot contains.
func Load(recipePath, paccelPath string) (*Model, error) {
	r, err := LoadRecipe(recipePath)
	if err != nil {
		return nil, err
	}
	pkg, err := libpaccel.ReadPackage(paccelPath)
	if err != nil {
		return nil, err
	}
	w, err := BindWeights(pkg, r)
	if err != nil {
		return nil, err
	}
	return &Model{R: r, W: w}, nil
}

// Forward runs the decoder over a token sequence and returns the logits for
// every position, shape [len(tokens)][vocab]. It recomputes the full sequence
// each call (no KV cache) for clarity; the algebra is the optimal-route graph:
// embed -> N×(RMSNorm, GQA+RoPE attention, RMSNorm, SwiGLU) -> RMSNorm -> head.
func (m *Model) Forward(tokens []int) [][]float32 {
	r, w := m.R, m.W
	S := len(tokens)
	H := r.HiddenSize
	hd := r.HeadDim
	nh := r.NumAttentionHeads
	nkv := r.NumKeyValueHeads
	group := nh / nkv
	scale := 1.0 / math.Sqrt(float64(hd))
	eps := float32(r.RMSNormEps)

	// Embedding lookup.
	h := make([][]float32, S)
	for t, id := range tokens {
		if id < 0 || id >= w.Embed.rows {
			id = 0
		}
		h[t] = append([]float32(nil), w.Embed.row(id)...)
	}

	for _, ly := range w.Layers {
		// --- attention block ---
		// One fused matmul produces q|k|v; slice the output columns and add the
		// optional projection biases. This is the V7 fused-layout attention path.
		x := rmsNormRows(h, ly.InputNorm, eps)
		outQ := nh * hd
		kvw := nkv * hd
		qkv := matmulRows(x, ly.QKV) // [S, outQ + 2*kvw]
		q := sliceCols(qkv, 0, outQ)
		k := sliceCols(qkv, outQ, kvw)
		v := sliceCols(qkv, outQ+kvw, kvw)
		addBias(q, ly.QBias)
		addBias(k, ly.KBias)
		addBias(v, ly.VBias)

		attn := make([][]float32, S)
		for t := range attn {
			attn[t] = make([]float32, nh*hd)
		}
		for head := 0; head < nh; head++ {
			kvh := head / group
			for t := 0; t < S; t++ {
				qh := headVec(q[t], head, hd)
				if ly.QNorm != nil {
					qh = rmsNormVec(qh, ly.QNorm, eps)
				}
				qh = rope(qh, t, r.RopeTheta)
				// causal scores over u <= t
				scores := make([]float64, t+1)
				maxS := math.Inf(-1)
				for u := 0; u <= t; u++ {
					kh := headVec(k[u], kvh, hd)
					if ly.KNorm != nil {
						kh = rmsNormVec(kh, ly.KNorm, eps)
					}
					kh = rope(kh, u, r.RopeTheta)
					dot := 0.0
					for d := 0; d < hd; d++ {
						dot += float64(qh[d]) * float64(kh[d])
					}
					s := dot * scale
					scores[u] = s
					if s > maxS {
						maxS = s
					}
				}
				// softmax
				var z float64
				for u := range scores {
					scores[u] = math.Exp(scores[u] - maxS)
					z += scores[u]
				}
				// weighted sum of V
				out := attn[t][head*hd : head*hd+hd]
				for u := 0; u <= t; u++ {
					wgt := float32(scores[u] / z)
					vv := headVec(v[u], kvh, hd)
					for d := 0; d < hd; d++ {
						out[d] += wgt * vv[d]
					}
				}
			}
		}
		o := matmulRows(attn, ly.Wo) // [S, H]
		for t := 0; t < S; t++ {
			for c := 0; c < H; c++ {
				h[t][c] += o[t][c]
			}
		}

		// --- MLP block (SwiGLU) ---
		// One fused matmul produces gate|up; slice and apply SiLU(gate) * up.
		x2 := rmsNormRows(h, ly.PostAttnNorm, eps)
		inter := r.IntermediateSize
		gateUp := matmulRows(x2, ly.GateUp) // [S, 2*inter]
		gu := make([][]float32, S)
		for t := 0; t < S; t++ {
			gu[t] = make([]float32, inter)
			gate := gateUp[t][0:inter]
			up := gateUp[t][inter : 2*inter]
			for j := 0; j < inter; j++ {
				gu[t][j] = silu(gate[j]) * up[j]
			}
		}
		d := matmulRows(gu, ly.Wdown) // [S, H]
		for t := 0; t < S; t++ {
			for c := 0; c < H; c++ {
				h[t][c] += d[t][c]
			}
		}
	}

	h = rmsNormRows(h, w.FinalNRM, eps)
	return matmulRows(h, w.LMHead) // [S, vocab]
}

// NextToken returns the greedy argmax token for the final position.
func (m *Model) NextToken(tokens []int) int {
	logits := m.Forward(tokens)
	last := logits[len(logits)-1]
	best, bestV := 0, float32(math.Inf(-1))
	for j, v := range last {
		if v > bestV {
			bestV, best = v, j
		}
	}
	return best
}

// --- primitive ops (mirror the validated reference) ---

// matmulRows computes y = x · W where x is [S, in] and W is the V7-oriented
// [in, out] matrix, giving [S, out]. This is the direct, transpose-free product.
func matmulRows(x [][]float32, wt matrix) [][]float32 {
	S := len(x)
	out := make([][]float32, S)
	for i := 0; i < S; i++ {
		yi := make([]float32, wt.cols)
		xi := x[i]
		for kk := 0; kk < wt.rows; kk++ {
			a := xi[kk]
			if a == 0 {
				continue
			}
			wrow := wt.row(kk)
			for j := 0; j < wt.cols; j++ {
				yi[j] += a * wrow[j]
			}
		}
		out[i] = yi
	}
	return out
}

func rmsNormRows(x [][]float32, weight []float32, eps float32) [][]float32 {
	out := make([][]float32, len(x))
	for t, row := range x {
		out[t] = rmsNormVec(row, weight, eps)
	}
	return out
}

func rmsNormVec(row, weight []float32, eps float32) []float32 {
	var ms float64
	for _, v := range row {
		ms += float64(v) * float64(v)
	}
	ms /= float64(len(row))
	inv := float32(1.0 / math.Sqrt(ms+float64(eps)))
	out := make([]float32, len(row))
	for k := range row {
		out[k] = row[k] * inv * weight[k]
	}
	return out
}

func headVec(row []float32, head, hd int) []float32 {
	return append([]float32(nil), row[head*hd:head*hd+hd]...)
}

// rope applies the NeoX rotate-half rotary embedding to one head vector.
func rope(vec []float32, pos int, theta float64) []float32 {
	hd := len(vec)
	half := hd / 2
	out := make([]float32, hd)
	for i := 0; i < half; i++ {
		freq := 1.0 / math.Pow(theta, float64(2*i)/float64(hd))
		ang := float64(pos) * freq
		c := float32(math.Cos(ang))
		s := float32(math.Sin(ang))
		x1, x2 := vec[i], vec[i+half]
		out[i] = x1*c - x2*s
		out[i+half] = x2*c + x1*s
	}
	if hd%2 == 1 {
		out[hd-1] = vec[hd-1]
	}
	return out
}

// sliceCols extracts a [S, width] column band starting at begin from a [S, *]
// matrix — used to recover q, k, v (and gate, up) from a fused matmul output.
func sliceCols(x [][]float32, begin, width int) [][]float32 {
	out := make([][]float32, len(x))
	for t := range x {
		out[t] = append([]float32(nil), x[t][begin:begin+width]...)
	}
	return out
}

// addBias adds a per-output-column bias vector to every row, if present.
func addBias(x [][]float32, bias []float32) {
	if bias == nil {
		return
	}
	for t := range x {
		row := x[t]
		n := len(row)
		if len(bias) < n {
			n = len(bias)
		}
		for j := 0; j < n; j++ {
			row[j] += bias[j]
		}
	}
}

func silu(x float32) float32 {
	return x / (1.0 + float32(math.Exp(-float64(x))))
}
