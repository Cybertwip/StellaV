package v7graph

import (
	"fmt"
	"math"

	"github.com/powerengine/paccel/libpaccel"
)

// packedMatrix keeps a direct-layout matrix resident in its PAccel quantized
// form when possible. Raw or synthesized fallback matrices are held as dense
// float32, preserving compatibility with older packages.
type packedMatrix struct {
	rows, cols int
	dense      []float32
	tensor     *libpaccel.QuantizedTensor
}

func packedFromDense(m matrix) packedMatrix {
	return packedMatrix{rows: m.rows, cols: m.cols, dense: m.data}
}

func packedFromRecord(rec libpaccel.Record, rows, cols int) (packedMatrix, error) {
	shape := rec.Shape
	if rec.Compressed() {
		shape = rec.Tensor.Shape
	}
	if len(shape) != 2 || shape[0] != uint64(rows) || shape[1] != uint64(cols) {
		return packedMatrix{}, fmt.Errorf("expected packed %dx%d tensor for %q, got shape %v", rows, cols, rec.Name, shape)
	}
	if rec.Compressed() {
		q := rec.Tensor
		return packedMatrix{rows: rows, cols: cols, tensor: &q}, nil
	}
	values, _, err := decodeRecord(rec)
	if err != nil {
		return packedMatrix{}, err
	}
	if len(values) != rows*cols {
		return packedMatrix{}, fmt.Errorf("expected packed %dx%d tensor for %q, got %d values", rows, cols, rec.Name, len(values))
	}
	return packedMatrix{rows: rows, cols: cols, dense: values}, nil
}

func (m packedMatrix) row(i int) []float32 {
	out := make([]float32, m.cols)
	if i < 0 || i >= m.rows {
		return out
	}
	if m.tensor != nil {
		libpaccel.DequantizeSpan(*m.tensor, i*m.cols, out)
		return out
	}
	copy(out, m.dense[i*m.cols:i*m.cols+m.cols])
	return out
}

func (m packedMatrix) matmulRows(x [][]float32) [][]float32 {
	S := len(x)
	out := make([][]float32, S)
	for i := range out {
		out[i] = make([]float32, m.cols)
	}
	if m.tensor == nil {
		for kk := 0; kk < m.rows; kk++ {
			wrow := m.dense[kk*m.cols : kk*m.cols+m.cols]
			for i := 0; i < S; i++ {
				a := x[i][kk]
				if a == 0 {
					continue
				}
				yi := out[i]
				for j := 0; j < m.cols; j++ {
					yi[j] += a * wrow[j]
				}
			}
		}
		return out
	}

	wrow := make([]float32, m.cols)
	for kk := 0; kk < m.rows; kk++ {
		used := false
		for i := 0; i < S; i++ {
			if x[i][kk] != 0 {
				used = true
				break
			}
		}
		if !used {
			continue
		}
		libpaccel.DequantizeSpan(*m.tensor, kk*m.cols, wrow)
		for i := 0; i < S; i++ {
			a := x[i][kk]
			if a == 0 {
				continue
			}
			yi := out[i]
			for j := 0; j < m.cols; j++ {
				yi[j] += a * wrow[j]
			}
		}
	}
	return out
}

func (m packedMatrix) residentBytes() uint64 {
	if m.tensor != nil {
		return uint64(len(m.tensor.Packed)) + uint64(len(m.tensor.Scales))*4
	}
	return uint64(len(m.dense)) * 4
}

func (m matrix) residentBytes() uint64 {
	return uint64(len(m.data)) * 4
}

// PackedLayerWeights is the V8-style binding for one decoder block: matmul
// weights remain packed, while small vectors are decoded once.
type PackedLayerWeights struct {
	InputNorm    []float32
	PostAttnNorm []float32
	QKV          packedMatrix
	QBias        []float32
	KBias        []float32
	VBias        []float32
	Wo           packedMatrix
	GateUp       packedMatrix
	Wdown        packedMatrix
	QNorm        []float32
	KNorm        []float32
}

// PackedWeights is the full V8-style direct binding. It uses the same recipe
// and aliases as V7, but keeps compressed matrices compressed in memory.
type PackedWeights struct {
	Embed    packedMatrix
	Layers   []PackedLayerWeights
	FinalNRM []float32
	LMHead   packedMatrix
}

// PackedModel runs the direct transformer graph with packed resident matrices.
type PackedModel struct {
	R *Recipe
	W *PackedWeights
}

// LoadPacked reads a V7 direct recipe/package and binds the packed runtime path.
func LoadPacked(recipePath, paccelPath string) (*PackedModel, error) {
	r, err := LoadRecipe(recipePath)
	if err != nil {
		return nil, err
	}
	pkg, err := libpaccel.ReadPackage(paccelPath)
	if err != nil {
		return nil, err
	}
	w, err := BindPackedWeights(pkg, r)
	if err != nil {
		return nil, err
	}
	return &PackedModel{R: r, W: w}, nil
}

// LoadV8 is the named entry point for the packed direct route. V8 currently
// reuses the V7 recipe and fused aliases; the behavioral difference is that
// packed matrices stay packed all the way into matmul.
func LoadV8(recipePath, paccelPath string) (*PackedModel, error) {
	return LoadPacked(recipePath, paccelPath)
}

// BindPackedWeights resolves every tensor the direct graph needs, preferring
// compressed fused aliases and falling back to dense compatibility paths only
// when a package lacks the V7 fused records.
func BindPackedWeights(pkg *libpaccel.Package, r *Recipe) (*PackedWeights, error) {
	idx := make(map[string]libpaccel.Record, len(pkg.Records))
	for _, rec := range pkg.Records {
		idx[rec.Name] = rec
	}
	has := func(name string) bool { _, ok := idx[name]; return ok }
	get := func(name string) ([]float32, []uint64, error) {
		rec, ok := idx[name]
		if !ok {
			return nil, nil, fmt.Errorf("v7: missing tensor %q", name)
		}
		return decodeRecord(rec)
	}
	getAny := func(names ...string) ([]float32, []uint64, error) {
		var last error
		for _, n := range names {
			if has(n) {
				return get(n)
			}
			last = fmt.Errorf("v7: missing tensor %q", n)
		}
		return nil, nil, last
	}
	getRecordAny := func(names ...string) (libpaccel.Record, error) {
		var last error
		for _, n := range names {
			if rec, ok := idx[n]; ok {
				return rec, nil
			}
			last = fmt.Errorf("v7: missing tensor %q", n)
		}
		return libpaccel.Record{}, last
	}
	optional := func(name string) []float32 {
		if v, _, err := get(name); err == nil {
			return v
		}
		return nil
	}

	H := r.HiddenSize
	outQ := r.NumAttentionHeads * r.HeadDim
	kv := r.NumKeyValueHeads * r.HeadDim
	inter := r.IntermediateSize

	w := &PackedWeights{}
	er, err := getRecordAny(libpaccel.V7EmbedName, "model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	if w.Embed, err = packedFromRecord(er, r.VocabSize, H); err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	w.Layers = make([]PackedLayerWeights, r.NumHiddenLayers)
	for i := 0; i < r.NumHiddenLayers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		lw := PackedLayerWeights{}
		if lw.InputNorm, _, err = get(p + "input_layernorm.weight"); err != nil {
			return nil, err
		}
		if lw.PostAttnNorm, _, err = get(p + "post_attention_layernorm.weight"); err != nil {
			return nil, err
		}
		if lw.QKV, err = bindPackedFusedQKV(has, get, getAny, getRecordAny, p, H, outQ, kv); err != nil {
			return nil, err
		}
		if lw.GateUp, err = bindPackedFusedGateUp(has, get, getAny, getRecordAny, p, H, inter); err != nil {
			return nil, err
		}
		if lw.Wo, err = bindPackedLinear(getRecordAny, p+"self_attn.o_proj", outQ, H); err != nil {
			return nil, err
		}
		if lw.Wdown, err = bindPackedLinear(getRecordAny, p+"mlp.down_proj", inter, H); err != nil {
			return nil, err
		}
		lw.QBias = optional(p + "self_attn.q_proj.bias")
		lw.KBias = optional(p + "self_attn.k_proj.bias")
		lw.VBias = optional(p + "self_attn.v_proj.bias")
		lw.QNorm = optional(p + "self_attn.q_norm.weight")
		lw.KNorm = optional(p + "self_attn.k_norm.weight")
		w.Layers[i] = lw
	}

	if w.FinalNRM, _, err = get("model.norm.weight"); err != nil {
		return nil, err
	}
	lr, err := getRecordAny(libpaccel.V7LMHeadName, "lm_head.weight.t")
	if err != nil {
		ev, _, derr := decodeRecord(er)
		if derr != nil {
			return nil, derr
		}
		t, terr := libpaccel.TransposeMatrix(ev, r.VocabSize, H)
		if terr != nil {
			return nil, fmt.Errorf("v7: no LM head and tied synthesis failed: %w", terr)
		}
		w.LMHead = packedMatrix{rows: H, cols: r.VocabSize, dense: t}
	} else if w.LMHead, err = packedFromRecord(lr, H, r.VocabSize); err != nil {
		return nil, fmt.Errorf("lm_head: %w", err)
	}
	return w, nil
}

func bindPackedFusedQKV(has func(string) bool, get func(string) ([]float32, []uint64, error),
	getAny func(...string) ([]float32, []uint64, error),
	getRecordAny func(...string) (libpaccel.Record, error), p string, H, outQ, kv int) (packedMatrix, error) {
	fused := libpaccel.V7Prefix + p + "self_attn." + libpaccel.V7QKVProjSuffix
	if has(fused) {
		rec, err := getRecordAny(fused)
		if err != nil {
			return packedMatrix{}, err
		}
		return packedFromRecord(rec, H, outQ+2*kv)
	}
	dense, err := bindFusedQKV(has, get, getAny, p, H, outQ, kv)
	if err != nil {
		return packedMatrix{}, err
	}
	return packedFromDense(dense), nil
}

func bindPackedFusedGateUp(has func(string) bool, get func(string) ([]float32, []uint64, error),
	getAny func(...string) ([]float32, []uint64, error),
	getRecordAny func(...string) (libpaccel.Record, error), p string, H, inter int) (packedMatrix, error) {
	fused := libpaccel.V7Prefix + p + "mlp." + libpaccel.V7GateUpProjSuffix
	if has(fused) {
		rec, err := getRecordAny(fused)
		if err != nil {
			return packedMatrix{}, err
		}
		return packedFromRecord(rec, H, 2*inter)
	}
	dense, err := bindFusedGateUp(has, get, getAny, p, H, inter)
	if err != nil {
		return packedMatrix{}, err
	}
	return packedFromDense(dense), nil
}

func bindPackedLinear(getRecordAny func(...string) (libpaccel.Record, error), base string, in, out int) (packedMatrix, error) {
	rec, err := getRecordAny(libpaccel.V7Prefix+base+".weight.t", base+".weight.t", base+".weight")
	if err != nil {
		return packedMatrix{}, err
	}
	return packedFromRecord(rec, in, out)
}

// ResidentBytes returns the bytes occupied by decoded dense weights.
func (w *Weights) ResidentBytes() uint64 {
	if w == nil {
		return 0
	}
	total := w.Embed.residentBytes() + uint64(len(w.FinalNRM))*4 + w.LMHead.residentBytes()
	for _, ly := range w.Layers {
		total += uint64(len(ly.InputNorm)+len(ly.PostAttnNorm)+len(ly.QBias)+len(ly.KBias)+len(ly.VBias)+len(ly.QNorm)+len(ly.KNorm)) * 4
		total += ly.QKV.residentBytes() + ly.Wo.residentBytes() + ly.GateUp.residentBytes() + ly.Wdown.residentBytes()
	}
	return total
}

// ResidentBytes returns the bytes occupied by the packed direct binding.
func (w *PackedWeights) ResidentBytes() uint64 {
	if w == nil {
		return 0
	}
	total := w.Embed.residentBytes() + uint64(len(w.FinalNRM))*4 + w.LMHead.residentBytes()
	for _, ly := range w.Layers {
		total += uint64(len(ly.InputNorm)+len(ly.PostAttnNorm)+len(ly.QBias)+len(ly.KBias)+len(ly.VBias)+len(ly.QNorm)+len(ly.KNorm)) * 4
		total += ly.QKV.residentBytes() + ly.Wo.residentBytes() + ly.GateUp.residentBytes() + ly.Wdown.residentBytes()
	}
	return total
}

// Forward runs the decoder over a token sequence, using packed resident
// matrices for every projection and head multiply.
func (m *PackedModel) Forward(tokens []int) [][]float32 {
	r, w := m.R, m.W
	S := len(tokens)
	H := r.HiddenSize
	hd := r.HeadDim
	nh := r.NumAttentionHeads
	nkv := r.NumKeyValueHeads
	group := nh / nkv
	scale := 1.0 / math.Sqrt(float64(hd))
	eps := float32(r.RMSNormEps)

	h := make([][]float32, S)
	for t, id := range tokens {
		if id < 0 || id >= w.Embed.rows {
			id = 0
		}
		h[t] = w.Embed.row(id)
	}

	for _, ly := range w.Layers {
		x := rmsNormRows(h, ly.InputNorm, eps)
		outQ := nh * hd
		kvw := nkv * hd
		qkv := ly.QKV.matmulRows(x)
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
				var z float64
				for u := range scores {
					scores[u] = math.Exp(scores[u] - maxS)
					z += scores[u]
				}
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
		o := ly.Wo.matmulRows(attn)
		for t := 0; t < S; t++ {
			for c := 0; c < H; c++ {
				h[t][c] += o[t][c]
			}
		}

		x2 := rmsNormRows(h, ly.PostAttnNorm, eps)
		inter := r.IntermediateSize
		gateUp := ly.GateUp.matmulRows(x2)
		gu := make([][]float32, S)
		for t := 0; t < S; t++ {
			gu[t] = make([]float32, inter)
			gate := gateUp[t][0:inter]
			up := gateUp[t][inter : 2*inter]
			for j := 0; j < inter; j++ {
				gu[t][j] = silu(gate[j]) * up[j]
			}
		}
		d := ly.Wdown.matmulRows(gu)
		for t := 0; t < S; t++ {
			for c := 0; c < H; c++ {
				h[t][c] += d[t][c]
			}
		}
	}

	h = rmsNormRows(h, w.FinalNRM, eps)
	return w.LMHead.matmulRows(h)
}

// NextToken returns the greedy argmax token for the final position.
func (m *PackedModel) NextToken(tokens []int) int {
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
