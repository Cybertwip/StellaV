package v7graph

import (
	"fmt"

	"github.com/powerengine/paccel/libpaccel"
)

// matrix is a row-major dense matrix of float32.
type matrix struct {
	rows, cols int
	data       []float32
}

func (m matrix) row(i int) []float32 { return m.data[i*m.cols : i*m.cols+m.cols] }

// LayerWeights holds the dequantized tensors for one decoder block. The
// attention projections are bound as a single fused QKV matrix and the MLP
// projections as a single fused gate-up matrix, both in the V7 [in, out]
// orientation, so the decoder issues one matmul per pair (see graph.go).
type LayerWeights struct {
	InputNorm    []float32 // [hidden]
	PostAttnNorm []float32 // [hidden]
	QKV          matrix    // [hidden, outQ + 2*kv] — q|k|v fused, pre-transposed
	QBias        []float32 // [outQ]  (optional, Qwen2)
	KBias        []float32 // [kv]    (optional)
	VBias        []float32 // [kv]    (optional)
	Wo           matrix    // [outQ, hidden]
	GateUp       matrix    // [hidden, 2*intermediate] — gate|up fused
	Wdown        matrix    // [intermediate, hidden]
	QNorm        []float32 // [head_dim] (optional, Qwen3)
	KNorm        []float32 // [head_dim] (optional, Qwen3)
}

// Weights is the full set of decoded model weights bound from a package.
type Weights struct {
	Embed    matrix // [vocab, hidden]
	Layers   []LayerWeights
	FinalNRM []float32 // [hidden]
	LMHead   matrix    // [hidden, vocab] (pre-transposed)
}

// BindWeights resolves every tensor the V7 graph needs from a decoded package,
// dequantizing on demand. It prefers the fused QKV / gate-up aliases and falls
// back to concatenating the split projections, so both the fused and split V7
// layouts run unchanged.
func BindWeights(pkg *libpaccel.Package, r *Recipe) (*Weights, error) {
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

	w := &Weights{}
	ev, es, err := getAny(libpaccel.V7EmbedName, "model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	if w.Embed, err = asMatrix(ev, es, r.VocabSize, H); err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	w.Layers = make([]LayerWeights, r.NumHiddenLayers)
	for i := 0; i < r.NumHiddenLayers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		lw := LayerWeights{}
		if lw.InputNorm, _, err = get(p + "input_layernorm.weight"); err != nil {
			return nil, err
		}
		if lw.PostAttnNorm, _, err = get(p + "post_attention_layernorm.weight"); err != nil {
			return nil, err
		}
		if lw.QKV, err = bindFusedQKV(has, get, getAny, p, H, outQ, kv); err != nil {
			return nil, err
		}
		if lw.GateUp, err = bindFusedGateUp(has, get, getAny, p, H, inter); err != nil {
			return nil, err
		}
		if lw.Wo, err = bindLinear(getAny, p+"self_attn.o_proj", outQ, H); err != nil {
			return nil, err
		}
		if lw.Wdown, err = bindLinear(getAny, p+"mlp.down_proj", inter, H); err != nil {
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
	lv, ls, err := getAny(libpaccel.V7LMHeadName, "lm_head.weight.t")
	if err != nil {
		t, terr := libpaccel.TransposeMatrix(ev, r.VocabSize, H) // tied head
		if terr != nil {
			return nil, fmt.Errorf("v7: no LM head and tied synthesis failed: %w", terr)
		}
		w.LMHead = matrix{rows: H, cols: r.VocabSize, data: t}
	} else if w.LMHead, err = asMatrix(lv, ls, H, r.VocabSize); err != nil {
		return nil, fmt.Errorf("lm_head: %w", err)
	}
	return w, nil
}

// bindFusedQKV returns the [hidden, outQ+2kv] fused attention weight, preferring
// the fused alias and falling back to column-concatenating the split q/k/v.
func bindFusedQKV(has func(string) bool, get func(string) ([]float32, []uint64, error),
	getAny func(...string) ([]float32, []uint64, error), p string, H, outQ, kv int) (matrix, error) {
	fused := libpaccel.V7Prefix + p + "self_attn." + libpaccel.V7QKVProjSuffix
	if has(fused) {
		v, s, err := get(fused)
		if err != nil {
			return matrix{}, err
		}
		return asMatrix(v, s, H, outQ+2*kv)
	}
	q, err := bindLinear(getAny, p+"self_attn.q_proj", H, outQ)
	if err != nil {
		return matrix{}, err
	}
	k, err := bindLinear(getAny, p+"self_attn.k_proj", H, kv)
	if err != nil {
		return matrix{}, err
	}
	vv, err := bindLinear(getAny, p+"self_attn.v_proj", H, kv)
	if err != nil {
		return matrix{}, err
	}
	return concatCols(H, q, k, vv), nil
}

// bindFusedGateUp returns the [hidden, 2*intermediate] fused MLP weight.
func bindFusedGateUp(has func(string) bool, get func(string) ([]float32, []uint64, error),
	getAny func(...string) ([]float32, []uint64, error), p string, H, inter int) (matrix, error) {
	fused := libpaccel.V7Prefix + p + "mlp." + libpaccel.V7GateUpProjSuffix
	if has(fused) {
		v, s, err := get(fused)
		if err != nil {
			return matrix{}, err
		}
		return asMatrix(v, s, H, 2*inter)
	}
	g, err := bindLinear(getAny, p+"mlp.gate_proj", H, inter)
	if err != nil {
		return matrix{}, err
	}
	u, err := bindLinear(getAny, p+"mlp.up_proj", H, inter)
	if err != nil {
		return matrix{}, err
	}
	return concatCols(H, g, u), nil
}

// concatCols concatenates same-row-count matrices along the column axis,
// reproducing the fused layout from split parts.
func concatCols(rows int, mats ...matrix) matrix {
	total := 0
	for _, m := range mats {
		total += m.cols
	}
	data := make([]float32, rows*total)
	for r := 0; r < rows; r++ {
		off := r * total
		for _, m := range mats {
			copy(data[off:off+m.cols], m.row(r))
			off += m.cols
		}
	}
	return matrix{rows: rows, cols: total, data: data}
}

// bindLinear loads a pre-transposed projection, trying the v7 reserved name
// first then the plain name, and asserts the expected [in, out] shape.
func bindLinear(getAny func(...string) ([]float32, []uint64, error), base string, in, out int) (matrix, error) {
	v, s, err := getAny(libpaccel.V7Prefix+base+".weight.t", base+".weight.t", base+".weight")
	if err != nil {
		return matrix{}, err
	}
	return asMatrix(v, s, in, out)
}

func asMatrix(values []float32, shape []uint64, rows, cols int) (matrix, error) {
	if rows*cols != len(values) {
		return matrix{}, fmt.Errorf("expected %dx%d=%d values, got %d (shape %v)", rows, cols, rows*cols, len(values), shape)
	}
	return matrix{rows: rows, cols: cols, data: values}, nil
}

func decodeRecord(rec libpaccel.Record) ([]float32, []uint64, error) {
	if rec.Compressed() {
		return libpaccel.TurboDequantize(rec.Tensor), rec.Tensor.Shape, nil
	}
	raw := rec.RawPayload
	if rec.Encoding == libpaccel.EncodingRawRLE {
		var err error
		if raw, err = libpaccel.RLEDecode(raw); err != nil {
			return nil, nil, err
		}
	}
	v := libpaccel.DecodeRawTensor(raw, rec.SourceDataType)
	if v == nil {
		return nil, nil, fmt.Errorf("v7: cannot decode tensor %q", rec.Name)
	}
	return v, rec.Shape, nil
}
