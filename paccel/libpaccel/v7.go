package libpaccel

import (
	"fmt"
	"strings"
)

// V7 is the "direct-only" weight layout: linear and LM-head weights are stored
// pre-transposed so a runtime computes y = x · W directly, with no transpose at
// load time and no temporary buffers. Combined with the v6 tiled int4 packing
// this is the optimal route to inference — it is what the V7 transformer_decoder
// graph (see the v7graph package) consumes. Aliases live under reserved names so
// they coexist with, or replace, the original-orientation tensors.
const (
	V7Prefix         = "v7."
	V7EmbedName      = "v7.model.embed_tokens.weight"
	V7LMHeadName     = "v7.lm_head.weight.t"
	V7RecipeFileName = "model.v7.json"
	V7Format         = "power-paccel-direct"
	V7PackageMode    = "direct-only"
	V7GraphKind      = "transformer_decoder"
	V7Version        = 7

	// Weight-layout identifiers. The fused layout is the current, optimal one:
	// the three attention projections (q,k,v) and the two MLP projections
	// (gate,up) are concatenated into a single pre-transposed matrix each, so a
	// decoder issues one matmul per pair instead of three and two. The split and
	// legacy qwen layouts remain readable.
	V7WeightLayoutFused = "v7-transformer-direct-fused-int4"
	V7WeightLayoutSplit = "v7-transformer-direct-only-int4"
	V7WeightLayoutQwen  = "v7-qwen-direct-only-int4"
	V7WeightLayout      = V7WeightLayoutFused // current default

	// Reserved fused-alias name fragments. They are appended to a layer prefix
	// that already ends in "self_attn." or "mlp.", and prefixed with "v7.":
	// e.g. v7.model.layers.0.self_attn.qkv_proj.weight.t.
	V7QKVProjSuffix    = "qkv_proj.weight.t"
	V7GateUpProjSuffix = "gate_up_proj.weight.t"
)

// V7WeightLayoutSupported reports whether a recipe weight-layout string is one
// this implementation can run.
func V7WeightLayoutSupported(layout string) bool {
	switch strings.TrimSpace(layout) {
	case V7WeightLayoutFused, V7WeightLayoutSplit, V7WeightLayoutQwen:
		return true
	default:
		return false
	}
}

// V7AliasPlan describes one alias to synthesize from a source tensor: the
// reserved name to store it under, and whether the source must be transposed.
type V7AliasPlan struct {
	Name      string
	Transpose bool
}

// V7AliasPlansFor returns the aliases to emit for a source tensor. Token
// embeddings are copied verbatim (used as a lookup, not a matmul); the LM head
// and every per-layer linear projection are stored transposed. When the
// checkpoint ties its LM head to the embedding, a transposed head is synthesized
// from the embedding (synthesizeTiedHead = true).
func V7AliasPlansFor(name string, shape []uint64, synthesizeTiedHead bool) []V7AliasPlan {
	if len(shape) != 2 || shape[0] == 0 || shape[1] == 0 {
		return nil
	}
	switch {
	case name == "model.embed_tokens.weight":
		plans := []V7AliasPlan{{Name: V7EmbedName}}
		if synthesizeTiedHead {
			plans = append(plans, V7AliasPlan{Name: V7LMHeadName, Transpose: true})
		}
		return plans
	case name == "lm_head.weight":
		return []V7AliasPlan{{Name: V7LMHeadName, Transpose: true}}
	case strings.HasPrefix(name, "model.layers.") && strings.HasSuffix(name, ".self_attn.qkv_proj.weight"):
		// Already-fused checkpoint (e.g. Phi-style): keep it fused, just transpose.
		return []V7AliasPlan{{Name: V7Prefix + name + ".t", Transpose: true}}
	case strings.HasPrefix(name, "model.layers.") && strings.HasSuffix(name, ".mlp.gate_up_proj.weight"):
		return []V7AliasPlan{{Name: V7Prefix + name + ".t", Transpose: true}}
	case v7IsDirectLinear(name):
		return []V7AliasPlan{{Name: V7Prefix + name + ".t", Transpose: true}}
	default:
		return nil
	}
}

// v7IsDirectLinear reports whether a per-layer weight is one of the seven linear
// projections (attention q/k/v/o, MLP gate/up/down) that the direct graph reads
// transposed.
func v7IsDirectLinear(name string) bool {
	if !strings.HasPrefix(name, "model.layers.") || !strings.HasSuffix(name, ".weight") {
		return false
	}
	for _, marker := range []string{
		".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
		".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
		".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
	} {
		if strings.HasSuffix(name, marker) {
			return true
		}
	}
	return false
}

// TransposeMatrix returns the transpose of a row-major [rows × cols] matrix as a
// [cols × rows] matrix. Transposing the [out, in] HF weight yields the [in, out]
// orientation the direct graph multiplies against.
func TransposeMatrix(values []float32, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 || len(values) != rows*cols {
		return nil, fmt.Errorf("v7 transpose: shape %dx%d does not match %d values", rows, cols, len(values))
	}
	out := make([]float32, len(values))
	for r := 0; r < rows; r++ {
		row := values[r*cols : r*cols+cols]
		for c, v := range row {
			out[c*rows+r] = v
		}
	}
	return out, nil
}

// FuseTransposedOutputMatrices concatenates several framework [out, in] weight
// matrices that share the same input width into a single pre-transposed
// [in, sum(out)] matrix. The result is laid out so that columns 0..out_0 are the
// transpose of part 0, the next out_1 columns are part 1, and so on. Therefore a
// single product y = x · fused yields the concatenation [x·W0^T | x·W1^T | ...],
// which the decoder recovers by slicing the output columns. This is the algebra
// behind fusing q/k/v into one attention matmul and gate/up into one MLP matmul.
//
// For each input index it emits, for each part, that part's column at the input
// index across all its rows — i.e. the transpose of the part — yielding the
// column-concatenated transpose without ever materializing the per-part
// transposes.
func FuseTransposedOutputMatrices(parts [][]float32, shapes [][]uint64) ([]float32, []uint64, error) {
	if len(parts) == 0 || len(parts) != len(shapes) {
		return nil, nil, fmt.Errorf("v7 fuse: %d parts vs %d shapes", len(parts), len(shapes))
	}
	inputs := shapes[0][1]
	var outputs uint64
	for i, s := range shapes {
		if len(s) != 2 || s[1] != inputs || uint64(len(parts[i])) != s[0]*s[1] {
			return nil, nil, fmt.Errorf("v7 fuse: part %d shape %v incompatible with input width %d", i, s, inputs)
		}
		outputs += s[0]
	}
	if inputs == 0 || outputs == 0 {
		return nil, nil, fmt.Errorf("v7 fuse: empty fused matrix")
	}
	out := make([]float32, 0, int(inputs*outputs))
	for input := uint64(0); input < inputs; input++ {
		for p, vals := range parts {
			rows, cols := shapes[p][0], shapes[p][1]
			for row := uint64(0); row < rows; row++ {
				out = append(out, vals[int(row*cols+input)])
			}
		}
	}
	return out, []uint64{inputs, outputs}, nil
}

// recordValues dequantizes a record to float32 regardless of its encoding.
func recordValues(rec Record) ([]float32, []uint64, error) {
	if rec.Compressed() {
		return TurboDequantize(rec.Tensor), rec.Tensor.Shape, nil
	}
	raw := rec.RawPayload
	if rec.Encoding == EncodingRawRLE {
		var err error
		if raw, err = RLEDecode(raw); err != nil {
			return nil, nil, err
		}
	}
	vals := DecodeRawTensor(raw, rec.SourceDataType)
	if vals == nil {
		return nil, nil, fmt.Errorf("v7: cannot decode raw tensor %q (dtype %d)", rec.Name, rec.SourceDataType)
	}
	return vals, rec.Shape, nil
}

// AppendV7Aliases augments a package with the current (fused) V7 direct layout.
//
// Pass 1 fuses, per layer, the q/k/v attention projections into one
// pre-transposed qkv matrix and the gate/up MLP projections into one gate_up
// matrix (see FuseTransposedOutputMatrices); the folded sources are not also
// emitted individually. Pass 2 emits ordinary pre-transposed aliases for
// everything else — the remaining projections (o_proj, down_proj), the
// embedding (verbatim), and the LM head (transposed, or synthesized from a tied
// embedding). Norm and bias vectors are left untouched under their original
// names. Every alias is re-quantized into the v6 tiled int4 layout.
func AppendV7Aliases(pkg *Package, bits, blockSize uint32) error {
	index := make(map[string]Record, len(pkg.Records))
	hasLMHead := false
	for _, rec := range pkg.Records {
		index[rec.Name] = rec
		if rec.Name == "lm_head.weight" {
			hasLMHead = true
		}
	}
	getHF := func(name string) ([]float32, []uint64, error) {
		rec, ok := index[name]
		if !ok {
			return nil, nil, fmt.Errorf("v7: missing source tensor %q", name)
		}
		v, s, err := recordValues(rec)
		if err != nil {
			return nil, nil, err
		}
		return v, NormalizedShape(s, len(v)), nil
	}
	quantAlias := func(name string, values []float32, shape []uint64) (Record, error) {
		tbits := ResolveMixedBitWidth(name, shape, bits)
		q, err := TurboQuantize(values, shape, tbits, blockSize, name, true)
		if err != nil {
			return Record{}, err
		}
		return Record{
			Name: name, SourceDataType: WireFloat, Encoding: EncodingCompressed,
			Shape: shape, SourcePayloadBytes: uint64(len(values)) * 4, Tensor: q,
		}, nil
	}

	consumed := map[string]bool{} // sources folded into a fused alias
	var aliases []Record

	// Pass 1: per-layer fusion, anchored on q_proj and gate_proj.
	fuse := func(anchorSuffix, fusedSuffix string, parts ...string) error {
		for _, rec := range pkg.Records {
			if !strings.HasPrefix(rec.Name, "model.layers.") || !strings.HasSuffix(rec.Name, anchorSuffix) {
				continue
			}
			prefix := strings.TrimSuffix(rec.Name, parts[0])
			names := make([]string, len(parts))
			ok := true
			for i, p := range parts {
				names[i] = prefix + p
				if _, present := index[names[i]]; !present {
					ok = false
				}
			}
			if !ok {
				continue
			}
			vals := make([][]float32, len(names))
			shapes := make([][]uint64, len(names))
			for i, n := range names {
				v, s, err := getHF(n)
				if err != nil {
					return err
				}
				if len(s) != 2 {
					ok = false
					break
				}
				vals[i], shapes[i] = v, s
			}
			if !ok {
				continue
			}
			fused, fshape, err := FuseTransposedOutputMatrices(vals, shapes)
			if err != nil {
				return err
			}
			a, err := quantAlias(V7Prefix+prefix+fusedSuffix, fused, fshape)
			if err != nil {
				return err
			}
			aliases = append(aliases, a)
			for _, n := range names {
				consumed[n] = true
			}
		}
		return nil
	}
	if err := fuse(".self_attn.q_proj.weight", V7QKVProjSuffix,
		"q_proj.weight", "k_proj.weight", "v_proj.weight"); err != nil {
		return err
	}
	if err := fuse(".mlp.gate_proj.weight", V7GateUpProjSuffix,
		"gate_proj.weight", "up_proj.weight"); err != nil {
		return err
	}

	// Pass 2: individual aliases for everything not folded into a fusion.
	for _, rec := range pkg.Records {
		if consumed[rec.Name] {
			continue
		}
		shape := rec.Shape
		if rec.Compressed() {
			shape = rec.Tensor.Shape
		}
		plans := V7AliasPlansFor(rec.Name, shape, !hasLMHead)
		if len(plans) == 0 {
			continue
		}
		values, vshape, err := recordValues(rec)
		if err != nil {
			return err
		}
		vshape = NormalizedShape(vshape, len(values))
		for _, plan := range plans {
			av, ashape := values, vshape
			if plan.Transpose {
				if len(vshape) != 2 {
					return fmt.Errorf("v7 alias %s: transpose needs a rank-2 tensor", plan.Name)
				}
				if av, err = TransposeMatrix(values, int(vshape[0]), int(vshape[1])); err != nil {
					return err
				}
				ashape = []uint64{vshape[1], vshape[0]}
			}
			a, err := quantAlias(plan.Name, av, ashape)
			if err != nil {
				return err
			}
			aliases = append(aliases, a)
		}
	}

	pkg.Records = append(pkg.Records, aliases...)
	return nil
}
