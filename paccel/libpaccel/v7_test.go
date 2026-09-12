package libpaccel

import (
	"math"
	"testing"
)

func randf(n int, seed uint64) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*6364136223846793005 + 1442695040888963407
		out[i] = float32((float64(s>>11)/float64(1<<53) - 0.5) * 2.0)
	}
	return out
}

// projRef computes y = x · W^T for an HF-orientation weight W [out, in].
func projRef(x []float32, S, in int, W []float32, out int) []float32 {
	y := make([]float32, S*out)
	for s := 0; s < S; s++ {
		for o := 0; o < out; o++ {
			var acc float64
			for i := 0; i < in; i++ {
				acc += float64(x[s*in+i]) * float64(W[o*in+i])
			}
			y[s*out+o] = float32(acc)
		}
	}
	return y
}

// matFused computes y = x · Wf for a fused V7 matrix Wf [in, out].
func matFused(x []float32, S, in int, Wf []float32, out int) []float32 {
	y := make([]float32, S*out)
	for s := 0; s < S; s++ {
		for o := 0; o < out; o++ {
			var acc float64
			for i := 0; i < in; i++ {
				acc += float64(x[s*in+i]) * float64(Wf[i*out+o])
			}
			y[s*out+o] = float32(acc)
		}
	}
	return y
}

// TestFuseTransposedEquivalence verifies the core V7 fusion identity: fusing
// q/k/v (column-concatenated transpose) and slicing the single matmul output
// reproduces the three separate projections exactly.
func TestFuseTransposedEquivalence(t *testing.T) {
	S, in, outQ, kv := 3, 6, 8, 4
	x := randf(S*in, 1)
	q := randf(outQ*in, 2)
	k := randf(kv*in, 3)
	v := randf(kv*in, 4)

	yq := projRef(x, S, in, q, outQ)
	yk := projRef(x, S, in, k, kv)
	yv := projRef(x, S, in, v, kv)

	fused, shape, err := FuseTransposedOutputMatrices(
		[][]float32{q, k, v},
		[][]uint64{{uint64(outQ), uint64(in)}, {uint64(kv), uint64(in)}, {uint64(kv), uint64(in)}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if shape[0] != uint64(in) || shape[1] != uint64(outQ+2*kv) {
		t.Fatalf("fused shape %v, want [%d %d]", shape, in, outQ+2*kv)
	}
	out := outQ + 2*kv
	f := matFused(x, S, in, fused, out)

	check := func(name string, ref []float32, begin, width int) {
		for s := 0; s < S; s++ {
			for j := 0; j < width; j++ {
				got := f[s*out+begin+j]
				want := ref[s*width+j]
				if math.Abs(float64(got-want)) > 1e-5 {
					t.Fatalf("%s[%d][%d]: %v != %v", name, s, j, got, want)
				}
			}
		}
	}
	check("q", yq, 0, outQ)
	check("k", yk, outQ, kv)
	check("v", yv, outQ+kv, kv)
}

// TestAppendV7AliasesFused checks AppendV7Aliases fuses q/k/v and gate/up into
// the reserved fused names and does not also emit the split projections.
func TestAppendV7AliasesFused(t *testing.T) {
	H, kv, inter, V := 8, 4, 16, 16
	mk := func(name string, n int, shape ...uint64) Record {
		q, err := TurboQuantize(randf(n, uint64(len(name))), shape, 8, DefaultBlockSize, name, false)
		if err != nil {
			t.Fatal(err)
		}
		return Record{Name: name, SourceDataType: WireFloat, Encoding: EncodingCompressed,
			Shape: shape, SourcePayloadBytes: uint64(n) * 4, Tensor: q}
	}
	pkg := &Package{BitWidth: 8, BlockSize: DefaultBlockSize}
	pkg.Records = append(pkg.Records,
		mk("model.embed_tokens.weight", V*H, uint64(V), uint64(H)),
		mk("model.layers.0.self_attn.q_proj.weight", H*H, uint64(H), uint64(H)),
		mk("model.layers.0.self_attn.k_proj.weight", kv*H, uint64(kv), uint64(H)),
		mk("model.layers.0.self_attn.v_proj.weight", kv*H, uint64(kv), uint64(H)),
		mk("model.layers.0.mlp.gate_proj.weight", inter*H, uint64(inter), uint64(H)),
		mk("model.layers.0.mlp.up_proj.weight", inter*H, uint64(inter), uint64(H)),
	)
	if err := AppendV7Aliases(pkg, 4, DefaultBlockSize); err != nil {
		t.Fatal(err)
	}
	names := map[string]Record{}
	for _, rec := range pkg.Records {
		names[rec.Name] = rec
	}
	qkv, ok := names["v7.model.layers.0.self_attn.qkv_proj.weight.t"]
	if !ok {
		t.Fatal("missing fused qkv alias")
	}
	if qkv.Tensor.Shape[0] != uint64(H) || qkv.Tensor.Shape[1] != uint64(H+2*kv) {
		t.Fatalf("fused qkv shape %v, want [%d %d]", qkv.Tensor.Shape, H, H+2*kv)
	}
	if _, ok := names["v7.model.layers.0.mlp.gate_up_proj.weight.t"]; !ok {
		t.Fatal("missing fused gate_up alias")
	}
	// Split aliases must NOT be present when fused (default).
	if _, ok := names["v7.model.layers.0.self_attn.q_proj.weight.t"]; ok {
		t.Fatal("split q alias should be folded into the fused alias")
	}
}
