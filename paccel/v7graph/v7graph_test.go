package v7graph

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/powerengine/paccel/libpaccel"
)

func synth(n int, seed uint64) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*6364136223846793005 + 1442695040888963407
		u := float64(s>>11) / float64(1<<53)
		out[i] = float32((u - 0.5) * 0.5)
	}
	return out
}

func rawFloatRecord(name string, values []float32, shape []uint64) libpaccel.Record {
	buf := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return libpaccel.Record{
		Name: name, SourceDataType: libpaccel.WireFloat, Encoding: libpaccel.EncodingRaw,
		Shape: shape, SourcePayloadBytes: uint64(len(buf)), RawPayload: buf,
	}
}

func compressedRecord(name string, values []float32, shape []uint64, bits uint32) libpaccel.Record {
	q, err := libpaccel.TurboQuantize(values, shape, bits, libpaccel.DefaultBlockSize, name, false)
	if err != nil {
		panic(err)
	}
	return libpaccel.Record{
		Name: name, SourceDataType: libpaccel.WireFloat, Encoding: libpaccel.EncodingCompressed,
		Shape: shape, SourcePayloadBytes: uint64(len(values)) * 4, Tensor: q,
	}
}

// TestV7FusedPipeline builds a tiny Qwen2-style checkpoint in original (HF)
// orientation, runs AppendV7Aliases to fuse q/k/v and gate/up into the V7 fused
// layout, binds the fused package, and runs a forward pass with attention
// biases — exercising fusion, binding, output slicing, and the decode graph.
func TestV7FusedPipeline(t *testing.T) {
	r := &Recipe{
		Format: libpaccel.V7Format, Version: libpaccel.V7Version, GraphKind: libpaccel.V7GraphKind,
		ModelType: "qwen2", WeightLayout: libpaccel.V7WeightLayoutFused, PackageMode: libpaccel.V7PackageMode,
		Weights: "model.paccel", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 2,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 4, VocabSize: 16,
		RMSNormEps: 1e-6, RopeTheta: 10000, HiddenAct: "silu", TieWordEmbeddings: false,
	}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	H, inter, V := 8, 16, 16
	outQ, kv := r.NumAttentionHeads*r.HeadDim, r.NumKeyValueHeads*r.HeadDim
	seed := uint64(0x51A7)
	nf := func(n int) []float32 { seed += 131; return synth(n, seed) }

	u64 := func(d ...int) []uint64 {
		s := make([]uint64, len(d))
		for i, x := range d {
			s[i] = uint64(x)
		}
		return s
	}
	pkg := &libpaccel.Package{BitWidth: 8, BlockSize: libpaccel.DefaultBlockSize}
	add := func(rec libpaccel.Record) { pkg.Records = append(pkg.Records, rec) }

	// Original HF-orientation weights [out, in].
	add(compressedRecord("model.embed_tokens.weight", nf(V*H), u64(V, H), 8))
	for i := 0; i < r.NumHiddenLayers; i++ {
		p := layerPrefix(i)
		add(rawFloatRecord(p+"input_layernorm.weight", ones(H), u64(H)))
		add(rawFloatRecord(p+"post_attention_layernorm.weight", ones(H), u64(H)))
		add(compressedRecord(p+"self_attn.q_proj.weight", nf(outQ*H), u64(outQ, H), 8))
		add(compressedRecord(p+"self_attn.k_proj.weight", nf(kv*H), u64(kv, H), 8))
		add(compressedRecord(p+"self_attn.v_proj.weight", nf(kv*H), u64(kv, H), 8))
		add(compressedRecord(p+"self_attn.o_proj.weight", nf(H*outQ), u64(H, outQ), 8))
		add(rawFloatRecord(p+"self_attn.q_proj.bias", nf(outQ), u64(outQ)))
		add(rawFloatRecord(p+"self_attn.k_proj.bias", nf(kv), u64(kv)))
		add(rawFloatRecord(p+"self_attn.v_proj.bias", nf(kv), u64(kv)))
		add(compressedRecord(p+"mlp.gate_proj.weight", nf(inter*H), u64(inter, H), 8))
		add(compressedRecord(p+"mlp.up_proj.weight", nf(inter*H), u64(inter, H), 8))
		add(compressedRecord(p+"mlp.down_proj.weight", nf(H*inter), u64(H, inter), 8))
	}
	add(rawFloatRecord("model.norm.weight", ones(H), u64(H)))
	add(compressedRecord("lm_head.weight", nf(V*H), u64(V, H), 8))

	// Fuse into the V7 layout.
	if err := libpaccel.AppendV7Aliases(pkg, 8, libpaccel.DefaultBlockSize); err != nil {
		t.Fatal(err)
	}
	// Confirm the fused aliases were produced.
	names := map[string]bool{}
	for _, rec := range pkg.Records {
		names[rec.Name] = true
	}
	if !names["v7.model.layers.0.self_attn.qkv_proj.weight.t"] {
		t.Fatal("missing fused qkv alias")
	}
	if !names["v7.model.layers.0.mlp.gate_up_proj.weight.t"] {
		t.Fatal("missing fused gate_up alias")
	}

	w, err := BindWeights(pkg, r)
	if err != nil {
		t.Fatal(err)
	}
	if w.Layers[0].QKV.cols != outQ+2*kv {
		t.Fatalf("fused qkv cols = %d, want %d", w.Layers[0].QKV.cols, outQ+2*kv)
	}
	if w.Layers[0].QBias == nil {
		t.Fatal("expected q bias to bind")
	}
	m := &Model{R: r, W: w}
	logits := m.Forward([]int{2, 7, 1, 9})
	if len(logits) != 4 || len(logits[0]) != V {
		t.Fatalf("logits shape = %dx%d, want 4x%d", len(logits), len(logits[0]), V)
	}
	for t0, row := range logits {
		for j, val := range row {
			if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
				t.Fatalf("non-finite logit at [%d][%d]", t0, j)
			}
		}
	}
	if nt := m.NextToken([]int{2, 7, 1, 9}); nt < 0 || nt >= V {
		t.Fatalf("next token %d out of range", nt)
	}

	pw, err := BindPackedWeights(pkg, r)
	if err != nil {
		t.Fatal(err)
	}
	if pw.Layers[0].QKV.tensor == nil {
		t.Fatal("expected packed qkv tensor to stay compressed")
	}
	if packed, dense := pw.ResidentBytes(), w.ResidentBytes(); packed >= dense {
		t.Fatalf("packed resident bytes = %d, want less than dense %d", packed, dense)
	}
	pm := &PackedModel{R: r, W: pw}
	packedLogits := pm.Forward([]int{2, 7, 1, 9})
	for t0 := range logits {
		for j := range logits[t0] {
			if diff := math.Abs(float64(logits[t0][j] - packedLogits[t0][j])); diff > 1e-6 {
				t.Fatalf("packed logit[%d][%d] differs by %g: dense=%v packed=%v",
					t0, j, diff, logits[t0][j], packedLogits[t0][j])
			}
		}
	}
	if got, want := pm.NextToken([]int{2, 7, 1, 9}), m.NextToken([]int{2, 7, 1, 9}); got != want {
		t.Fatalf("packed next token = %d, want %d", got, want)
	}
}

func ones(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func layerPrefix(i int) string { return "model.layers." + itoa(i) + "." }

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}
