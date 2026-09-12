package libpaccel

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// synthWeights returns a deterministic pseudo-random float32 slice spanning a
// realistic dynamic range so block scales vary across the tensor.
func synthWeights(n int) []float32 {
	out := make([]float32, n)
	state := uint64(0x9e3779b97f4a7c15)
	for i := range out {
		state = state*6364136223846793005 + 1442695040888963407
		u := float64(state>>11) / float64(1<<53) // [0,1)
		out[i] = float32((u - 0.5) * 6.0)        // ~[-3, 3)
	}
	return out
}

func TestFp16RoundTrip(t *testing.T) {
	for _, v := range []float32{0, 1, -1, 0.5, -2.25, 65504, 1e-4, -1e-4, 3.1415927} {
		got := Fp16ToFloat32(Float32ToFp16(v))
		// fp16 has ~3-4 significant digits; allow proportional tolerance.
		tol := float32(math.Abs(float64(v))) * 0.01
		if tol < 1e-6 {
			tol = 1e-6
		}
		if math.Abs(float64(got-v)) > float64(tol) {
			t.Errorf("fp16 round-trip %v -> %v (tol %v)", v, got, tol)
		}
	}
}

func TestPackUnpackLinear(t *testing.T) {
	for _, bits := range []uint32{2, 3, 4, 5, 8} {
		maxCode := byte((1 << bits) - 1)
		codes := make([]byte, 1000)
		for i := range codes {
			codes[i] = byte(i) & maxCode
		}
		packed := PackLowBits(codes, bits)
		back := UnpackLowBits(packed, len(codes), bits)
		for i := range codes {
			if codes[i] != back[i] {
				t.Fatalf("bits=%d index=%d: %d != %d", bits, i, codes[i], back[i])
			}
		}
	}
}

func TestQuantizeDequantizeParity(t *testing.T) {
	values := synthWeights(4096)
	for _, bits := range []uint32{2, 4, 8} {
		q, err := TurboQuantize(values, []uint64{uint64(len(values))}, bits, DefaultBlockSize, "block.weight", false)
		if err != nil {
			t.Fatalf("bits=%d: %v", bits, err)
		}
		deq := TurboDequantize(q)
		if len(deq) != len(values) {
			t.Fatalf("bits=%d: length %d != %d", bits, len(deq), len(values))
		}
		// Per-element error must stay within half the element's block scale,
		// which is the theoretical bound for round-to-nearest quantization.
		for i := range values {
			block := i / int(DefaultBlockSize)
			tol := q.Scales[block]*0.5 + 1e-5
			if math.Abs(float64(values[i]-deq[i])) > float64(tol) {
				t.Fatalf("bits=%d index=%d: |%v-%v| exceeds %v", bits, i, values[i], deq[i], tol)
			}
		}
	}
}

func TestTileInt4RoundTrip(t *testing.T) {
	// Rank-2, inner dim a multiple of 32, eligible for the v6 tiled int4 layout.
	rows, cols := uint64(8), uint64(64)
	values := synthWeights(int(rows * cols))
	q, err := TurboQuantize(values, []uint64{rows, cols}, 4, DefaultBlockSize, "model.layers.0.power_t", true)
	if err != nil {
		t.Fatal(err)
	}
	if q.Layout != PackedLayoutOutputTileInt4 {
		t.Fatalf("expected tiled layout, got %d", q.Layout)
	}
	deq := TurboDequantize(q)
	for i := range values {
		block := i / int(DefaultBlockSize)
		tol := q.Scales[block]*0.5 + 1e-5
		if math.Abs(float64(values[i]-deq[i])) > float64(tol) {
			t.Fatalf("tile index=%d: |%v-%v| exceeds %v", i, values[i], deq[i], tol)
		}
	}
}

func TestDequantizeSpanMatchesFullDequantize(t *testing.T) {
	rows, cols := uint64(16), uint64(64)
	values := synthWeights(int(rows * cols))
	q, err := TurboQuantize(values, []uint64{rows, cols}, 4, DefaultBlockSize, "v7.model.layers.0.self_attn.qkv_proj.weight.t", true)
	if err != nil {
		t.Fatal(err)
	}
	full := TurboDequantize(q)
	got := make([]float32, cols)
	DequantizeSpan(q, int(7*cols), got)
	for i := range got {
		if got[i] != full[int(7*cols)+i] {
			t.Fatalf("span[%d] = %v, want %v", i, got[i], full[int(7*cols)+i])
		}
	}
	if v := DequantizeValue(q, int(12*cols+3)); v != full[int(12*cols+3)] {
		t.Fatalf("value = %v, want %v", v, full[int(12*cols+3)])
	}
}

func TestI8RoundTrip(t *testing.T) {
	values := synthWeights(512)
	payload := QuantizeF32ToI8(values)
	deq, err := DequantizeI8ToF32(payload)
	if err != nil {
		t.Fatal(err)
	}
	maxAbs := float32(0)
	for _, v := range values {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}
	tol := maxAbs/127.0*0.5 + 1e-5
	for i := range values {
		if math.Abs(float64(values[i]-deq[i])) > float64(tol) {
			t.Fatalf("i8 index=%d: |%v-%v| exceeds %v", i, values[i], deq[i], tol)
		}
	}
}

func TestRLERoundTrip(t *testing.T) {
	// Highly repetitive payload (RLE should win); 8-byte elements like int64.
	raw := make([]byte, 0, 8000)
	for i := 0; i < 1000; i++ {
		v := byte(i / 100) // long runs
		for j := 0; j < 8; j++ {
			raw = append(raw, v)
		}
	}
	enc := RLEEncode(raw, 8)
	if enc == nil {
		t.Fatal("expected RLE to encode repetitive payload")
	}
	dec, err := RLEDecode(enc)
	if err != nil {
		t.Fatal(err)
	}
	if len(dec) != len(raw) {
		t.Fatalf("rle length %d != %d", len(dec), len(raw))
	}
	for i := range raw {
		if raw[i] != dec[i] {
			t.Fatalf("rle byte %d mismatch", i)
		}
	}
}

func TestPackageRoundTrip(t *testing.T) {
	values := synthWeights(2048)
	q, err := TurboQuantize(values, []uint64{2048}, 4, DefaultBlockSize, "block.weight", false)
	if err != nil {
		t.Fatal(err)
	}
	pkg := &Package{
		BitWidth:  4,
		BlockSize: DefaultBlockSize,
		ModelSize: 12345,
		Records: []Record{{
			Name:               "block.weight",
			SourceDataType:     WireFloat,
			Encoding:           EncodingCompressed,
			Shape:              []uint64{2048},
			SourcePayloadBytes: 2048 * 4,
			Tensor:             q,
		}},
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "test.paccel")
	if err := WritePackage(path, pkg); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	got, err := ReadPackage(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Version != LatestPackageVersion || got.ModelSize != 12345 || len(got.Records) != 1 {
		t.Fatalf("header mismatch: %+v", got)
	}
	rec := got.Records[0]
	if rec.Name != "block.weight" || !rec.Compressed() {
		t.Fatalf("record mismatch: %+v", rec)
	}
	// Codes survive the disk round-trip; scales are fp16 so allow tolerance.
	deqOrig := TurboDequantize(q)
	deqRead := TurboDequantize(rec.Tensor)
	for i := range deqOrig {
		if math.Abs(float64(deqOrig[i]-deqRead[i])) > float64(math.Abs(float64(deqOrig[i])))*0.02+1e-4 {
			t.Fatalf("package round-trip index %d: %v != %v", i, deqOrig[i], deqRead[i])
		}
	}
}
