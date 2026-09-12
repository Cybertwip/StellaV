package audio

import (
	"math"
	"testing"
)

func TestAudioGenerate(t *testing.T) {
	cfg := Default()
	cfg.SamplesPerFrame = 64 // small for a fast test
	cfg.ModelDim = 32
	cfg.FF = 64
	cfg.Steps = 2
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	pcm, err := e.Generate("a calm piano melody", 0.05, 42)
	if err != nil {
		t.Fatal(err)
	}
	frames := e.LatentFrames(0.05)
	want := frames * cfg.SamplesPerFrame * cfg.Channels
	if len(pcm) != want {
		t.Fatalf("pcm len %d, want %d", len(pcm), want)
	}
	peak := float32(0)
	for i, v := range pcm {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite sample at %d", i)
		}
		if a := float32(math.Abs(float64(v))); a > peak {
			peak = a
		}
	}
	if peak > 1.0001 {
		t.Fatalf("audio not normalized, peak %v", peak)
	}
}
