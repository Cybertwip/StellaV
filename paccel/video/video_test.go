package video

import (
	"math"
	"testing"
)

func TestVideoGenerate(t *testing.T) {
	cfg := Default()
	cfg.Width, cfg.Height = 32, 32 // small for a fast test
	cfg.Frames = 4
	cfg.LatentDim = 32
	cfg.FF = 64
	cfg.Steps = 2
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	px, err := e.Generate("a bird flying over the ocean", 3)
	if err != nil {
		t.Fatal(err)
	}
	want := cfg.latentTokens() * cfg.patchPixels()
	if len(px) != want {
		t.Fatalf("video len %d, want %d", len(px), want)
	}
	for i, v := range px {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite at %d", i)
		}
	}
}
