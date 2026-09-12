package image

import (
	"math"
	"testing"
)

func TestImageGenerate(t *testing.T) {
	cfg := Default()
	cfg.Width, cfg.Height = 32, 32 // small for a fast test
	cfg.LatentDim = 32
	cfg.FF = 64
	cfg.Steps = 2
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	px, err := e.Generate("a red apple on a table", 7)
	if err != nil {
		t.Fatal(err)
	}
	want := cfg.Width * cfg.Height * cfg.Channels
	if len(px) != want {
		t.Fatalf("pixels len %d, want %d", len(px), want)
	}
	for i, v := range px {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite pixel at %d", i)
		}
	}
}
