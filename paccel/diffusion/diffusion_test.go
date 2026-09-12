package diffusion

import (
	"math"
	"testing"
)

func rnd(n int, seed uint64) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*6364136223846793005 + 1442695040888963407
		out[i] = float32((float64(s>>11)/float64(1<<53) - 0.5) * 2.0)
	}
	return out
}

func TestKarrasSchedule(t *testing.T) {
	sig := KarrasSigmas(10, 700.0, 0.002)
	if len(sig) != 10 {
		t.Fatalf("len %d", len(sig))
	}
	for i := 1; i < len(sig); i++ {
		if sig[i] >= sig[i-1] {
			t.Fatalf("sigmas not strictly decreasing at %d: %v >= %v", i, sig[i], sig[i-1])
		}
	}
	if math.Abs(sig[0]-700.0) > 1e-6 || math.Abs(sig[9]-0.002) > 1e-6 {
		t.Fatalf("endpoints %v %v", sig[0], sig[9])
	}
}

// TestEulerRecoversX0 checks the v-prediction -> x0 algebra: with the model
// output that corresponds to a clean sample x0, the final Euler step returns x0.
func TestEulerRecoversX0(t *testing.T) {
	s := NewEulerKarras(1, 1000, 0.00085, 0.012, 0.002, 700.0)
	sigma := s.sigmas[0]
	denom := sigma*sigma + 1.0
	x0 := rnd(64, 1)
	sample := rnd(64, 2)
	// Solve for the v that makes predOriginal == x0.
	v := make([]float32, len(x0))
	for i := range v {
		v[i] = float32((float64(sample[i])/denom - float64(x0[i])) * math.Sqrt(denom) / sigma)
	}
	out := s.Step(v, sample, 0)
	for i := range x0 {
		if math.Abs(float64(out[i]-x0[i])) > 1e-3 {
			t.Fatalf("euler x0[%d]: %v != %v", i, out[i], x0[i])
		}
	}
}

// TestLCMRecoversX0 checks the epsilon -> x0 algebra of a single LCM step.
func TestLCMRecoversX0(t *testing.T) {
	s := NewLCM(1, 1000, 50, 0, 0.00085, 0.012)
	tIdx := s.timesteps[0]
	a := s.alphasCumprod[tIdx]
	sqA, sqB := math.Sqrt(a), math.Sqrt(1-a)
	x0 := rnd(64, 3)
	eps := rnd(64, 4)
	xt := make([]float32, len(x0))
	for i := range xt {
		xt[i] = float32(sqA*float64(x0[i]) + sqB*float64(eps[i]))
	}
	out := s.Step(eps, xt, 0) // single step: returns x0
	for i := range x0 {
		if math.Abs(float64(out[i]-x0[i])) > 1e-3 {
			t.Fatalf("lcm x0[%d]: %v != %v", i, out[i], x0[i])
		}
	}
}

func TestCFG(t *testing.T) {
	uncond := []float32{0, 1, 2}
	cond := []float32{2, 2, 2}
	out := CFG(uncond, cond, 3.0) // uncond + 3*(cond-uncond)
	want := []float32{6, 4, 2}
	for i := range want {
		if math.Abs(float64(out[i]-want[i])) > 1e-6 {
			t.Fatalf("cfg[%d]=%v want %v", i, out[i], want[i])
		}
	}
	if got := CFG(uncond, cond, 1.0); &got[0] != &cond[0] {
		t.Fatal("scale<=1 should return cond unchanged")
	}
}

// zeroBackbone predicts no update; identityDecoder returns the latent.
type zeroBackbone struct{}

func (zeroBackbone) Denoise(latent []float32, _ float64, _ []float32) ([]float32, error) {
	return make([]float32, len(latent)), nil
}

type identityDecoder struct{}

func (identityDecoder) Decode(latent []float32) ([]float32, error) { return latent, nil }

func TestPipelineRuns(t *testing.T) {
	p := &Pipeline{
		Scheduler:     NewLCM(4, 1000, 50, 0, 0.00085, 0.012),
		Backbone:      zeroBackbone{},
		Decoder:       identityDecoder{},
		GuidanceScale: 1.0,
	}
	latent := rnd(128, 5)
	out, err := p.Generate(latent, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 128 {
		t.Fatalf("len %d", len(out))
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite at %d", i)
		}
	}
}
