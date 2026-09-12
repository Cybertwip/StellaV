package compute

import (
	"context"
	"math"
	"testing"
)

func newTestEngine(workers int) *Engine {
	e, err := NewEngine(Options{CPUWorkers: workers, CPUGrain: 8, DisableGPU: true})
	if err != nil {
		panic(err)
	}
	return e
}

func seq(n int, seed uint64) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*6364136223846793005 + 1442695040888963407
		out[i] = float32((float64(s>>11)/float64(1<<53) - 0.5) * 2.0)
	}
	return out
}

func close32(a, b float32, tol float32) bool { return float32(math.Abs(float64(a-b))) <= tol }

func TestMatMulBiasParity(t *testing.T) {
	rows, inner, cols := 5, 7, 4
	lhs := seq(rows*inner, 1)
	rhs := seq(cols*inner, 2)
	bias := seq(cols, 3)
	for _, workers := range []int{1, 4} {
		e := newTestEngine(workers)
		dst := make([]float32, rows*cols)
		if err := e.MatMulBias(context.Background(), dst, lhs, rhs, bias, rows, inner, cols); err != nil {
			t.Fatal(err)
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				var acc float32
				for i := 0; i < inner; i++ {
					acc += lhs[r*inner+i] * rhs[c*inner+i]
				}
				acc += bias[c]
				if !close32(dst[r*cols+c], acc, 1e-4) {
					t.Fatalf("workers=%d dst[%d,%d]=%v want %v", workers, r, c, dst[r*cols+c], acc)
				}
			}
		}
	}
}

func TestAttentionParity(t *testing.T) {
	qRows, kRows, hd := 4, 6, 8
	q := seq(qRows*hd, 10)
	k := seq(kRows*hd, 11)
	v := seq(kRows*hd, 12)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	e := newTestEngine(2)
	out := make([]float32, qRows*hd)
	if err := e.Attention(context.Background(), out, q, k, v, qRows, kRows, hd, scale); err != nil {
		t.Fatal(err)
	}
	// Naive reference.
	for i := 0; i < qRows; i++ {
		scores := make([]float64, kRows)
		mx := math.Inf(-1)
		for j := 0; j < kRows; j++ {
			var dot float64
			for d := 0; d < hd; d++ {
				dot += float64(q[i*hd+d]) * float64(k[j*hd+d])
			}
			scores[j] = dot * float64(scale)
			if scores[j] > mx {
				mx = scores[j]
			}
		}
		var z float64
		for j := range scores {
			scores[j] = math.Exp(scores[j] - mx)
			z += scores[j]
		}
		for d := 0; d < hd; d++ {
			var acc float64
			for j := 0; j < kRows; j++ {
				acc += scores[j] / z * float64(v[j*hd+d])
			}
			if !close32(out[i*hd+d], float32(acc), 1e-4) {
				t.Fatalf("attn out[%d,%d]=%v want %v", i, d, out[i*hd+d], acc)
			}
		}
	}
}

func TestVectorOps(t *testing.T) {
	e := newTestEngine(3)
	ctx := context.Background()
	n := 33
	a := seq(n, 20)
	b := seq(n, 21)

	dst := append([]float32(nil), a...)
	if err := e.SAXPY(ctx, dst, b, 2.0); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if !close32(dst[i], a[i]+2*b[i], 1e-5) {
			t.Fatalf("saxpy[%d]", i)
		}
	}
	add := make([]float32, n)
	if err := e.Add(ctx, add, a, b); err != nil {
		t.Fatal(err)
	}
	sc := make([]float32, n)
	if err := e.Scale(ctx, sc, a, 0.5); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if !close32(add[i], a[i]+b[i], 1e-5) || !close32(sc[i], a[i]*0.5, 1e-5) {
			t.Fatalf("add/scale[%d]", i)
		}
	}
}
