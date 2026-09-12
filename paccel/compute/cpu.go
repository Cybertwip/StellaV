package compute

import (
	"context"
	"math"
	"sync"
)

func (e *Engine) cpuSAXPY(ctx context.Context, dst, x []float32, alpha float32) error {
	return e.parallelFloat32(ctx, len(dst), func(start, end int) {
		saxpyKernel(dst[start:end], x[start:end], alpha)
	})
}

func (e *Engine) cpuAdd(ctx context.Context, dst, a, b []float32) error {
	return e.parallelFloat32(ctx, len(dst), func(start, end int) {
		addKernel(dst[start:end], a[start:end], b[start:end])
	})
}

func (e *Engine) cpuScale(ctx context.Context, dst, src []float32, scale float32) error {
	return e.parallelFloat32(ctx, len(dst), func(start, end int) {
		scaleKernel(dst[start:end], src[start:end], scale)
	})
}

func (e *Engine) cpuMatMulBias(ctx context.Context, dst, lhs, rhs, bias []float32, rows, inner, cols int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	workers := e.opts.CPUWorkers
	if workers < 1 {
		workers = 1
	}
	if workers > rows {
		workers = rows
	}
	if workers <= 1 || rows <= 1 {
		matMulBiasKernel(dst, lhs, rhs, bias, rows, inner, cols)
		return ctx.Err()
	}

	var wg sync.WaitGroup
	wg.Add(workers)
	for worker := 0; worker < workers; worker++ {
		start := worker * rows / workers
		end := (worker + 1) * rows / workers
		go func() {
			defer wg.Done()
			if ctx.Err() == nil {
				matMulBiasKernel(dst[start*cols:end*cols], lhs[start*inner:end*inner], rhs, bias, end-start, inner, cols)
			}
		}()
	}
	wg.Wait()
	return ctx.Err()
}

// cpuAttention computes fused softmax(scale*q*k^T)*v for a slice of query rows.
// Each query row is independent, so the work fans out across CPUWorkers.
func (e *Engine) cpuAttention(ctx context.Context, out, q, k, v []float32, qRows, kRows, headDim int, scale float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	workers := e.opts.CPUWorkers
	if workers < 1 {
		workers = 1
	}
	if workers > qRows {
		workers = qRows
	}
	if workers <= 1 || qRows <= 1 {
		attentionRows(out, q, k, v, qRows, kRows, headDim, scale)
		return ctx.Err()
	}

	var wg sync.WaitGroup
	wg.Add(workers)
	for worker := 0; worker < workers; worker++ {
		start := worker * qRows / workers
		end := (worker + 1) * qRows / workers
		go func() {
			defer wg.Done()
			if ctx.Err() == nil {
				attentionRows(
					out[start*headDim:end*headDim],
					q[start*headDim:end*headDim],
					k, v, end-start, kRows, headDim, scale)
			}
		}()
	}
	wg.Wait()
	return ctx.Err()
}

// attentionRows computes fused softmax(scale*q*k^T)*v for qRows query rows into
// out. out, q are [qRows,headDim]; k, v are [kRows,headDim].
func attentionRows(out, q, k, v []float32, qRows, kRows, headDim int, scale float32) {
	scores := make([]float32, kRows)
	for i := 0; i < qRows; i++ {
		qi := q[i*headDim : i*headDim+headDim]
		maxScore := float32(-1e30)
		for j := 0; j < kRows; j++ {
			kj := k[j*headDim : j*headDim+headDim]
			var acc float32
			d := 0
			for ; d+3 < headDim; d += 4 {
				acc += qi[d]*kj[d] + qi[d+1]*kj[d+1] + qi[d+2]*kj[d+2] + qi[d+3]*kj[d+3]
			}
			for ; d < headDim; d++ {
				acc += qi[d] * kj[d]
			}
			acc *= scale
			scores[j] = acc
			if acc > maxScore {
				maxScore = acc
			}
		}
		var sumExp float32
		for j := 0; j < kRows; j++ {
			ex := math32Exp(scores[j] - maxScore)
			scores[j] = ex
			sumExp += ex
		}
		invSum := float32(1) / sumExp
		oi := out[i*headDim : i*headDim+headDim]
		for d := 0; d < headDim; d++ {
			oi[d] = 0
		}
		for j := 0; j < kRows; j++ {
			w := scores[j] * invSum
			if w == 0 {
				continue
			}
			vj := v[j*headDim : j*headDim+headDim]
			for d := 0; d < headDim; d++ {
				oi[d] += w * vj[d]
			}
		}
	}
}

func math32Exp(x float32) float32 { return float32(math.Exp(float64(x))) }

func (e *Engine) parallelFloat32(ctx context.Context, n int, fn func(start, end int)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	workers := e.opts.CPUWorkers
	if workers < 1 {
		workers = 1
	}
	if e.opts.CPUGrain > 0 {
		chunks := (n + e.opts.CPUGrain - 1) / e.opts.CPUGrain
		if chunks > 0 && chunks < workers {
			workers = chunks
		}
	}
	if workers <= 1 || n < e.opts.CPUGrain {
		fn(0, n)
		return ctx.Err()
	}

	var wg sync.WaitGroup
	wg.Add(workers)
	for worker := 0; worker < workers; worker++ {
		start := worker * n / workers
		end := (worker + 1) * n / workers
		go func() {
			defer wg.Done()
			if ctx.Err() == nil {
				fn(start, end)
			}
		}()
	}
	wg.Wait()
	return ctx.Err()
}

func saxpyKernel(dst, x []float32, alpha float32) {
	for i := range dst {
		dst[i] += alpha * x[i]
	}
}

func addKernel(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func scaleKernel(dst, src []float32, scale float32) {
	for i := range dst {
		dst[i] = src[i] * scale
	}
}

// matMulBiasKernel computes dst[r,c] = lhs[r,:] dot rhs[c,:] + bias[c], with
// lhs [rows,inner], rhs [cols,inner] (row-major weight), dst [rows,cols].
func matMulBiasKernel(dst, lhs, rhs, bias []float32, rows, inner, cols int) {
	for r := 0; r < rows; r++ {
		lrow := lhs[r*inner : r*inner+inner]
		drow := dst[r*cols : r*cols+cols]
		for c := 0; c < cols; c++ {
			wrow := rhs[c*inner : c*inner+inner]
			i := 0
			var s0, s1, s2, s3 float32
			for ; i+3 < inner; i += 4 {
				s0 += lrow[i+0] * wrow[i+0]
				s1 += lrow[i+1] * wrow[i+1]
				s2 += lrow[i+2] * wrow[i+2]
				s3 += lrow[i+3] * wrow[i+3]
			}
			acc := s0 + s1 + s2 + s3
			for ; i < inner; i++ {
				acc += lrow[i] * wrow[i]
			}
			if bias != nil {
				acc += bias[c]
			}
			drow[c] = acc
		}
	}
}
