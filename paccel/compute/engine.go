package compute

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Options controls CPU/GPU cooperative execution.
type Options struct {
	CPUWorkers          int
	CPUGrain            int
	DisableGPU          bool
	RequireGPU          bool
	UseAllGPUs          bool
	PreferLowPowerGPU   bool
	PreferGPURegistryID uint64
	GPUFraction         float64
}

// DefaultOptions keeps a CPU tail active so large work runs cooperatively.
func DefaultOptions() Options {
	return Options{
		CPUWorkers:  runtime.GOMAXPROCS(0),
		CPUGrain:    32768,
		UseAllGPUs:  true,
		GPUFraction: 0.70,
	}
}

// gpuBackend is the interface a native GPU backend implements. This standalone
// build ships no GPU backend (see gpu_stub.go); the engine runs CPU-only.
type gpuBackend interface {
	Device() Device
	Close()
	SAXPY(ctx context.Context, dst, x []float32, alpha float32) error
	Add(ctx context.Context, dst, a, b []float32) error
	Scale(ctx context.Context, dst, src []float32, scale float32) error
	MatMulBias(ctx context.Context, dst, lhs, rhs, bias []float32, rows, inner, cols int) error
	Attention(ctx context.Context, out, q, k, v []float32, qRows, kRows, headDim int, scale float32) error
}

// Engine owns the selected compute backends.
type Engine struct {
	opts   Options
	cpu    Device
	gpus   []gpuBackend
	gpuErr error
}

func NewEngine(opts Options) (*Engine, error) {
	opts = normalizeOptions(opts)
	engine := &Engine{opts: opts, cpu: cpuDevice()}

	if !opts.DisableGPU {
		gpus, err := newMetalBackends(opts)
		if err != nil {
			engine.gpuErr = err
			if opts.RequireGPU {
				return nil, err
			}
		} else {
			engine.gpus = gpus
		}
	} else if opts.RequireGPU {
		return nil, fmt.Errorf("%w: GPU disabled by options", ErrEmptyDevice)
	}
	if opts.RequireGPU && len(engine.gpus) == 0 {
		return nil, fmt.Errorf("%w: no GPU backend in this build", ErrEmptyDevice)
	}
	return engine, nil
}

func normalizeOptions(opts Options) Options {
	defaults := DefaultOptions()
	if opts.CPUWorkers <= 0 {
		opts.CPUWorkers = defaults.CPUWorkers
	}
	if opts.CPUGrain <= 0 {
		opts.CPUGrain = defaults.CPUGrain
	}
	if opts.GPUFraction <= 0 || math.IsNaN(opts.GPUFraction) || math.IsInf(opts.GPUFraction, 0) {
		opts.GPUFraction = defaults.GPUFraction
	}
	if opts.GPUFraction > 1 {
		opts.GPUFraction = 1
	}
	return opts
}

func (e *Engine) Close() {
	if e == nil {
		return
	}
	for _, gpu := range e.gpus {
		gpu.Close()
	}
	e.gpus = nil
}

func (e *Engine) GPUError() error {
	if e == nil {
		return nil
	}
	return e.gpuErr
}

// Plan splits n work items across CPU and any GPUs. CPU-only here.
func (e *Engine) Plan(n int) Plan {
	if e == nil || n <= 0 {
		return Plan{}
	}
	if len(e.gpus) == 0 || n == 1 {
		return Plan{Total: n, Segments: []Segment{{Device: e.cpu, Start: 0, End: n}}}
	}
	activeGPUs := len(e.gpus)
	if activeGPUs > n-1 {
		activeGPUs = n - 1
	}
	gpuN := int(float64(n) * e.opts.GPUFraction)
	if gpuN < activeGPUs {
		gpuN = activeGPUs
	}
	if gpuN >= n {
		gpuN = n - 1
	}
	segments := make([]Segment, 0, activeGPUs+1)
	start := 0
	for i := 0; i < activeGPUs; i++ {
		gpu := e.gpus[i]
		end := gpuN * (i + 1) / activeGPUs
		if end > start {
			segments = append(segments, Segment{Device: gpu.Device(), Start: start, End: end})
			start = end
		}
	}
	if gpuN < n {
		segments = append(segments, Segment{Device: e.cpu, Start: gpuN, End: n})
	}
	return Plan{Total: n, Segments: segments}
}

// SAXPY computes dst[i] += alpha*x[i].
func (e *Engine) SAXPY(ctx context.Context, dst, x []float32, alpha float32) error {
	if e == nil {
		return ErrEmptyDevice
	}
	if err := checkSameLength("SAXPY", len(dst), namedLen{"x", len(x)}); err != nil {
		return err
	}
	if len(dst) == 0 {
		return nil
	}
	plan := e.Plan(len(dst))
	return e.runPlan(ctx, plan,
		func(s Segment) error { return e.cpuSAXPY(ctx, dst[s.Start:s.End], x[s.Start:s.End], alpha) },
		func(s Segment) error {
			return e.gpuForSegment(s).SAXPY(ctx, dst[s.Start:s.End], x[s.Start:s.End], alpha)
		})
}

// Add computes dst[i] = a[i]+b[i].
func (e *Engine) Add(ctx context.Context, dst, a, b []float32) error {
	if e == nil {
		return ErrEmptyDevice
	}
	if err := checkSameLength("Add", len(dst), namedLen{"a", len(a)}, namedLen{"b", len(b)}); err != nil {
		return err
	}
	if len(dst) == 0 {
		return nil
	}
	plan := e.Plan(len(dst))
	return e.runPlan(ctx, plan,
		func(s Segment) error { return e.cpuAdd(ctx, dst[s.Start:s.End], a[s.Start:s.End], b[s.Start:s.End]) },
		func(s Segment) error {
			return e.gpuForSegment(s).Add(ctx, dst[s.Start:s.End], a[s.Start:s.End], b[s.Start:s.End])
		})
}

// Scale computes dst[i] = src[i]*scale.
func (e *Engine) Scale(ctx context.Context, dst, src []float32, scale float32) error {
	if e == nil {
		return ErrEmptyDevice
	}
	if err := checkSameLength("Scale", len(dst), namedLen{"src", len(src)}); err != nil {
		return err
	}
	if len(dst) == 0 {
		return nil
	}
	plan := e.Plan(len(dst))
	return e.runPlan(ctx, plan,
		func(s Segment) error { return e.cpuScale(ctx, dst[s.Start:s.End], src[s.Start:s.End], scale) },
		func(s Segment) error {
			return e.gpuForSegment(s).Scale(ctx, dst[s.Start:s.End], src[s.Start:s.End], scale)
		})
}

// MatMulBias computes dst[row,col] = lhs[row,:] dot rhs[col,:] + bias[col].
// lhs is [rows,inner], rhs is [cols,inner], dst is [rows,cols].
func (e *Engine) MatMulBias(ctx context.Context, dst, lhs, rhs, bias []float32, rows, inner, cols int) error {
	if e == nil {
		return ErrEmptyDevice
	}
	if rows < 0 || inner < 0 || cols < 0 {
		return fmt.Errorf("%w: MatMulBias dimensions must be non-negative", ErrShapeMismatch)
	}
	if err := checkSameLength("MatMulBias", rows*cols, namedLen{"dst", len(dst)}); err != nil {
		return err
	}
	if err := checkSameLength("MatMulBias", rows*inner, namedLen{"lhs", len(lhs)}); err != nil {
		return err
	}
	if err := checkSameLength("MatMulBias", cols*inner, namedLen{"rhs", len(rhs)}); err != nil {
		return err
	}
	if bias != nil {
		if err := checkSameLength("MatMulBias", cols, namedLen{"bias", len(bias)}); err != nil {
			return err
		}
	}
	if rows == 0 || inner == 0 || cols == 0 {
		return nil
	}
	plan := e.Plan(rows)
	return e.runPlan(ctx, plan,
		func(s Segment) error {
			return e.cpuMatMulBias(ctx, dst[s.Start*cols:s.End*cols], lhs[s.Start*inner:s.End*inner], rhs, bias, s.Len(), inner, cols)
		},
		func(s Segment) error {
			return e.gpuForSegment(s).MatMulBias(ctx, dst[s.Start*cols:s.End*cols], lhs[s.Start*inner:s.End*inner], rhs, bias, s.Len(), inner, cols)
		})
}

// Attention computes out[i,:] = softmax_j(scale * q[i,:] . k[j,:]) . v[j,:].
func (e *Engine) Attention(ctx context.Context, out, q, k, v []float32, qRows, kRows, headDim int, scale float32) error {
	if e == nil {
		return ErrEmptyDevice
	}
	if qRows < 0 || kRows < 0 || headDim < 0 {
		return fmt.Errorf("%w: Attention dimensions must be non-negative", ErrShapeMismatch)
	}
	if err := checkSameLength("Attention", qRows*headDim, namedLen{"q", len(q)}); err != nil {
		return err
	}
	if err := checkSameLength("Attention", kRows*headDim, namedLen{"k", len(k)}); err != nil {
		return err
	}
	if err := checkSameLength("Attention", kRows*headDim, namedLen{"v", len(v)}); err != nil {
		return err
	}
	if err := checkSameLength("Attention", qRows*headDim, namedLen{"out", len(out)}); err != nil {
		return err
	}
	if qRows == 0 || kRows == 0 || headDim == 0 {
		return nil
	}
	plan := e.Plan(qRows)
	return e.runPlan(ctx, plan,
		func(s Segment) error {
			return e.cpuAttention(ctx, out[s.Start*headDim:s.End*headDim], q[s.Start*headDim:s.End*headDim], k, v, s.Len(), kRows, headDim, scale)
		},
		func(s Segment) error {
			return e.gpuForSegment(s).Attention(ctx, out[s.Start*headDim:s.End*headDim], q[s.Start*headDim:s.End*headDim], k, v, s.Len(), kRows, headDim, scale)
		})
}

func (e *Engine) runPlan(ctx context.Context, plan Plan, cpuFn, gpuFn func(Segment) error) error {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	errCh := make(chan error, len(plan.Segments))
	var wg sync.WaitGroup
	for _, segment := range plan.Segments {
		if segment.Len() == 0 {
			continue
		}
		segment := segment
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := ctx.Err(); err != nil {
				errCh <- err
				cancel()
				return
			}
			var err error
			if segment.Device.Kind == DeviceGPU && e.gpuForSegment(segment) != nil {
				err = gpuFn(segment)
			} else {
				err = cpuFn(segment)
			}
			if err != nil {
				errCh <- err
				cancel()
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	return ctx.Err()
}

func (e *Engine) gpuForSegment(segment Segment) gpuBackend {
	if e == nil {
		return nil
	}
	for _, gpu := range e.gpus {
		device := gpu.Device()
		if device.RegistryID != 0 && device.RegistryID == segment.Device.RegistryID {
			return gpu
		}
		if device.ID == segment.Device.ID && device.Backend == segment.Device.Backend {
			return gpu
		}
	}
	return nil
}
