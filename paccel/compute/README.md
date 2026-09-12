# compute — the cooperative compute engine

A dependency-free, **cgo-free** standalone port of the PowerEngine compute engine:
the float32 primitives every native diffusion and transformer engine runs on,
fanned out across CPU workers.

## Primitives

| Op | Meaning |
| --- | --- |
| `SAXPY(dst, x, a)` | `dst[i] += a*x[i]` |
| `Add(dst, a, b)` | `dst[i] = a[i] + b[i]` |
| `Scale(dst, src, s)` | `dst[i] = src[i]*s` |
| `MatMulBias(dst, lhs, rhs, bias, rows, inner, cols)` | `dst[r,c] = lhs[r,:]·rhs[c,:] + bias[c]` (rhs is row-major `[cols, inner]` — the HF `[out, in]` weight, so this is `y = x·Wᵀ + b`) |
| `Attention(out, q, k, v, qRows, kRows, headDim, scale)` | fused `softmax(scale·q·kᵀ)·v` |

## Cooperative plan

The upstream engine splits each op across CPU and Metal GPUs via a `Plan` of
device `Segment`s and runs them concurrently. This standalone build keeps the
identical API and the plan machinery but ships **no GPU backend** (`gpu_stub.go`),
so every plan is a single CPU segment and the package builds anywhere with no cgo.
A GPU backend can be reintroduced behind `newMetalBackends` without touching
callers.

```go
e, _ := compute.NewEngine(compute.Options{DisableGPU: true})
e.MatMulBias(ctx, dst, x, W, bias, rows, inner, cols)
e.Attention(ctx, out, q, k, v, qRows, kRows, headDim, scale)
```

`engine_test.go` checks `MatMulBias` and `Attention` against naive references
across 1 and N workers.
