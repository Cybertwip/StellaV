# v7graph — the V7 direct graph (optimal inference route)

V7 is the optimal route from a quantized checkpoint to a token. It removes the
ONNX export and interpreter entirely: weights are stored **pre-transposed** so a
matmul needs no runtime transpose, and the compute graph is reconstructed from a
tiny `model.v7.json` recipe instead of being serialized.

This package runs the recipe directly — a Qwen2/Qwen3 (Llama-family) decoder —
against the block-quantized, direct-layout weights decoded by `libpaccel`. It
depends only on the standard library and `libpaccel` (no onnxruntime, no cgo).

## The graph

```
h = Embed[token]
for each layer:
    a = RMSNorm(h);  q,k,v = a·Wq, a·Wk, a·Wv        (pre-transposed)
    RoPE(theta) on q,k;  grouped-query causal attention
    h = h + (attn)·Wo
    m = RMSNorm(h);  h = h + ( SiLU(m·Wgate) * (m·Wup) )·Wdown
h = RMSNorm(h);  logits = h·Wlm_head
```

Every `·` is a plain row-major multiply against a pre-transposed int4/8-bit
block-quantized matrix. Optional Qwen3 per-head q/k RMSNorm is applied if present.

### Fused projections (`v7-transformer-direct-fused-int4`)

The projections that share an input are **fused** into one matmul. The three
attention weights concatenate (pre-transposed) along the output axis into one
matrix, and gate/up likewise; one product yields all parts, recovered by slicing
output columns:

```
W_qkv     = [ Wq^T | Wk^T | Wv^T ]   shape [hidden, out_q + 2·kv]
x · W_qkv = [ x·Wq^T | x·Wk^T | x·Wv^T ]   -> slice into q, k, v  (+ biases)
W_gate_up = [ Wg^T | Wu^T ]          shape [hidden, 2·intermediate]
```

The fusion is **exact** — bit-identical to the separate products — so it changes
only performance: one matmul replaces three (attention) and one replaces two
(MLP), with a single contiguous tiled-int4 operand per pair. `BindWeights` reads
the fused aliases when present and otherwise reconstructs them from the split
projections, so both layouts run unchanged.

## API

```go
m, _ := v7graph.Load("model.v7.json", "model.paccel")
logits := m.Forward([]int{1, 5, 9, 3})   // [seq][vocab]
next   := m.NextToken([]int{1, 5, 9, 3})  // greedy argmax
```

Produce a V7 package from a Hugging Face checkpoint with
`paccelchat compress <dir> <out.paccel> --v7`, which appends the pre-transposed
aliases (`libpaccel.AppendV7Aliases`) and writes the recipe
(`v7graph.RecipeFromHFConfig`).

## Why it is optimal

No graph to parse, no runtime to load, no transpose buffers; the v6 tiled int4
layout streams vector-friendly tiles; mixed precision keeps embeddings/head/norms
at 8-bit and the blocks at 4-bit. Smallest resident footprint and fewest
instructions per token of any route — ideal for memory-bound, coherent-memory
accelerators such as NVIDIA DGX Spark (GB10), whose NVFP4 4-bit tensor cores use
the same 16-element block-scaled construction as PAccel int4.
