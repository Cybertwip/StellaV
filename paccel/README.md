# PAccel — bare-minimum port (no onnxruntime)

This directory is a self-contained, dependency-free extraction of the PAccel
model-weight format from `External/inference`. It drops onnxruntime, protobuf,
Eigen, abseil and every other heavy dependency, keeping only what is needed to
**read, write, quantize and dequantize** `.paccel` containers, plus a simple
chatbot frontend that can download models from Hugging Face.

```
Caveats/
├── paccel-format.pdf     Microsoft pitch + full algebra/algorithm spec
├── go.mod                single Go module: github.com/powerengine/paccel
├── libpaccel/            the format library (pure Go, stdlib only)
│   ├── fp16.go             IEEE-754 half <-> single conversions
│   ├── quantize.go         block-wise symmetric quantize / dequantize
│   ├── pack.go             low-bit packing + v6 AVX tiled-int4 layout
│   ├── paccel.go           PACONNX1 container reader / writer (v4–v6)
│   ├── decode.go           source-dtype decode + mixed-precision policy
│   ├── safetensors.go      safetensors -> .paccel ingest
│   ├── v7.go               V7 direct layout: pre-transposed aliases + transpose
│   ├── quant_i8.go         int8 activation wire quant (distributed path)
│   ├── rle.go              raw run-length codec
│   └── paccel_test.go      round-trip parity tests
├── v7graph/              the V7 direct graph — the optimal inference route
│   ├── recipe.go           model.v7.json schema + validation
│   ├── config.go           Hugging Face config.json -> recipe
│   ├── weights.go          bind pre-transposed weights from a package
│   ├── graph.go            transformer_decoder forward (RMSNorm/RoPE/GQA/SwiGLU)
│   └── v7graph_test.go     synthetic end-to-end forward test
├── paccelchat/           the frontend: a simple chatbot
│   ├── main.go             CLI: pull / compress (--v7) / info / chat
│   ├── hf.go               Hugging Face Hub download
│   ├── engine.go           embedding-grounded chat engine over libpaccel
│   └── engine_v7.go        V7 direct-graph chat engine (optimal route)
├── compute/              the cooperative compute engine (CPU, no cgo)
│   ├── engine.go           SAXPY / Add / Scale / MatMulBias / Attention + plans
│   ├── cpu.go              the float32 CPU kernels
│   └── gpu_stub.go         GPU backend stub (Metal plugs in here upstream)
├── diffusion/            samplers: Euler/Karras v-pred, LCM epsilon, CFG, pipeline
├── mmengine/             reference DiT backbone + decoder on the compute engine
├── audio/                standalone text-to-audio engine (Stable Audio 3 style)
├── image/                standalone few-step text-to-image engine (LCM)
├── video/                standalone text-to-video engine (CogVideoX style)
└── cpp/                  extracted minimal C++ accelerator (the only C++ needed)
    ├── include/accelerator/accelerator.hpp
    ├── src/accelerator.cpp
    ├── test/selftest.cpp
    └── CMakeLists.txt
```

## Why no onnxruntime

The original engine only used onnxruntime through a thin `dlopen` bridge
(`External/paccel/src/paccel_ort_bridge.cpp`) for an optional accelerated
matmul. The PAccel **format itself** — the quantization algebra and the
container — never needed it. This port keeps the format and ships a portable
scalar matmul (`cpp/`) so nothing links against onnxruntime.

## Build & test

Go library + frontend (needs Go 1.21+):

```bash
go build ./...
go test ./libpaccel/
go run ./paccelchat help
```

C++ accelerator (needs a C++20 compiler):

```bash
cd cpp && cmake -B build && cmake --build build && ctest --test-dir build
# or, no cmake:
g++ -std=c++20 -O2 -Iinclude src/accelerator.cpp test/selftest.cpp -o selftest && ./selftest
```

## V7 — the optimal inference route

V7 is a "direct-only" mode that stores every linear/LM-head weight **pre-transposed**
(no runtime transpose) and pairs the package with a tiny `model.v7.json` recipe.
A compact, dependency-free decoder (`v7graph`) runs the Qwen2/Qwen3 (Llama-family)
forward pass — embed → N×(RMSNorm, RoPE + grouped-query attention, SwiGLU) → RMSNorm
→ head — directly against the block-quantized weights. No ONNX, no interpreter, no
transpose buffers: the smallest footprint and fewest instructions per token the
format supports.

The current layout, `v7-transformer-direct-fused-int4`, additionally **fuses** the
projections that share an input — q/k/v into one attention matmul and gate/up into
one MLP matmul — by column-concatenating their pre-transposed forms. One product
yields all parts (recovered by slicing output columns); the fusion is bit-identical
to the separate products, so it raises arithmetic intensity without changing
numerics. Attention biases (Qwen2) are added to the slices.

It is an especially strong fit for NVIDIA's GB10-class hardware (DGX Spark): the
128 GB **coherent** CPU+GPU unified memory lets the 8-bit sensitive tensors run on
the Arm cores and the 4-bit blocks on the Blackwell NVFP4 tensor cores from one
zero-copy weight image — and PAccel's block-int4 is the same construction as NVFP4
(16-element blocks with a block scale). See `paccel-format.pdf` §12–13.

```bash
go run ./paccelchat compress ./models/qwen ./models/qwen/model.paccel --bits 4 --v7
# -> writes model.paccel (with v7 aliases) + model.v7.json
go run ./paccelchat chat --paccel ./models/qwen/model.paccel --model ./models/qwen
# -> auto-detects model.v7.json and runs the V7 direct graph
```

## Multimodal engines (compute + diffusion)

Beyond text, the same dependency-free philosophy extends to the native
audio/image/video engines. They all run on one small **compute engine**
(`compute/`) — a cooperative executor exposing `MatMulBias`, fused `Attention`,
and elementwise ops on CPU workers (the upstream Metal GPU backend is stubbed for
a no-cgo build). On top sits a **diffusion core** (`diffusion/`) with the exact
sampler algebra: the scaled-linear/Karras noise schedules, the **Euler
v-prediction** sampler (audio/video), the few-step **LCM epsilon** sampler
(image), classifier-free guidance, and a generic latent-diffusion loop.

The three engines are thin pipelines over that core:

| Engine | Pipeline | Sampler |
| --- | --- | --- |
| `audio/` | T5Gemma → DiT → VAE (Stable Audio 3) | Euler v-prediction |
| `image/` | CLIP/T5 → UNet → VAE | LCM epsilon (1–8 steps) |
| `video/` | T5 → 3D DiT → 3D VAE (CogVideoX) | Euler v-prediction |

```go
e, _ := audio.New(audio.Default())
pcm, _ := e.Generate("a calm piano melody", 5.0, 42) // interleaved stereo PCM
```

Each ships a reference DiT backbone and decoder (`mmengine/`) built on the compute
engine, so it runs end-to-end today. The production backbones and VAEs load from a
`.paccel` via `libpaccel` and plug into the same `diffusion.Backbone` /
`diffusion.Decoder` seams. See `paccel-format.pdf` §14 and `diffusion/README.md`.

## End-to-end demo

```bash
go run ./paccelchat pull Qwen/Qwen2.5-0.5B-Instruct --out ./models
go run ./paccelchat compress ./models/Qwen__* ./models/qwen.paccel --bits 4 --v7
go run ./paccelchat info ./models/qwen.paccel
go run ./paccelchat chat --paccel ./models/qwen.paccel --model ./models/Qwen__*
```

See `paccel-format.pdf` for the exact algebra, the container byte layout, the V7
direct graph, the NVIDIA DGX Spark analysis, and the delivery plan.
