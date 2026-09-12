# paccelchat — the frontend

A simple chatbot with Hugging Face model-download capabilities. Standard library
+ `libpaccel` only; no onnxruntime, no cgo.

## Commands

```
paccelchat pull     <repo> [--rev main] [--out ./models]      # download from HF Hub
paccelchat compress <model-dir|file.safetensors> <out.paccel> [--bits 4] [--block 32]
paccelchat info     <file.paccel> [-v]                        # inspect a container
paccelchat chat     [--paccel f.paccel] [--model dir] [--max-tokens 24]
```

Environment: `HF_TOKEN` (gated repos), `HF_ENDPOINT` (mirror host).

## How the chat works

The chat engine loads a `.paccel`, dequantizes the model's **real token-embedding
matrix** through `libpaccel`, and generates tokens by a nearest-neighbour walk in
embedding space (cosine similarity, with autoregressive-style context feedback).

This proves the codec decodes real model weights into usable float tensors. It is
deliberately **not** a full autoregressive transformer — the heavy graph executor
is out of scope for the bare-minimum port. The `Engine` interface in `engine.go`
is the single seam where a full executor (or the native `cpp/` matmul path) would
plug in; `libpaccel` already supplies it dequantized weights on demand.
