# Stella V Python trainer

Project overview, OpenAI/OpenCode setup, and the Go runtime live in the [root README](../README.md).

This directory is the Python trainer, ScienceOpen indexer, and optional `/v1` server.

Stella V is a local medical-research model. It:

1. Indexes **ScienceOpen preprints** (Crossref prefix `10.14293`, type `posted-content`).
2. Stores them in a persistent knowledge base.
3. Retrieves passages with a **GPU-capable tensor engine** (hashed embeddings, linear projection, multi-head scaled dot-product attention, batched GEMM).
4. Reasons over the ranked passages with a selectable **1.5B or 3B** local model (`qwen2.5:1.5b` / `qwen2.5:3b` via Ollama).

This is research assistance, not diagnosis or treatment advice. Game-engine and Grokipedia sources have been removed.

## Install

```bash
pip install -r requirements.txt
python setup_ollama.py   # pulls qwen2.5:1.5b and qwen2.5:3b
```

Optional GPU tensors: `pip install torch` (CUDA or Apple MPS).

## Ask

```bash
python main.py bootstrap
python main.py index-scienceopen "cancer immunotherapy" "randomized controlled trial"
python main.py ask "What do checkpoint inhibitors do in melanoma trials?" --reason 1.5b
python main.py ask "Summarize confounding in observational studies." --reason 3b
```

`--local-only` skips the reasoner and live ScienceOpen search. `--no-live-search` uses the local tensor index only.

## Self-contained agent + OpenCode

`serve` (same as `gui`) is the self-contained agentic app. It exposes OpenAI-compatible endpoints so OpenCode and other OpenAI clients can use Stella V as a local provider.

```bash
# Go app (recommended)
go run . serve --addr 127.0.0.1:8765 --reason 1.5b

# Python equivalent
python main.py serve --addr 127.0.0.1:8765
```

Endpoints:

| Method | Path | Use |
| --- | --- | --- |
| GET | `/v1/models` | List `stella-v`, `stella-v-1.5b`, `stella-v-3b`, `stella-v-local` |
| POST | `/v1/chat/completions` | Chat + streaming + tool calls |
| POST | `/v1/responses` | OpenAI Responses API |
| POST | `/v1/completions` | Legacy completions |
| POST | `/v1/embeddings` | Tensor-engine embeddings |
| POST | `/v1/agent/run` | Stella-native ScienceOpen tool loop |

Copy `opencode.example.json` to `~/.config/opencode/opencode.json` (or your project `opencode.json`):

```json
{
  "model": "stella/stella-v-1.5b",
  "provider": {
    "stella": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Stella V",
      "options": {
        "baseURL": "http://127.0.0.1:8765/v1",
        "apiKey": "stella-local"
      },
      "models": {
        "stella-v-1.5b": { "name": "Stella V 1.5B", "tool_call": true },
        "stella-v-3b": { "name": "Stella V 3B", "tool_call": true }
      }
    }
  }
}
```

Then in OpenCode pick **Stella V / stella-v-1.5b**. Optional: `export STELLAV_API_KEY=stella-local` to require that bearer token.

Quick check:

```bash
curl -s http://127.0.0.1:8765/v1/models
curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"stella-v-local","messages":[{"role":"user","content":"what is a preprint?"}]}'
```

## Go CLI

From `Tools/StellaV`:

```bash
go test .
go run . bootstrap
go run . index "cancer immunotherapy"
go run . ask --reason 1.5b "what is a randomized trial?"
go run . serve --reason 1.5b
```

Export order is unchanged: PAccel, safetensors, ONNX. Retrieval projection tensors are included.

## Learn

```bash
python main.py learn --reason 1.5b
python main.py learn --no-scienceopen-default --scienceopen-query "heart failure outcomes"
```

Default learning indexes the medical ScienceOpen query manifest (clinical trials, epidemiology, oncology, cardiology, immunology, infectious disease, pharmacology, genomics, neurology, public health).

## Environment

```bash
export STELLAV_REASON_SIZE=1.5b          # or 3b
export STELLAV_REASON_1_5B=qwen2.5:1.5b
export STELLAV_REASON_3B=qwen2.5:3b
export STELLAV_OLLAMA_BASE=http://localhost:11434
export STELLAV_TENSOR_DEVICE=mps         # cpu | cuda | mps
export STELLAV_LIVE_SEARCH=1
export STELLAV_KB_PATH=stellav_knowledge_base.json
```
