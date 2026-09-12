# Stella V

Local medical-research agent. Stella V indexes **ScienceOpen preprints**, ranks them with a **GPU-capable tensor engine**, and reasons over the ranked passages with a selectable **1.5B or 3B** model. It ships as a self-contained app with a browser UI and **OpenAI-compatible HTTP endpoints**, so tools like [OpenCode](https://opencode.ai) can use it as a local provider.

This is research assistance, not diagnosis or treatment advice. Preprints are unreviewed manuscripts.

## What it does

1. Searches ScienceOpen-registered preprints (Crossref prefix `10.14293`, type `posted-content`).
2. Stores abstracts as a persistent knowledge base.
3. Retrieves passages with hashed n-gram embeddings, a linear projection, multi-head scaled dot-product attention, and batched GEMM. CUDA / Apple MPS is used when PyTorch is available; otherwise CPU.
4. Reasons with `qwen2.5:1.5b` or `qwen2.5:3b` via Ollama. The reasoner uses the tensor engine as a fast backend to select evidence.
5. Serves OpenAI-style `/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/v1/embeddings`, plus a small ScienceOpen tool loop at `/v1/agent/run`.

Medical topics: clinical trials, epidemiology, oncology, cardiology, immunology, infectious disease, pharmacology, genomics, neurology, public health.

## Layout

```
.
├── main.go                 Go CLI: bootstrap, ask, index, serve
├── openai.go               OpenAI-compatible /v1 API
├── scienceopen.go          ScienceOpen / Crossref preprint client
├── tensor_engine.go        Retrieval tensors (linear + attention)
├── reason.go               1.5B / 3B reasoner selection
├── opencode.example.json   Drop-in OpenCode provider config
└── Python/                 Trainer, ScienceOpen indexer, optional serve
```

The Go binary is the runtime. The Python tree trains the linear memory, indexes preprints, and can serve the same OpenAI API.

## Requirements

- Go 1.21+
- Python 3.10+ (trainer / optional API)
- [Ollama](https://ollama.com) for the 1.5B and 3B reasoners
- Network access to Crossref (`api.crossref.org`) for ScienceOpen metadata

PAccel export currently depends on `github.com/powerengine/paccel`. The `go.mod` `replace` line points at a sibling PowerEngine tree; for a standalone repo, vendor that module or drop the PAccel export path.

## Quick start (Go)

```bash
# reasoners (once)
python Python/setup_ollama.py          # pulls qwen2.5:1.5b and qwen2.5:3b

go test .
go run . bootstrap
go run . index "cancer immunotherapy"
go run . ask --reason 1.5b "what is a randomized trial?"
go run . serve --addr 127.0.0.1:8765 --reason 1.5b
```

Open http://127.0.0.1:8765 for the UI. The OpenAI base URL is http://127.0.0.1:8765/v1.

### CLI

| Command | Purpose |
| --- | --- |
| `bootstrap` | Write a medical bootstrap model |
| `train` | Train from JSONL `question` / `answer` pairs |
| `ask` | One-shot question (`--reason 1.5b\|3b`) |
| `chat` | REPL (`/teach`, `/index`, `/export`, `/save`) |
| `teach` | Store a teaching pair |
| `index` | Fetch ScienceOpen preprints into tensor memory |
| `serve` / `gui` | Browser UI + OpenAI `/v1` API |
| `export` | `paccel`, `safetensors`, or `onnx` |
| `info` | Model or `.paccel` stats |

```bash
go run . ask --backend-mode local "what is a preprint?"
go run . ask --reason 3b "summarize confounding in observational studies"
go run . index --rows 8 "checkpoint inhibitor melanoma"
go run . export safetensors stella.safetensors
```

`--backend-mode local` skips Ollama and live ScienceOpen. `--reason 1.5b` or `3b` selects the reasoner.

## OpenAI API and OpenCode

`serve` is the agent process. Clients that speak OpenAI chat completions can point at it.

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/v1/models` | `stella-v`, `stella-v-1.5b`, `stella-v-3b`, `stella-v-local` |
| `POST` | `/v1/chat/completions` | Chat, SSE streaming, tool calls |
| `POST` | `/v1/responses` | OpenAI Responses API |
| `POST` | `/v1/completions` | Legacy completions |
| `POST` | `/v1/embeddings` | Tensor-engine embeddings |
| `POST` | `/v1/agent/run` | Built-in ScienceOpen search / retrieve / index loop |
| `GET` | `/health` | Process + model stats |

Models:

- `stella-v` / `stella-v-1.5b` — 1.5B reasoner + retrieval
- `stella-v-3b` — 3B reasoner + retrieval
- `stella-v-local` — tensor memory only, no Ollama

When the client sends tools (OpenCode’s bash/edit/write tools), Stella forwards them to the reasoner and returns OpenAI `tool_calls`. Without tools, it retrieves ScienceOpen passages and answers from that evidence.

Stella itself does not write files. It **pushes retrieved preprint evidence**. If OpenCode (or another client) asks the **1.5B or 3B** model to write a script, Stella:

1. Searches ScienceOpen for the scientific topic (not the words “write a python script”).
2. Drafts the file from those passages (computational research prototype only).
3. Returns a `write` tool call so OpenCode creates the file.

That is how a request like “write a python chempy script … based on the research” is supposed to work with `stella-v-1.5b`. Stella will not stop at “therefore the answer is to use mRNA technology.” The drafted program is a research sketch grounded in unreviewed preprints — not a real vaccine, not manufacturing, and not clinical advice.

### Curl

```bash
curl -s http://127.0.0.1:8765/v1/models

curl -s http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"stella-v-local","messages":[{"role":"user","content":"what is a preprint?"}]}'
```

If `STELLAV_API_KEY` is set, send `Authorization: Bearer <key>` on `/v1` routes.

### OpenCode

Copy `opencode.example.json` to `~/.config/opencode/opencode.json` or a project `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "stella/stella-v-1.5b",
  "small_model": "stella/stella-v-1.5b",
  "provider": {
    "stella": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Stella V",
      "options": {
        "baseURL": "http://127.0.0.1:8765/v1",
        "apiKey": "stella-local"
      },
      "models": {
        "stella-v-1.5b": {
          "name": "Stella V 1.5B",
          "tool_call": true,
          "limit": { "context": 8192, "output": 2048 }
        },
        "stella-v-3b": {
          "name": "Stella V 3B",
          "tool_call": true,
          "limit": { "context": 8192, "output": 2048 }
        }
      }
    }
  }
}
```

Start Stella (`go run . serve`), then in OpenCode pick **Stella V / stella-v-1.5b**. `baseURL` must include `/v1`.

Any OpenAI-compatible client works the same way:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="stella-local")
print(client.chat.completions.create(
    model="stella-v-1.5b",
    messages=[{"role": "user", "content": "What is randomization in a trial?"}],
).choices[0].message.content)
```

## Python trainer

```bash
cd Python
pip install -r requirements.txt
python setup_ollama.py
python main.py bootstrap
python main.py index-scienceopen "cancer immunotherapy" "randomized controlled trial"
python main.py ask "What do checkpoint inhibitors do in melanoma trials?" --reason 1.5b
python main.py serve --addr 127.0.0.1:8765
python test_stella.py
```

| Command | Purpose |
| --- | --- |
| `bootstrap` | Medical QA bootstrap + linear model |
| `index-scienceopen` | Fetch and persist ScienceOpen preprints |
| `ask` | Retrieve + reason (`--reason 1.5b\|3b`) |
| `learn` | Agentic teacher loop over medical categories |
| `serve` | OpenAI-compatible API |
| `gui` | Optional pygame console (`pip install pygame`) |

Optional GPU tensors: `pip install torch`.

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `STELLA_MODEL` | `stella_model.json` | Go model path |
| `STELLA_ADDR` | `127.0.0.1:8765` | HTTP bind address |
| `STELLAV_REASON_SIZE` | `1.5b` | `1.5b` or `3b` |
| `STELLAV_REASON_1_5B` | `qwen2.5:1.5b` | Ollama 1.5B tag |
| `STELLAV_REASON_3B` | `qwen2.5:3b` | Ollama 3B tag |
| `STELLAV_OLLAMA_BASE` | `http://127.0.0.1:11434` | Ollama URL |
| `STELLAV_TENSOR_DEVICE` | auto | `cpu`, `cuda`, or `mps` |
| `STELLAV_LIVE_SEARCH` | on | Set `0` to disable live ScienceOpen on ask |
| `STELLAV_API_KEY` | unset | Bearer token required for `/v1` if set |
| `STELLAV_KB_PATH` | `stellav_knowledge_base.json` | Python knowledge base |
| `STELLAV_USER_AGENT` | StellaV/1.0 … | Crossref User-Agent |

## Tests

```bash
go test .
cd Python && python test_stella.py
```

## Exports

Export order is intentional: **PAccel** (runtime package), **safetensors** (tensor interchange), **ONNX** (small feature-to-logits graph). Retrieval projection tensors (`query_proj`, `key_proj`, `value_proj`) are included in safetensors / PAccel.

## License

Use and redistribute with the rest of this tree. Do not present Stella V output as clinical advice.
