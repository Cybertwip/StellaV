package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type guiServer struct {
	mu        sync.Mutex
	model     *StellaModel
	modelPath string
	runtime   ConquerorConfig
}

func RunGUI(addr, modelPath string, model *StellaModel, runtime ConquerorConfig) error {
	s := &guiServer{model: model, modelPath: modelPath, runtime: runtime.normalized()}
	handler := withCORS(s.routes())
	log.Printf("Stella V agent listening on http://%s", addr)
	log.Printf("OpenAI-compatible API: http://%s/v1  (chat/completions, models, responses)", addr)
	log.Printf("OpenCode baseURL: http://%s/v1  models: stella-v-1.5b, stella-v-3b", addr)
	return http.ListenAndServe(addr, handler)
}

func (s *guiServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleHome)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/state", s.handleState)
	mux.HandleFunc("/api/chat", s.handleChat)
	mux.HandleFunc("/api/teach", s.handleTeach)
	mux.HandleFunc("/api/export", s.handleExport)
	mux.HandleFunc("/api/index", s.handleScienceOpenIndex)
	mux.HandleFunc("/api/reason", s.handleReason)
	mux.HandleFunc("/v1/models", s.handleOpenAIModels)
	mux.HandleFunc("/v1/models/", s.handleOpenAIModels)
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChatCompletions)
	mux.HandleFunc("/v1/completions", s.handleOpenAICompletions)
	mux.HandleFunc("/v1/responses", s.handleOpenAIResponses)
	mux.HandleFunc("/v1/embeddings", s.handleOpenAIEmbeddings)
	mux.HandleFunc("/v1/agent/run", s.handleAgentRun)
	return mux
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, OpenAI-Beta")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/v1") && !sAuthOK(r) {
			http.Error(w, `{"error":{"message":"invalid api key","type":"invalid_request_error"}}`, http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func sAuthOK(r *http.Request) bool {
	want := strings.TrimSpace(os.Getenv("STELLAV_API_KEY"))
	if want == "" {
		return true
	}
	got := strings.TrimSpace(r.Header.Get("Authorization"))
	got = strings.TrimPrefix(got, "Bearer ")
	got = strings.TrimPrefix(got, "bearer ")
	return got == want
}

func (s *guiServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	writeJSON(w, map[string]any{
		"ok":      true,
		"name":    "Stella V",
		"object":  "stella.health",
		"stats":   s.model.Stats(),
		"runtime": NewReplyEngine(s.model, s.runtime).Status(),
		"openai":  "/v1",
	})
}

func (s *guiServer) handleHome(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(stellaHTML))
}

func (s *guiServer) handleState(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	writeJSON(w, map[string]any{
		"stats":      s.model.Stats(),
		"model_path": s.modelPath,
		"runtime":    NewReplyEngine(s.model, s.runtime).Status(),
	})
}

func (s *guiServer) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	reply := NewReplyEngine(s.model, s.runtime).Reply(req.Message)
	stats := s.model.Stats()
	runtime := NewReplyEngine(s.model, s.runtime).Status()
	s.mu.Unlock()
	writeJSON(w, map[string]any{"reply": reply, "stats": stats, "runtime": runtime})
}

func (s *guiServer) handleTeach(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Question string `json:"question"`
		Answer   string `json:"answer"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Question) == "" || strings.TrimSpace(req.Answer) == "" {
		http.Error(w, "question and answer are required", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.model.AddSample(req.Question, req.Answer)
	err := s.model.Save(s.modelPath)
	stats := s.model.Stats()
	s.mu.Unlock()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"ok": true, "stats": stats})
}

func (s *guiServer) handleScienceOpenIndex(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Query string `json:"query"`
		Rows  int    `json:"rows"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	if req.Rows <= 0 {
		req.Rows = 8
	}
	s.mu.Lock()
	n, err := indexScienceOpenIntoModel(s.model, req.Query, req.Rows)
	if err == nil {
		err = s.model.Save(s.modelPath)
	}
	stats := s.model.Stats()
	s.mu.Unlock()
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"ok": true, "indexed": n, "stats": stats})
}

func (s *guiServer) handleReason(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Size string `json:"size"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.runtime.ReasonSize = NormalizeReasonSize(req.Size)
	s.model.ReasonSize = s.runtime.ReasonSize
	_ = s.model.Save(s.modelPath)
	runtime := NewReplyEngine(s.model, s.runtime).Status()
	s.mu.Unlock()
	writeJSON(w, map[string]any{"ok": true, "runtime": runtime})
}

func (s *guiServer) handleExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Format string `json:"format"`
		Output string `json:"output"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	format := strings.ToLower(strings.TrimSpace(req.Format))
	output := strings.TrimSpace(req.Output)
	if output == "" {
		output = defaultExportPath(s.modelPath, format)
	}
	s.mu.Lock()
	var err error
	switch format {
	case "paccel":
		err = s.model.ExportPAccel(output, 4, 32)
	case "safetensors":
		err = s.model.ExportSafetensors(output)
	case "onnx":
		err = s.model.ExportONNX(output)
	default:
		err = fmt.Errorf("unsupported export format %q", format)
	}
	stats := s.model.Stats()
	s.mu.Unlock()
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"ok": true, "path": output, "stats": stats})
}

func defaultExportPath(modelPath, format string) string {
	base := strings.TrimSuffix(modelPath, filepath.Ext(modelPath))
	switch format {
	case "paccel":
		return base + ".paccel"
	case "safetensors":
		return base + ".safetensors"
	case "onnx":
		return base + ".onnx"
	default:
		return base + ".export"
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

const stellaHTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stella V · medical research</title>
<style>
:root {
  color-scheme: dark;
  --bg: #15171a;
  --panel: #20242a;
  --panel-2: #262b32;
  --ink: #f3f6f8;
  --muted: #aeb8c2;
  --line: #3a424c;
  --green: #57c78b;
  --amber: #d7b451;
  --red: #df6d75;
  --blue: #66a7d8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input, textarea { font: inherit; }
.shell {
  min-height: 100vh;
  display: grid;
  grid-template-rows: 56px 1fr;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid var(--line);
  padding: 0 20px;
  background: #191c20;
}
h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0;
}
.stats {
  display: flex;
  gap: 14px;
  color: var(--muted);
  white-space: nowrap;
}
main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  min-height: 0;
}
.chat {
  display: grid;
  grid-template-rows: 1fr auto;
  min-width: 0;
  min-height: 0;
}
#log {
  overflow: auto;
  padding: 20px;
}
.turn {
  max-width: 900px;
  margin: 0 0 14px;
  padding: 12px 14px;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
}
.turn.user {
  margin-left: auto;
  background: #1d2830;
  border-color: #335267;
}
.turn .role {
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 5px;
}
.turn .meta {
  color: var(--muted);
  font-size: 12px;
  margin-top: 8px;
}
.composer {
  border-top: 1px solid var(--line);
  padding: 14px 20px;
  background: #191c20;
}
.row {
  display: flex;
  gap: 10px;
  align-items: stretch;
}
textarea, input {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--ink);
  background: #111316;
  padding: 10px 12px;
  resize: vertical;
}
textarea { min-height: 52px; max-height: 180px; }
button {
  min-width: 82px;
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--ink);
  background: var(--panel-2);
  padding: 0 14px;
  cursor: pointer;
}
button.primary { background: #244436; border-color: #377758; }
button.export { color: #101214; background: var(--amber); border-color: #e0c26f; }
button:disabled { opacity: .55; cursor: default; }
aside {
  border-left: 1px solid var(--line);
  background: #191c20;
  padding: 18px;
  overflow: auto;
}
h2 {
  margin: 0 0 10px;
  font-size: 14px;
  color: var(--muted);
  font-weight: 700;
  letter-spacing: 0;
}
.tool {
  padding: 14px 0 18px;
  border-bottom: 1px solid var(--line);
}
.tool:first-child { padding-top: 0; }
.tool:last-child { border-bottom: 0; }
.stack { display: grid; gap: 10px; }
.exports { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
#status { color: var(--muted); margin-top: 12px; min-height: 20px; }
@media (max-width: 860px) {
  main { grid-template-columns: 1fr; }
  aside { border-left: 0; border-top: 1px solid var(--line); }
  .stats { display: none; }
}
</style>
</head>
<body>
<div class="shell">
  <header>
    <h1>Stella V · medical research</h1>
    <div class="stats">
      <span id="sampleStat">Samples 0</span>
      <span id="vocabStat">Vocab 0</span>
      <span id="chunkStat">Preprints 0</span>
      <span id="backendStat">Reasoner 1.5b</span>
    </div>
  </header>
  <main>
    <section class="chat">
      <div id="log"></div>
      <form id="chatForm" class="composer">
        <div class="row">
          <textarea id="message" autocomplete="off" spellcheck="true"></textarea>
          <button class="primary" type="submit">Send</button>
        </div>
      </form>
    </section>
    <aside>
      <div class="tool">
        <h2>Reasoner</h2>
        <div class="exports">
          <button data-reason="1.5b">1.5B</button>
          <button data-reason="3b">3B</button>
        </div>
      </div>
      <div class="tool">
        <h2>ScienceOpen</h2>
        <form id="indexForm" class="stack">
          <input id="indexQ" placeholder="preprint query or DOI">
          <button class="primary" type="submit">Index</button>
        </form>
      </div>
      <div class="tool">
        <h2>Teach</h2>
        <form id="teachForm" class="stack">
          <input id="teachQ" placeholder="Question">
          <textarea id="teachA" placeholder="Answer"></textarea>
          <button class="primary" type="submit">Save</button>
        </form>
      </div>
      <div class="tool">
        <h2>Export</h2>
        <div class="exports">
          <button class="export" data-export="paccel">PAccel</button>
          <button data-export="safetensors">Safe</button>
          <button data-export="onnx">ONNX</button>
        </div>
        <div id="status"></div>
      </div>
    </aside>
  </main>
</div>
<script>
const log = document.querySelector("#log");
const message = document.querySelector("#message");
const statusEl = document.querySelector("#status");

function addTurn(role, text, meta) {
  const node = document.createElement("div");
  node.className = "turn " + (role === "You" ? "user" : "bot");
  node.innerHTML = "<div class='role'></div><div class='text'></div><div class='meta'></div>";
  node.querySelector(".role").textContent = role;
  node.querySelector(".text").textContent = text;
  node.querySelector(".meta").textContent = meta || "";
  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
}

function updateStats(stats) {
  if (!stats) return;
  document.querySelector("#sampleStat").textContent = "Samples " + stats.samples;
  document.querySelector("#vocabStat").textContent = "Vocab " + stats.vocab;
  document.querySelector("#chunkStat").textContent = "Preprints " + (stats.chunks || 0);
}

function updateRuntime(runtime) {
  if (!runtime) return;
  const label = (runtime.reason_size || "1.5b") + " / " + (runtime.model || "local") + " / " + (runtime.tensor_device || "cpu");
  document.querySelector("#backendStat").textContent = "Reasoner " + label;
}

async function postJSON(url, body) {
  const res = await fetch(url, { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

document.querySelector("#chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = message.value.trim();
  if (!text) return;
  message.value = "";
  addTurn("You", text, "");
  try {
    const data = await postJSON("/api/chat", {message: text});
    updateStats(data.stats);
    updateRuntime(data.runtime);
    addTurn("Stella", data.reply.text, data.reply.source + " | confidence " + data.reply.confidence.toFixed(3));
  } catch (err) {
    addTurn("Stella", String(err), "error");
  }
});

document.querySelector("#indexForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const data = await postJSON("/api/index", {query: document.querySelector("#indexQ").value, rows: 8});
    updateStats(data.stats);
    statusEl.textContent = "Indexed " + data.indexed + " ScienceOpen preprint(s).";
  } catch (err) {
    statusEl.textContent = String(err);
  }
});

for (const button of document.querySelectorAll("[data-reason]")) {
  button.addEventListener("click", async () => {
    const size = button.getAttribute("data-reason");
    try {
      const data = await postJSON("/api/reason", {size});
      updateRuntime(data.runtime);
      statusEl.textContent = "Reasoner set to " + size + ".";
    } catch (err) {
      statusEl.textContent = String(err);
    }
  });
}

document.querySelector("#teachForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const data = await postJSON("/api/teach", {
      question: document.querySelector("#teachQ").value,
      answer: document.querySelector("#teachA").value
    });
    updateStats(data.stats);
    statusEl.textContent = "Saved teaching pair.";
  } catch (err) {
    statusEl.textContent = String(err);
  }
});

for (const button of document.querySelectorAll("[data-export]")) {
  button.addEventListener("click", async () => {
    const format = button.getAttribute("data-export");
    statusEl.textContent = "Exporting " + format + "...";
    try {
      const data = await postJSON("/api/export", {format});
      updateStats(data.stats);
      statusEl.textContent = "Wrote " + data.path;
    } catch (err) {
      statusEl.textContent = String(err);
    }
  });
}

fetch("/api/state").then(r => r.json()).then(data => {
  updateStats(data.stats);
  updateRuntime(data.runtime);
  addTurn("Stella", "Medical-research agent loaded from " + data.model_path + ". OpenAI-compatible API is at /v1 for OpenCode. Models: stella-v-1.5b and stella-v-3b. This is not clinical advice.", "ready");
});
</script>
</body>
</html>`
