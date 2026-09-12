package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	Reason1p5B          = "1.5b"
	Reason3B            = "3b"
	defaultOllamaBase   = "http://127.0.0.1:11434"
	defaultOllama1p5B   = "qwen2.5:1.5b"
	defaultOllama3B     = "qwen2.5:3b"
	defaultConqueror1p5 = "hf-qwen-qwen2-5-1-5b"
	defaultConqueror3B  = "hf-qwen-qwen2-5-3b"
)

func NormalizeReasonSize(value string) string {
	key := strings.ToLower(strings.TrimSpace(value))
	key = strings.ReplaceAll(key, "_", "-")
	switch key {
	case Reason3B, "3", "3.0b", "qwen2.5:3b", "qwen2.5-3b", "llama3.2:3b":
		return Reason3B
	default:
		return Reason1p5B
	}
}

func ollamaModelFor(size string) string {
	size = NormalizeReasonSize(size)
	if size == Reason3B {
		return envOr("STELLAV_REASON_3B", defaultOllama3B)
	}
	return envOr("STELLAV_REASON_1_5B", defaultOllama1p5B)
}

func conquerorModelFor(size string) string {
	size = NormalizeReasonSize(size)
	if size == Reason3B {
		return envOr("STELLAV_CONQUEROR_3B", defaultConqueror3B)
	}
	return envOr("STELLAV_CONQUEROR_1_5B", defaultConqueror1p5)
}

func reasonerSystemPrompt() string {
	return strings.Join([]string{
		"You are Stella V, a local medical-research assistant.",
		"You retrieve ScienceOpen preprints through a tensor engine, then reason over that evidence.",
		"Use only the supplied passages plus established study methods.",
		"Select the passages that actually answer the question.",
		"Cite ScienceOpen DOIs. Treat preprints as unreviewed evidence.",
		"This is not diagnosis or treatment advice.",
		"Do not discuss games or game engines.",
	}, " ")
}

func agentSystemPrompt() string {
	return strings.Join([]string{
		"You are Stella V, a local medical-research assistant with OpenAI-compatible tool calling.",
		"You retrieve ScienceOpen preprints and push that evidence into context. You do not write files to disk.",
		"When the client (OpenCode) provides write, edit, or bash tools, use those tools to fulfill script and file requests.",
		"Prefer tools over guessing. Do not replace a requested script with a literature summary.",
		"When medical evidence is relevant, cite ScienceOpen DOIs. Treat preprints as unreviewed.",
		"This is not diagnosis or treatment advice. Computational research code is not a therapy or manufacturing protocol.",
		"Return a concise final answer when no further tool call is needed.",
	}, " ")
}

func formatRankedPassages(hits []RankedChunk, limit int) string {
	if limit <= 0 || limit > len(hits) {
		limit = len(hits)
	}
	var b strings.Builder
	for i := 0; i < limit; i++ {
		hit := hits[i]
		fmt.Fprintf(&b, "[%d] %s\nDOI: %s\nURL: %s\nTensor score: %.4f\nPassage: %s\n\n",
			i+1,
			singleLine(hit.Chunk.Title, 180),
			hit.Chunk.DOI,
			hit.Chunk.URL,
			hit.Score,
			singleLine(hit.Chunk.Text, 900),
		)
	}
	if b.Len() == 0 {
		return "(no preprint passages retrieved)"
	}
	return strings.TrimSpace(b.String())
}

func (m *StellaModel) BuildReasonPrompt(question string, local Prediction, hits []RankedChunk) string {
	var b strings.Builder
	b.WriteString(reasonerSystemPrompt())
	b.WriteString("\n\nQuestion:\n")
	b.WriteString(strings.TrimSpace(question))
	b.WriteByte('\n')
	if strings.TrimSpace(local.Text) != "" && local.Source != "fallback" && local.Source != "empty" {
		b.WriteString("\nLocal tensor-memory draft:\n")
		b.WriteString(singleLine(local.Text, 420))
		b.WriteByte('\n')
	}
	b.WriteString("\nScienceOpen preprint passages ranked by the Stella V tensor engine:\n")
	b.WriteString(formatRankedPassages(hits, 6))
	b.WriteString("\n\nSelect the relevant passages, reason over them, and answer. End with the DOIs you used.")
	return b.String()
}

type ollamaGenerateRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Stream  bool           `json:"stream"`
	Options map[string]any `json:"options"`
}

type ollamaGenerateResponse struct {
	Response string `json:"response"`
	Error    string `json:"error"`
}

func queryOllama(ctx context.Context, model, prompt string) (string, error) {
	return queryOllamaPredict(ctx, model, prompt, 512)
}

func queryOllamaPredict(ctx context.Context, model, prompt string, numPredict int) (string, error) {
	if numPredict <= 0 {
		numPredict = 512
	}
	base := strings.TrimRight(envOr("STELLAV_OLLAMA_BASE", defaultOllamaBase), "/")
	body, err := json.Marshal(ollamaGenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
		Options: map[string]any{
			"temperature": 0.15,
			"num_predict": numPredict,
			"num_ctx":     8192,
		},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var parsed ollamaGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", err
	}
	if parsed.Error != "" {
		return "", fmt.Errorf("ollama: %s", parsed.Error)
	}
	return strings.TrimSpace(parsed.Response), nil
}

var (
	ollamaAliveAt time.Time
	ollamaAliveOK bool
)

func ollamaAlive() bool {
	if time.Since(ollamaAliveAt) < 5*time.Second {
		return ollamaAliveOK
	}
	base := strings.TrimRight(envOr("STELLAV_OLLAMA_BASE", defaultOllamaBase), "/")
	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/api/tags", nil)
	if err != nil {
		ollamaAliveAt = time.Now()
		ollamaAliveOK = false
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		ollamaAliveAt = time.Now()
		ollamaAliveOK = false
		return false
	}
	defer resp.Body.Close()
	ollamaAliveAt = time.Now()
	ollamaAliveOK = resp.StatusCode >= 200 && resp.StatusCode < 300
	return ollamaAliveOK
}

type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaChatMsg `json:"messages"`
	Tools    []openaiTool    `json:"tools,omitempty"`
	Stream   bool            `json:"stream"`
	Options  map[string]any  `json:"options,omitempty"`
}

type ollamaChatMsg struct {
	Role      string              `json:"role"`
	Content   string              `json:"content"`
	ToolCalls []ollamaToolCallOut `json:"tool_calls,omitempty"`
}

type ollamaToolCallOut struct {
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type ollamaChatResponse struct {
	Message ollamaChatMsg `json:"message"`
	Error   string        `json:"error"`
}

func collectClientSystem(messages []openaiMessage) string {
	var b strings.Builder
	for _, msg := range messages {
		if strings.ToLower(strings.TrimSpace(msg.Role)) != "system" {
			continue
		}
		text := flattenMessageContent(msg.Content)
		if text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(text)
	}
	return strings.TrimSpace(b.String())
}

func queryOllamaChat(cfg ConquerorConfig, messages []openaiMessage, tools []openaiTool, rag string, maxTokens int) (openaiMessage, error) {
	if !ollamaAlive() {
		return openaiMessage{}, fmt.Errorf("ollama is not reachable")
	}
	model := ollamaModelFor(cfg.ReasonSize)
	sys := agentSystemPrompt()
	if clientSys := collectClientSystem(messages); clientSys != "" {
		sys += "\n\nClient instructions (honor these tool schemas):\n" + truncateRunes(clientSys, 2000)
	}
	if strings.TrimSpace(rag) != "" {
		sys += "\n\n" + rag
	}
	outMsgs := []ollamaChatMsg{{Role: "system", Content: sys}}
	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		if role == "system" {
			continue
		}
		if role == "" {
			role = "user"
		}
		converted := ollamaChatMsg{Role: role, Content: flattenMessageContent(msg.Content)}
		for _, call := range msg.ToolCalls {
			args := json.RawMessage(call.Function.Arguments)
			if !json.Valid(args) {
				enc, _ := json.Marshal(call.Function.Arguments)
				args = enc
			}
			item := ollamaToolCallOut{Type: "function"}
			item.Function.Name = call.Function.Name
			item.Function.Arguments = args
			converted.ToolCalls = append(converted.ToolCalls, item)
		}
		outMsgs = append(outMsgs, converted)
	}
	if maxTokens <= 0 {
		maxTokens = cfg.NumPredict
	}
	if maxTokens <= 0 {
		maxTokens = 512
	}
	body, err := json.Marshal(ollamaChatRequest{
		Model:    model,
		Messages: outMsgs,
		Tools:    tools,
		Stream:   false,
		Options: map[string]any{
			"temperature": 0.2,
			"num_predict": maxTokens,
			"num_ctx":     8192,
		},
	})
	if err != nil {
		return openaiMessage{}, err
	}
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 90 * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	base := strings.TrimRight(envOr("STELLAV_OLLAMA_BASE", defaultOllamaBase), "/")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return openaiMessage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return openaiMessage{}, err
	}
	defer resp.Body.Close()
	var parsed ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return openaiMessage{}, err
	}
	if parsed.Error != "" {
		return openaiMessage{}, fmt.Errorf("ollama: %s", parsed.Error)
	}
	content, _ := json.Marshal(parsed.Message.Content)
	out := openaiMessage{Role: "assistant", Content: content}
	for i, call := range parsed.Message.ToolCalls {
		args := strings.TrimSpace(string(call.Function.Arguments))
		if args == "" {
			args = "{}"
		} else if json.Valid([]byte(args)) && len(args) > 0 && args[0] != '"' {
			// already a JSON object/string payload
		} else {
			var asString string
			if json.Unmarshal(call.Function.Arguments, &asString) == nil {
				args = asString
			}
		}
		out.ToolCalls = append(out.ToolCalls, openaiToolCall{
			ID:    newOpenAIID("call_"),
			Type:  "function",
			Index: i,
			Function: openaiToolFunction{
				Name:      call.Function.Name,
				Arguments: args,
			},
		})
	}
	return out, nil
}

func tensorDeviceLabel() string {
	if os.Getenv("STELLAV_TENSOR_DEVICE") != "" {
		return os.Getenv("STELLAV_TENSOR_DEVICE")
	}
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "" {
		return "cuda"
	}
	return "cpu+vector"
}
