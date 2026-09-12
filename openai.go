package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	openAIModel15   = "stella-v-1.5b"
	openAIModel3    = "stella-v-3b"
	openAIModelAuto = "stella-v"
	openAIModelLoc  = "stella-v-local"
)

type openaiErrorBody struct {
	Error openaiError `json:"error"`
}

type openaiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

type openaiModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type openaiMessage struct {
	Role       string           `json:"role"`
	Content    json.RawMessage  `json:"content,omitempty"`
	Name       string           `json:"name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []openaiToolCall `json:"tool_calls,omitempty"`
}

type openaiToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Index    int                `json:"index,omitempty"`
	Function openaiToolFunction `json:"function"`
}

type openaiToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openaiTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	} `json:"function"`
}

type openaiChatRequest struct {
	Model       string          `json:"model"`
	Messages    []openaiMessage `json:"messages"`
	Stream      bool            `json:"stream"`
	Temperature *float64        `json:"temperature"`
	MaxTokens   int             `json:"max_tokens"`
	Tools       []openaiTool    `json:"tools"`
	Input       json.RawMessage `json:"input,omitempty"`
	Prompt      string          `json:"prompt,omitempty"`
}

type openaiChatResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []openaiChatChoice `json:"choices"`
	Usage   openaiUsage        `json:"usage"`
}

type openaiChatChoice struct {
	Index        int           `json:"index"`
	Message      openaiMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openaiResponseAPI struct {
	ID         string          `json:"id"`
	Object     string          `json:"object"`
	CreatedAt  int64           `json:"created_at"`
	Status     string          `json:"status"`
	Model      string          `json:"model"`
	Output     []openaiOutItem `json:"output"`
	OutputText string          `json:"output_text"`
	Usage      map[string]int  `json:"usage"`
}

type openaiOutItem struct {
	ID      string             `json:"id"`
	Type    string             `json:"type"`
	Role    string             `json:"role,omitempty"`
	Content []openaiOutContent `json:"content,omitempty"`
	Status  string             `json:"status,omitempty"`
}

type openaiOutContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func stellaOpenAIModels() []openaiModel {
	created := time.Now().Unix()
	return []openaiModel{
		{ID: openAIModelAuto, Object: "model", Created: created, OwnedBy: "stella-v"},
		{ID: openAIModel15, Object: "model", Created: created, OwnedBy: "stella-v"},
		{ID: openAIModel3, Object: "model", Created: created, OwnedBy: "stella-v"},
		{ID: openAIModelLoc, Object: "model", Created: created, OwnedBy: "stella-v"},
	}
}

func mapOpenAIModel(id string) (reasonSize string, localOnly bool) {
	key := strings.ToLower(strings.TrimSpace(id))
	key = strings.TrimPrefix(key, "stella/")
	key = strings.TrimPrefix(key, "stellav/")
	switch key {
	case openAIModel3, "stella-3b", "3b", "qwen2.5:3b":
		return Reason3B, false
	case openAIModelLoc, "stella-local", "local":
		return Reason1p5B, true
	default:
		return Reason1p5B, false
	}
}

func flattenMessageContent(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		return strings.TrimSpace(asString)
	}
	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err == nil {
		var b strings.Builder
		for _, part := range parts {
			if text, ok := part["text"].(string); ok {
				if b.Len() > 0 {
					b.WriteByte('\n')
				}
				b.WriteString(text)
				continue
			}
			if text, ok := part["content"].(string); ok {
				if b.Len() > 0 {
					b.WriteByte('\n')
				}
				b.WriteString(text)
			}
		}
		return strings.TrimSpace(b.String())
	}
	return strings.TrimSpace(string(raw))
}

func lastUserText(messages []openaiMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		role := strings.ToLower(strings.TrimSpace(messages[i].Role))
		if role == "user" || role == "human" {
			if text := flattenMessageContent(messages[i].Content); text != "" {
				return text
			}
		}
	}
	if len(messages) == 0 {
		return ""
	}
	return flattenMessageContent(messages[len(messages)-1].Content)
}

func conversationTranscript(messages []openaiMessage) string {
	var b strings.Builder
	for _, msg := range messages {
		role := strings.TrimSpace(msg.Role)
		if role == "" {
			role = "user"
		}
		text := flattenMessageContent(msg.Content)
		if text == "" && len(msg.ToolCalls) > 0 {
			text = "[tool_calls]"
		}
		if text == "" {
			continue
		}
		fmt.Fprintf(&b, "%s: %s\n", role, text)
	}
	return strings.TrimSpace(b.String())
}

func newOpenAIID(prefix string) string {
	var buf [8]byte
	_, _ = rand.Read(buf[:])
	return prefix + hex.EncodeToString(buf[:])
}

func writeOpenAIError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(openaiErrorBody{Error: openaiError{
		Message: message,
		Type:    "invalid_request_error",
	}})
}

func (s *guiServer) handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "GET required")
		return
	}
	models := stellaOpenAIModels()
	path := strings.TrimPrefix(r.URL.Path, "/v1/models")
	path = strings.Trim(path, "/")
	if path != "" {
		for _, model := range models {
			if model.ID == path {
				writeJSON(w, model)
				return
			}
		}
		writeOpenAIError(w, http.StatusNotFound, "model not found: "+path)
		return
	}
	writeJSON(w, map[string]any{"object": "list", "data": models})
}

func (s *guiServer) handleOpenAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req openaiChatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 8<<20)).Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 && strings.TrimSpace(req.Prompt) != "" {
		raw, _ := json.Marshal(req.Prompt)
		req.Messages = []openaiMessage{{Role: "user", Content: raw}}
	}
	result, err := s.completeOpenAI(req)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if req.Stream {
		writeChatStream(w, result)
		return
	}
	writeJSON(w, result)
}

func (s *guiServer) handleOpenAICompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req openaiChatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 8<<20)).Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		raw, _ := json.Marshal(req.Prompt)
		req.Messages = []openaiMessage{{Role: "user", Content: raw}}
	}
	result, err := s.completeOpenAI(req)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, map[string]any{
		"id":      result.ID,
		"object":  "text_completion",
		"created": result.Created,
		"model":   result.Model,
		"choices": []map[string]any{{
			"index":         0,
			"text":          flattenMessageContent(result.Choices[0].Message.Content),
			"finish_reason": result.Choices[0].FinishReason,
		}},
		"usage": result.Usage,
	})
}

func (s *guiServer) handleOpenAIResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req openaiChatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 8<<20)).Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		req.Messages = messagesFromResponsesInput(req.Input)
	}
	if len(req.Messages) == 0 && strings.TrimSpace(req.Prompt) != "" {
		raw, _ := json.Marshal(req.Prompt)
		req.Messages = []openaiMessage{{Role: "user", Content: raw}}
	}
	result, err := s.completeOpenAI(req)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error())
		return
	}
	text := flattenMessageContent(result.Choices[0].Message.Content)
	writeJSON(w, openaiResponseAPI{
		ID:         strings.Replace(result.ID, "chatcmpl-", "resp_", 1),
		Object:     "response",
		CreatedAt:  result.Created,
		Status:     "completed",
		Model:      result.Model,
		OutputText: text,
		Output: []openaiOutItem{{
			ID:     newOpenAIID("msg_"),
			Type:   "message",
			Role:   "assistant",
			Status: "completed",
			Content: []openaiOutContent{{
				Type: "output_text",
				Text: text,
			}},
		}},
		Usage: map[string]int{
			"input_tokens":  result.Usage.PromptTokens,
			"output_tokens": result.Usage.CompletionTokens,
			"total_tokens":  result.Usage.TotalTokens,
		},
	})
}

func (s *guiServer) handleOpenAIEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req struct {
		Model string          `json:"model"`
		Input json.RawMessage `json:"input"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	inputs := []string{}
	var one string
	if err := json.Unmarshal(req.Input, &one); err == nil {
		inputs = []string{one}
	} else {
		_ = json.Unmarshal(req.Input, &inputs)
	}
	engine := NewTensorEngine()
	data := make([]map[string]any, 0, len(inputs))
	for i, text := range inputs {
		vec := engine.Embed(text)
		floats := make([]float64, len(vec))
		for j, v := range vec {
			floats[j] = float64(v)
		}
		data = append(data, map[string]any{
			"object":    "embedding",
			"index":     i,
			"embedding": floats,
		})
	}
	writeJSON(w, map[string]any{
		"object": "list",
		"model":  firstNonEmpty(req.Model, openAIModelAuto),
		"data":   data,
		"usage":  map[string]int{"prompt_tokens": 0, "total_tokens": 0},
	})
}

func (s *guiServer) completeOpenAI(req openaiChatRequest) (openaiChatResponse, error) {
	modelID := firstNonEmpty(req.Model, openAIModelAuto)
	reasonSize, localOnly := mapOpenAIModel(modelID)
	userText := lastUserText(req.Messages)
	if strings.TrimSpace(userText) == "" {
		userText = conversationTranscript(req.Messages)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	cfg := s.runtime.normalized()
	cfg.ReasonSize = reasonSize
	if localOnly {
		cfg.Mode = RuntimeLocalOnly
		cfg.LiveSearch = false
	}
	if looksLikeArtifactRequest(userText) {
		if write, ok := findWriteTool(req.Tools); ok {
			return s.completeArtifactWrite(cfg, req, userText, write)
		}
		pred := NewReplyEngine(s.model, cfg).Reply(userText)
		return chatFromPrediction(modelID, userText, pred), nil
	}
	if len(req.Tools) > 0 && !localOnly {
		chat, err := s.completeWithTools(cfg, req, userText)
		if err == nil {
			return chat, nil
		}
	}

	engine := NewReplyEngine(s.model, cfg)
	if transcript := conversationTranscript(req.Messages); transcript != "" && !strings.EqualFold(transcript, "user: "+userText) {
		userText = transcript + "\n\nLatest user message: " + lastUserText(req.Messages)
	}
	pred := engine.Reply(userText)
	return chatFromPrediction(modelID, userText, pred), nil
}

func chatFromPrediction(modelID, userText string, pred Prediction) openaiChatResponse {
	content, _ := json.Marshal(pred.Text)
	promptTok := utf8.RuneCountInString(userText)
	compTok := utf8.RuneCountInString(pred.Text)
	return openaiChatResponse{
		ID:      newOpenAIID("chatcmpl-"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []openaiChatChoice{{
			Index: 0,
			Message: openaiMessage{
				Role:    "assistant",
				Content: content,
			},
			FinishReason: "stop",
		}},
		Usage: openaiUsage{
			PromptTokens:     promptTok,
			CompletionTokens: compTok,
			TotalTokens:      promptTok + compTok,
		},
	}
}

func (s *guiServer) completeArtifactWrite(cfg ConquerorConfig, req openaiChatRequest, userText string, write openaiTool) (openaiChatResponse, error) {
	transcript := conversationTranscript(req.Messages)
	query := researchQueryFrom(userText, transcript)
	if cfg.LiveSearch && cfg.Mode != RuntimeLocalOnly {
		_, _ = indexScienceOpenIntoModel(s.model, query, 6)
	}
	hits := NewTensorEngine().Search(query, s.model.Chunks, nil, 6)
	if len(hits) == 0 {
		hits = NewTensorEngine().Search(userText, s.model.Chunks, s.model.Samples, 6)
	}
	if ev := selectArtifactEvidence(userText, transcript, hits); ev.Chunk.DOI != "" || ev.Chunk.Text != "" {
		hits = prependHit(hits, ev)
	}
	local := s.model.Predict(query)
	code := ""
	if cfg.Mode != RuntimeLocalOnly && ollamaAlive() {
		prompt := s.model.BuildCodePrompt(userText, local, hits)
		n := req.MaxTokens
		if n < 1024 {
			n = 1536
		}
		if n > 2048 {
			n = 2048
		}
		timeout := cfg.Timeout
		if timeout <= 0 {
			timeout = 90 * time.Second
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if reply, err := queryOllamaPredict(ctx, ollamaModelFor(cfg.ReasonSize), prompt, n); err == nil {
			code = extractFencedCode(reply)
			if code == "" && looksLikeSource(reply) {
				code = strings.TrimSpace(reply)
			}
		}
	}
	if strings.TrimSpace(code) == "" {
		code = fallbackResearchScript(userText, hits, transcript)
	}
	path := inferArtifactPath(userText)
	note, _ := json.Marshal(artifactAssistantNote(path, hits))
	msg := openaiMessage{
		Role:      "assistant",
		Content:   note,
		ToolCalls: []openaiToolCall{synthesizeWriteCall(write, path, code)},
	}
	modelID := firstNonEmpty(req.Model, openAIModelAuto)
	promptTok := utf8.RuneCountInString(userText)
	compTok := utf8.RuneCountInString(code)
	return openaiChatResponse{
		ID:      newOpenAIID("chatcmpl-"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []openaiChatChoice{{
			Index:        0,
			Message:      msg,
			FinishReason: "tool_calls",
		}},
		Usage: openaiUsage{
			PromptTokens:     promptTok,
			CompletionTokens: compTok,
			TotalTokens:      promptTok + compTok,
		},
	}, nil
}

func (s *guiServer) completeWithTools(cfg ConquerorConfig, req openaiChatRequest, userText string) (openaiChatResponse, error) {
	rag := ""
	if looksMedical(userText) || looksLikeArtifactRequest(userText) {
		searchQ := userText
		if looksLikeArtifactRequest(userText) {
			searchQ = researchQueryFrom(userText, conversationTranscript(req.Messages))
		}
		hits := NewTensorEngine().Search(searchQ, s.model.Chunks, s.model.Samples, 4)
		if cfg.LiveSearch {
			_, _ = indexScienceOpenIntoModel(s.model, searchQ, 4)
			hits = NewTensorEngine().Search(searchQ, s.model.Chunks, s.model.Samples, 4)
		}
		if len(hits) > 0 {
			rag = "Retrieved ScienceOpen evidence (use this to ground any script or answer; do not stop at a summary if the user asked for a file):\n" + formatRankedPassages(hits, 4)
		}
	}
	reply, err := queryOllamaChat(cfg, req.Messages, req.Tools, rag, req.MaxTokens)
	if err != nil {
		return openaiChatResponse{}, err
	}
	modelID := firstNonEmpty(req.Model, openAIModelAuto)
	finish := "stop"
	if len(reply.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	promptTok := utf8.RuneCountInString(userText)
	compTok := utf8.RuneCountInString(flattenMessageContent(reply.Content))
	return openaiChatResponse{
		ID:      newOpenAIID("chatcmpl-"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []openaiChatChoice{{
			Index:        0,
			Message:      reply,
			FinishReason: finish,
		}},
		Usage: openaiUsage{
			PromptTokens:     promptTok,
			CompletionTokens: compTok,
			TotalTokens:      promptTok + compTok,
		},
	}, nil
}

func writeChatStream(w http.ResponseWriter, result openaiChatResponse) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)
	writeEvent := func(v any) {
		raw, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", raw)
		if flusher != nil {
			flusher.Flush()
		}
	}
	msg := result.Choices[0].Message
	writeEvent(map[string]any{
		"id":      result.ID,
		"object":  "chat.completion.chunk",
		"created": result.Created,
		"model":   result.Model,
		"choices": []map[string]any{{
			"index": 0,
			"delta": map[string]any{"role": "assistant"},
		}},
	})
	if len(msg.ToolCalls) > 0 {
		writeEvent(map[string]any{
			"id":      result.ID,
			"object":  "chat.completion.chunk",
			"created": result.Created,
			"model":   result.Model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{"tool_calls": msg.ToolCalls},
				"finish_reason": nil,
			}},
		})
		writeEvent(map[string]any{
			"id":      result.ID,
			"object":  "chat.completion.chunk",
			"created": result.Created,
			"model":   result.Model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": "tool_calls",
			}},
		})
	} else {
		text := flattenMessageContent(msg.Content)
		const chunk = 48
		for i := 0; i < len(text); i += chunk {
			end := i + chunk
			if end > len(text) {
				end = len(text)
			}
			writeEvent(map[string]any{
				"id":      result.ID,
				"object":  "chat.completion.chunk",
				"created": result.Created,
				"model":   result.Model,
				"choices": []map[string]any{{
					"index": 0,
					"delta": map[string]any{"content": text[i:end]},
				}},
			})
		}
		writeEvent(map[string]any{
			"id":      result.ID,
			"object":  "chat.completion.chunk",
			"created": result.Created,
			"model":   result.Model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": "stop",
			}},
		})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

func messagesFromResponsesInput(raw json.RawMessage) []openaiMessage {
	if len(raw) == 0 {
		return nil
	}
	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil && strings.TrimSpace(asString) != "" {
		body, _ := json.Marshal(asString)
		return []openaiMessage{{Role: "user", Content: body}}
	}
	var msgs []openaiMessage
	if err := json.Unmarshal(raw, &msgs); err == nil && len(msgs) > 0 {
		for i := range msgs {
			if msgs[i].Role == "" {
				msgs[i].Role = "user"
			}
		}
		return msgs
	}
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil
	}
	out := make([]openaiMessage, 0, len(items))
	for _, item := range items {
		role, _ := item["role"].(string)
		if role == "" {
			role = "user"
		}
		text := ""
		switch content := item["content"].(type) {
		case string:
			text = content
		case []any:
			var b strings.Builder
			for _, part := range content {
				if m, ok := part.(map[string]any); ok {
					if t, ok := m["text"].(string); ok {
						if b.Len() > 0 {
							b.WriteByte('\n')
						}
						b.WriteString(t)
					}
				}
			}
			text = b.String()
		}
		if strings.TrimSpace(text) == "" {
			continue
		}
		body, _ := json.Marshal(text)
		out = append(out, openaiMessage{Role: role, Content: body})
	}
	return out
}
