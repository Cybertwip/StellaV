package main

import (
	"encoding/json"
	"net/http"
	"strings"
)

func stellaAgentTools() []openaiTool {
	search := openaiTool{Type: "function"}
	search.Function.Name = "scienceopen_search"
	search.Function.Description = "Search ScienceOpen medical preprints and return ranked titles, DOIs, and abstracts."
	search.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"rows":{"type":"integer"}},"required":["query"]}`)

	retrieve := openaiTool{Type: "function"}
	retrieve.Function.Name = "retrieve_knowledge"
	retrieve.Function.Description = "Retrieve ranked passages from Stella V's local tensor memory of indexed ScienceOpen preprints."
	retrieve.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"k":{"type":"integer"}},"required":["query"]}`)

	index := openaiTool{Type: "function"}
	index.Function.Name = "index_preprint"
	index.Function.Description = "Index ScienceOpen preprints for a query or DOI into Stella V's persistent tensor memory."
	index.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"rows":{"type":"integer"}},"required":["query"]}`)
	return []openaiTool{search, retrieve, index}
}

func (s *guiServer) executeStellaTool(name, arguments string) string {
	var args map[string]any
	_ = json.Unmarshal([]byte(arguments), &args)
	query, _ := args["query"].(string)
	query = strings.TrimSpace(query)
	rows := 6
	if v, ok := args["rows"].(float64); ok && v > 0 {
		rows = int(v)
	}
	k := 6
	if v, ok := args["k"].(float64); ok && v > 0 {
		k = int(v)
	}
	switch name {
	case "scienceopen_search":
		papers, err := SearchScienceOpen(query, rows)
		if err != nil {
			return err.Error()
		}
		raw, _ := json.Marshal(papers)
		return string(raw)
	case "retrieve_knowledge":
		hits := NewTensorEngine().Search(query, s.model.Chunks, s.model.Samples, k)
		type hit struct {
			Title string  `json:"title"`
			DOI   string  `json:"doi"`
			URL   string  `json:"url"`
			Text  string  `json:"text"`
			Score float64 `json:"score"`
		}
		out := make([]hit, 0, len(hits))
		for _, item := range hits {
			out = append(out, hit{Title: item.Chunk.Title, DOI: item.Chunk.DOI, URL: item.Chunk.URL, Text: item.Chunk.Text, Score: item.Score})
		}
		raw, _ := json.Marshal(out)
		return string(raw)
	case "index_preprint":
		n, err := indexScienceOpenIntoModel(s.model, query, rows)
		if err != nil {
			return err.Error()
		}
		_ = s.model.Save(s.modelPath)
		raw, _ := json.Marshal(map[string]any{"indexed": n, "chunks": len(s.model.Chunks)})
		return string(raw)
	default:
		return "unknown tool: " + name
	}
}

func (s *guiServer) handleAgentRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var req struct {
		Input    string `json:"input"`
		Model    string `json:"model"`
		MaxSteps int    `json:"max_steps"`
		Reason   string `json:"reason"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.Input) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "input is required")
		return
	}
	if req.MaxSteps <= 0 {
		req.MaxSteps = 4
	}
	if req.MaxSteps > 8 {
		req.MaxSteps = 8
	}
	reasonSize, localOnly := mapOpenAIModel(firstNonEmpty(req.Model, req.Reason, openAIModelAuto))
	s.mu.Lock()
	defer s.mu.Unlock()
	cfg := s.runtime.normalized()
	cfg.ReasonSize = reasonSize
	if localOnly {
		pred := NewReplyEngine(s.model, cfg).Reply(req.Input)
		writeJSON(w, map[string]any{"output": pred.Text, "source": pred.Source, "steps": []any{}})
		return
	}

	userRaw, _ := json.Marshal(req.Input)
	messages := []openaiMessage{{Role: "user", Content: userRaw}}
	tools := stellaAgentTools()
	steps := []map[string]any{}
	final := ""
	for step := 0; step < req.MaxSteps; step++ {
		reply, err := s.completeWithTools(cfg, openaiChatRequest{Model: firstNonEmpty(req.Model, openAIModelAuto), Messages: messages, Tools: tools}, req.Input)
		if err != nil {
			pred := NewReplyEngine(s.model, cfg).Reply(req.Input)
			final = pred.Text
			break
		}
		msg := reply.Choices[0].Message
		if len(msg.ToolCalls) == 0 {
			final = flattenMessageContent(msg.Content)
			break
		}
		messages = append(messages, msg)
		for _, call := range msg.ToolCalls {
			result := s.executeStellaTool(call.Function.Name, call.Function.Arguments)
			steps = append(steps, map[string]any{
				"tool":      call.Function.Name,
				"arguments": call.Function.Arguments,
				"result":    truncateRunes(result, 4000),
			})
			body, _ := json.Marshal(result)
			messages = append(messages, openaiMessage{
				Role:       "tool",
				ToolCallID: call.ID,
				Content:    body,
			})
		}
	}
	if strings.TrimSpace(final) == "" {
		final = NewReplyEngine(s.model, cfg).Reply(req.Input).Text
	}
	writeJSON(w, map[string]any{
		"object": "stella.agent_run",
		"output": final,
		"steps":  steps,
		"model":  firstNonEmpty(req.Model, openAIModelAuto),
	})
}

func truncateRunes(text string, limit int) string {
	if limit <= 0 || len(text) <= limit {
		return text
	}
	if limit <= 3 {
		return text[:limit]
	}
	return text[:limit-3] + "..."
}
