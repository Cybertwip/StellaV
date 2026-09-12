package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testServer(t *testing.T) *guiServer {
	t.Helper()
	m := NewStellaModel()
	for _, sample := range BootstrapSamples() {
		m.AddSample(sample.Question, sample.Answer)
	}
	m.AddChunk(
		"10.14293/example.trial",
		"Randomization in clinical trials",
		"Randomization assigns participants by chance so confounders are balanced in expectation.",
		scienceOpenURL("10.14293/example.trial"),
		"scienceopen",
	)
	return &guiServer{
		model:     m,
		modelPath: t.TempDir() + "/stella.json",
		runtime:   ConquerorConfig{Mode: RuntimeLocalOnly, ReasonSize: Reason1p5B, LiveSearch: false},
	}
}

func TestOpenAIModelsList(t *testing.T) {
	srv := httptest.NewServer(testServer(t).routes())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var payload struct {
		Data []openaiModel `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	ids := map[string]bool{}
	for _, model := range payload.Data {
		ids[model.ID] = true
	}
	for _, want := range []string{openAIModelAuto, openAIModel15, openAIModel3, openAIModelLoc} {
		if !ids[want] {
			t.Fatalf("missing model %s", want)
		}
	}
}

func TestOpenAIChatCompletionsLocal(t *testing.T) {
	srv := httptest.NewServer(testServer(t).routes())
	defer srv.Close()
	body := []byte(`{"model":"stella-v-local","messages":[{"role":"user","content":"what is stella v"}]}`)
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, raw)
	}
	var payload openaiChatResponse
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	text := flattenMessageContent(payload.Choices[0].Message.Content)
	if !strings.Contains(strings.ToLower(text), "stella") {
		t.Fatalf("unexpected reply %q", text)
	}
	if payload.Object != "chat.completion" {
		t.Fatalf("object %s", payload.Object)
	}
}

func TestOpenAIResponsesAndStream(t *testing.T) {
	srv := httptest.NewServer(testServer(t).routes())
	defer srv.Close()
	body := []byte(`{"model":"stella-v-local","input":"what is a randomized trial?"}`)
	resp, err := http.Post(srv.URL+"/v1/responses", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var payload openaiResponseAPI
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	if payload.Status != "completed" || payload.OutputText == "" {
		t.Fatalf("bad responses payload %#v", payload)
	}

	streamBody := []byte(`{"model":"stella-v-local","stream":true,"messages":[{"role":"user","content":"what is stella v"}]}`)
	stream, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(streamBody))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Body.Close()
	raw, _ := io.ReadAll(stream.Body)
	if !strings.Contains(string(raw), "data: [DONE]") {
		t.Fatalf("missing SSE done: %s", raw)
	}
}

func TestOpenAIArtifactReturnsWriteToolCall(t *testing.T) {
	srv := testServer(t)
	srv.model.AddChunk(
		"10.14293/s2199-1006.1.sor-.ppsvhlh.v1",
		"mRNA Vaccine: The Next Generation Vaccine Revolution in Medical Science & Vaccinology",
		"RNA vaccines have potential as Novel therapeutic options for major disease such as cancer for development of personalized medicine.",
		scienceOpenURL("10.14293/s2199-1006.1.sor-.ppsvhlh.v1"),
		"scienceopen",
	)
	httpSrv := httptest.NewServer(srv.routes())
	defer httpSrv.Close()

	body := []byte(`{
		"model":"stella-v-local",
		"messages":[{"role":"user","content":"Write a python chempy script to begin a cancer vaccine."}],
		"tools":[{
			"type":"function",
			"function":{
				"name":"write",
				"description":"Write a file",
				"parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}
			}
		}]
	}`)
	resp, err := http.Post(httpSrv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, raw)
	}
	var payload openaiChatResponse
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason %s body %s", payload.Choices[0].FinishReason, raw)
	}
	calls := payload.Choices[0].Message.ToolCalls
	if len(calls) != 1 || calls[0].Function.Name != "write" {
		t.Fatalf("tool calls %#v", calls)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(args["file_path"], ".py") {
		t.Fatalf("path %q", args["file_path"])
	}
	if !strings.Contains(args["content"], "chempy") {
		t.Fatalf("script missing chempy: %s", args["content"][:min(200, len(args["content"]))])
	}
	if !strings.Contains(args["content"], "10.14293/s2199-1006.1.sor-.ppsvhlh.v1") {
		t.Fatalf("script missing DOI")
	}
	if !strings.Contains(args["content"], "NOT a vaccine") {
		t.Fatalf("script missing research disclaimer")
	}
	note := flattenMessageContent(payload.Choices[0].Message.Content)
	if !strings.Contains(strings.ToLower(note), "scienceopen") {
		t.Fatalf("assistant note should push retrieved evidence, got %q", note)
	}
}

func TestOpenAIFollowUpScriptUsesConversationDOI(t *testing.T) {
	srv := testServer(t)
	httpSrv := httptest.NewServer(srv.routes())
	defer httpSrv.Close()
	body := []byte(`{
		"model":"stella-v-local",
		"messages":[
			{"role":"user","content":"Write a python chempy script to begin a cancer vaccine."},
			{"role":"assistant","content":"ScienceOpen preprint: 10.14293/s2199-1006.1.sor-.ppsvhlh.v1\nPassage: RNA vaccines have potential as Novel therapeutic options for cancer."},
			{"role":"user","content":"Write the python script now, do that based on the research"}
		],
		"tools":[{
			"type":"function",
			"function":{
				"name":"write",
				"parameters":{"type":"object","properties":{"filePath":{"type":"string"},"content":{"type":"string"}},"required":["filePath","content"]}
			}
		}]
	}`)
	resp, err := http.Post(httpSrv.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, raw)
	}
	var payload openaiChatResponse
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected write tool call, got %s", raw)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(payload.Choices[0].Message.ToolCalls[0].Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if _, ok := args["filePath"]; !ok {
		t.Fatalf("expected camelCase filePath for OpenCode schema, got %#v", args)
	}
}

func TestMapOpenAIModel(t *testing.T) {
	size, local := mapOpenAIModel("stella-v-3b")
	if size != Reason3B || local {
		t.Fatalf("3b mapping: %s local=%v", size, local)
	}
	size, local = mapOpenAIModel("stella-v-local")
	if !local || size != Reason1p5B {
		t.Fatalf("local mapping: %s local=%v", size, local)
	}
}
