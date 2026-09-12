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
