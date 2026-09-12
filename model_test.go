package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/powerengine/paccel/libpaccel"
)

func TestReasonSizeAliases(t *testing.T) {
	if NormalizeReasonSize("3b") != Reason3B {
		t.Fatalf("3b should map to 3b")
	}
	if NormalizeReasonSize("qwen2.5:1.5b") != Reason1p5B {
		t.Fatalf("1.5b alias should map to 1.5b")
	}
	if ollamaModelFor(Reason3B) == ollamaModelFor(Reason1p5B) {
		t.Fatalf("1.5b and 3b should select different Ollama models")
	}
}

func TestTensorEngineRanksMatchingPreprint(t *testing.T) {
	m := NewStellaModel()
	m.AddChunk(
		"10.14293/example.immuno",
		"Checkpoint inhibitors in melanoma",
		"PD-1 blockade can produce durable responses in melanoma immunotherapy trials.",
		scienceOpenURL("10.14293/example.immuno"),
		"scienceopen",
	)
	m.AddChunk(
		"10.14293/example.soil",
		"Agricultural soil microbiome",
		"Crop yield depends on soil nitrogen cycling in wheat fields far from oncology clinics.",
		scienceOpenURL("10.14293/example.soil"),
		"scienceopen",
	)
	hits := NewTensorEngine().Search("cancer immunotherapy checkpoint PD-1 melanoma", m.Chunks, nil, 2)
	if len(hits) == 0 {
		t.Fatal("expected ranked preprint hits")
	}
	if !strings.Contains(strings.ToLower(hits[0].Chunk.Title), "checkpoint") {
		t.Fatalf("expected oncology preprint first, got %q", hits[0].Chunk.Title)
	}
	reply := m.Predict("what do checkpoint inhibitors do in melanoma immunotherapy?")
	if !strings.Contains(strings.ToLower(reply.Text), "pd-1") && !strings.Contains(strings.ToLower(reply.Text), "melanoma") {
		t.Fatalf("tensor-backed reply missing preprint content: %q", reply.Text)
	}
}

func TestMedicalBootstrapHasNoGameEngine(t *testing.T) {
	for _, sample := range BootstrapSamples() {
		blob := strings.ToLower(sample.Question + " " + sample.Answer)
		if strings.Contains(blob, "game engine") || strings.Contains(blob, "swapchain") || strings.Contains(blob, "gamepad") {
			t.Fatalf("bootstrap still contains game content: %#v", sample)
		}
	}
	reply := semanticFallback("what is a preprint")
	if !strings.Contains(strings.ToLower(reply), "peer review") {
		t.Fatalf("expected preprint fallback, got %q", reply)
	}
}

func TestStellaLearnsAndExports(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "stella.json")
	m := NewStellaModel()
	for _, sample := range BootstrapSamples() {
		m.AddSample(sample.Question, sample.Answer)
	}
	m.AddSample("what is the secret build target", "The build target is Stella.")
	if err := m.Save(modelPath); err != nil {
		t.Fatalf("save model: %v", err)
	}
	loaded, err := LoadModel(modelPath)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	reply := loaded.Predict("what is the secret build target")
	if !strings.Contains(strings.ToLower(reply.Text), "stella") {
		t.Fatalf("reply %q does not contain learned answer", reply.Text)
	}

	safePath := filepath.Join(dir, "stella.safetensors")
	if err := loaded.ExportSafetensors(safePath); err != nil {
		t.Fatalf("export safetensors: %v", err)
	}
	if info, err := os.Stat(safePath); err != nil || info.Size() == 0 {
		t.Fatalf("safetensors not written: size=%v err=%v", info, err)
	}

	paccelPath := filepath.Join(dir, "stella.paccel")
	if err := loaded.ExportPAccel(paccelPath, 4, 32); err != nil {
		t.Fatalf("export paccel: %v", err)
	}
	pkg, err := libpaccel.ReadPackage(paccelPath)
	if err != nil {
		t.Fatalf("read paccel: %v", err)
	}
	if len(pkg.Records) == 0 {
		t.Fatalf("paccel package has no records")
	}

	onnxPath := filepath.Join(dir, "stella.onnx")
	if err := loaded.ExportONNX(onnxPath); err != nil {
		t.Fatalf("export onnx: %v", err)
	}
	if info, err := os.Stat(onnxPath); err != nil || info.Size() == 0 {
		t.Fatalf("onnx not written: size=%v err=%v", info, err)
	}
}
