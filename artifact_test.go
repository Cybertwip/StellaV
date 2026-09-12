package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestLooksLikeArtifactRequest(t *testing.T) {
	yes := []string{
		"Write a python chempy script to begin a cancer vaccine.",
		"Write the python script now, do that based on the research",
		"generate a python script for mrna vaccine kinetics",
		"create foo.py from the preprint",
	}
	for _, text := range yes {
		if !looksLikeArtifactRequest(text) {
			t.Fatalf("expected artifact request: %q", text)
		}
	}
	no := []string{
		"what is a randomized trial?",
		"summarize this preprint about cancer vaccines",
		"create a vaccine for clinical use",
		"what do checkpoint inhibitors do in melanoma?",
	}
	for _, text := range no {
		if looksLikeArtifactRequest(text) {
			t.Fatalf("did not expect artifact request: %q", text)
		}
	}
}

func TestResearchQueryFromStripsScriptBoilerplate(t *testing.T) {
	q := researchQueryFrom("Write a python chempy script to begin a cancer vaccine.", "")
	low := strings.ToLower(q)
	if !strings.Contains(low, "cancer") || !strings.Contains(low, "vaccine") {
		t.Fatalf("expected cancer vaccine terms, got %q", q)
	}
	if strings.Contains(low, "python") || strings.Contains(low, "chempy") || strings.Contains(low, "script") {
		t.Fatalf("boilerplate left in query: %q", q)
	}
}

func TestResearchQueryFromFollowUpUsesDOI(t *testing.T) {
	transcript := `assistant: ScienceOpen preprint: 10.14293/s2199-1006.1.sor-.ppsvhlh.v1
Passage: RNA vaccines have potential as novel therapeutic options for cancer.`
	q := researchQueryFrom("Write the python script now, do that based on the research", transcript)
	if !strings.Contains(q, "10.14293/s2199-1006.1.sor-.ppsvhlh.v1") {
		t.Fatalf("expected DOI in follow-up query, got %q", q)
	}
}

func TestInferArtifactPath(t *testing.T) {
	path := inferArtifactPath("Write a python chempy script to begin a cancer vaccine.")
	if !strings.HasSuffix(path, ".py") {
		t.Fatalf("expected .py path, got %q", path)
	}
	if !strings.Contains(path, "cancer") && !strings.Contains(path, "vaccine") {
		t.Fatalf("expected topic in filename, got %q", path)
	}
	named := inferArtifactPath("please write kinetics.py for the model")
	if named != "kinetics.py" {
		t.Fatalf("explicit filename: %q", named)
	}
}

func TestExtractFencedCode(t *testing.T) {
	got := extractFencedCode("Sure.\n```python\nprint(1)\n```\n")
	if got != "print(1)" {
		t.Fatalf("fenced extract: %q", got)
	}
	src := "#!/usr/bin/env python3\nimport os\n\ndef main():\n    pass\n"
	if extractFencedCode(src) != strings.TrimSpace(src) {
		t.Fatalf("source extract failed")
	}
	if extractFencedCode("Therefore, the answer is to use mRNA technology.") != "" {
		t.Fatalf("research dump should not count as code")
	}
}

func TestFindWriteToolAndKeys(t *testing.T) {
	camel := openaiTool{Type: "function"}
	camel.Function.Name = "write"
	camel.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"filePath":{"type":"string"},"content":{"type":"string"}},"required":["filePath","content"]}`)
	tool, ok := findWriteTool([]openaiTool{camel})
	if !ok || tool.Function.Name != "write" {
		t.Fatal("missing write tool")
	}
	pathKey, contentKey := writeToolArgKeys(tool)
	if pathKey != "filePath" || contentKey != "content" {
		t.Fatalf("keys %s %s", pathKey, contentKey)
	}

	snake := openaiTool{Type: "function"}
	snake.Function.Name = "write_file"
	snake.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}`)
	pathKey, contentKey = writeToolArgKeys(snake)
	if pathKey != "file_path" || contentKey != "content" {
		t.Fatalf("snake keys %s %s", pathKey, contentKey)
	}

	if _, ok := findWriteTool([]openaiTool{{Type: "function"}}); ok {
		t.Fatal("todoless empty tool should not match")
	}
	todo := openaiTool{Type: "function"}
	todo.Function.Name = "todowrite"
	if _, ok := findWriteTool([]openaiTool{todo}); ok {
		t.Fatal("todowrite should not be treated as a file write tool")
	}
}

func TestSynthesizeWriteCall(t *testing.T) {
	tool := openaiTool{Type: "function"}
	tool.Function.Name = "write"
	tool.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}}}`)
	call := synthesizeWriteCall(tool, "cancer_vaccine_research.py", "print('ok')")
	if call.Function.Name != "write" {
		t.Fatalf("name %s", call.Function.Name)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if args["file_path"] != "cancer_vaccine_research.py" || args["content"] != "print('ok')" {
		t.Fatalf("args %#v", args)
	}
}

func TestSelectArtifactEvidencePrefersMatchingPreprint(t *testing.T) {
	hits := []RankedChunk{
		{Chunk: KnowledgeChunk{Title: "Randomization in clinical trials", DOI: "10.14293/example.trial", Text: "Randomization assigns participants by chance.", Source: "scienceopen"}, Score: 0.8},
		{Chunk: KnowledgeChunk{Title: "mRNA Vaccine Revolution", DOI: "10.14293/s2199-1006.1.sor-.ppsvhlh.v1", Text: "RNA vaccines have potential as therapeutic options for cancer.", Source: "scienceopen"}, Score: 0.2},
	}
	ev := selectArtifactEvidence("Write a python chempy script to begin a cancer vaccine.", "", hits)
	if ev.Chunk.DOI != "10.14293/s2199-1006.1.sor-.ppsvhlh.v1" {
		t.Fatalf("expected vaccine preprint, got %q", ev.Chunk.DOI)
	}
}

func TestFallbackChempyScriptCitesDOI(t *testing.T) {
	hits := []RankedChunk{{
		Chunk: KnowledgeChunk{
			Title: "mRNA Vaccine: The Next Generation Vaccine Revolution in Medical Science & Vaccinology",
			DOI:   "10.14293/s2199-1006.1.sor-.ppsvhlh.v1",
			URL:   "https://www.scienceopen.com/hosted-document?doi=10.14293/s2199-1006.1.sor-.ppsvhlh.v1",
			Text:  "RNA vaccines have potential as Novel therapeutic options for major disease such as cancer for development of personalized medicine.",
		},
		Score: 0.9,
	}}
	script := fallbackResearchScript("Write a python chempy script to begin a cancer vaccine.", hits)
	for _, need := range []string{
		"chempy",
		"10.14293/s2199-1006.1.sor-.ppsvhlh.v1",
		"NOT a vaccine",
		"personalized",
		"ReactionSystem",
	} {
		if !strings.Contains(script, need) {
			t.Fatalf("fallback missing %q", need)
		}
	}
}

func TestBuildCodePromptAsksForCodeNotSummary(t *testing.T) {
	m := NewStellaModel()
	hits := []RankedChunk{{
		Chunk: KnowledgeChunk{Title: "mRNA vaccines", DOI: "10.14293/example", Text: "RNA vaccines for cancer."},
	}}
	prompt := m.BuildCodePrompt("Write a python chempy script to begin a cancer vaccine.", Prediction{}, hits)
	low := strings.ToLower(prompt)
	if !strings.Contains(low, "write the requested program") && !strings.Contains(low, "write the file") {
		t.Fatalf("code prompt missing write instruction")
	}
	if strings.Contains(prompt, "End with the DOIs you used") {
		t.Fatalf("code prompt should not use the Q&A closer")
	}
	if !strings.Contains(prompt, "10.14293/example") {
		t.Fatalf("expected passage DOI in code prompt")
	}
}
