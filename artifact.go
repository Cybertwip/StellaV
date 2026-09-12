package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

var (
	doiRE      = regexp.MustCompile(`10\.\d{4,9}/[-._;()/:A-Za-z0-9]+`)
	filenameRE = regexp.MustCompile(`(?i)\b(?:[\w.-]+/)*[\w.-]+\.(py|r|js|ts|go|rs|ipynb|sh|rb|jl|txt)\b`)
	fenceRE    = regexp.MustCompile("(?s)```(?:[A-Za-z0-9_+-]+)?\\s*\\n(.*?)```")
)

func looksLikeArtifactRequest(text string) bool {
	q := strings.ToLower(text)
	if strings.TrimSpace(q) == "" {
		return false
	}
	verbs := []string{"write", "create", "generate", "make", "draft", "scaffold", "emit", "produce"}
	nouns := []string{"script", "python", "chempy", "notebook", "module", "program", "source file", ".py", "code"}
	hasVerb := false
	for _, verb := range verbs {
		if strings.Contains(q, verb) {
			hasVerb = true
			break
		}
	}
	hasNoun := false
	for _, noun := range nouns {
		if strings.Contains(q, noun) {
			hasNoun = true
			break
		}
	}
	if hasVerb && hasNoun {
		return true
	}
	if strings.Contains(q, "write the script") || strings.Contains(q, "script now") {
		return true
	}
	if filenameRE.FindString(text) != "" && hasVerb {
		return true
	}
	return false
}

func researchQueryFrom(userText, transcript string) string {
	cleaned := stripArtifactBoilerplate(userText)
	if wordCount(cleaned) < 2 {
		cleaned = stripArtifactBoilerplate(transcript)
	}
	if wordCount(cleaned) < 2 {
		cleaned = keepMedicalTerms(userText + " " + transcript)
	}
	if doi := doiRE.FindString(transcript + " " + userText); doi != "" {
		cleaned = strings.TrimSpace(cleaned + " " + strings.TrimRight(doi, ".,;"))
	}
	cleaned = strings.Join(strings.Fields(cleaned), " ")
	if cleaned == "" {
		return strings.TrimSpace(userText)
	}
	return cleaned
}

func stripArtifactBoilerplate(text string) string {
	stop := map[string]bool{
		"write": true, "writing": true, "create": true, "creating": true, "generate": true,
		"generating": true, "make": true, "making": true, "draft": true, "scaffold": true,
		"python": true, "chempy": true, "script": true, "scripts": true, "file": true,
		"files": true, "code": true, "program": true, "now": true, "please": true,
		"based": true, "research": true, "the": true, "a": true, "an": true, "to": true,
		"for": true, "begin": true, "beginning": true, "start": true, "starting": true,
		"do": true, "that": true, "this": true, "with": true, "using": true, "use": true,
		"want": true, "wants": true, "user": true, "preprint": true, "passage": true,
		"passages": true, "therefore": true, "answer": true, "from": true, "found": true,
		"in": true, "of": true, "and": true, "or": true, "on": true, "it": true,
		"is": true, "be": true, "can": true, "should": true, "able": true, "just": true,
		"so": true, "those": true, "these": true, "i": true, "will": true, "get": true,
		"numpy": true, "pandas": true, "scipy": true, "jupyter": true, "notebook": true,
	}
	var kept []string
	for _, tok := range strings.Fields(strings.ToLower(text)) {
		tok = strings.Trim(tok, ".,;:!?\"'`")
		if tok == "" || stop[tok] {
			continue
		}
		if filenameRE.MatchString(tok) {
			continue
		}
		kept = append(kept, tok)
	}
	return strings.Join(kept, " ")
}

func keepMedicalTerms(text string) string {
	hints := []string{
		"cancer", "vaccine", "mrna", "rna", "immun", "oncolog", "tumor", "tumour",
		"antigen", "peptide", "trial", "therap", "patient", "clinic", "epidem",
	}
	q := strings.ToLower(text)
	var kept []string
	seen := map[string]bool{}
	for _, tok := range strings.Fields(q) {
		tok = strings.Trim(tok, ".,;:!?\"'`")
		for _, hint := range hints {
			if strings.Contains(tok, hint) && !seen[tok] {
				kept = append(kept, tok)
				seen[tok] = true
				break
			}
		}
	}
	return strings.Join(kept, " ")
}

func wordCount(text string) int {
	return len(strings.Fields(text))
}

func inferArtifactPath(userText string) string {
	if name := filenameRE.FindString(userText); name != "" {
		return name
	}
	lower := strings.ToLower(userText)
	stem := slugStem(researchQueryFrom(userText, ""))
	if stem == "" {
		stem = "research_prototype"
	}
	if strings.Contains(lower, "vaccine") || strings.Contains(lower, "therap") {
		if !strings.Contains(stem, "research") {
			stem += "_research"
		}
	}
	switch {
	case strings.Contains(lower, "notebook") || strings.Contains(lower, "ipynb"):
		return stem + ".ipynb"
	case strings.Contains(lower, "chempy") || strings.Contains(lower, "python") || strings.Contains(lower, "script"):
		return stem + ".py"
	case strings.Contains(lower, "golang") || strings.Contains(lower, " go "):
		return stem + ".go"
	default:
		return stem + ".py"
	}
}

func slugStem(text string) string {
	var parts []string
	for _, field := range strings.Fields(strings.ToLower(text)) {
		var b strings.Builder
		for _, r := range field {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				b.WriteRune(r)
			}
		}
		tok := b.String()
		if tok == "" || len(tok) < 2 {
			continue
		}
		parts = append(parts, tok)
		if len(parts) >= 4 {
			break
		}
	}
	if len(parts) == 0 {
		return ""
	}
	stem := strings.Join(parts, "_")
	if len(stem) > 48 {
		stem = stem[:48]
	}
	return stem
}

func extractFencedCode(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	if m := fenceRE.FindStringSubmatch(text); len(m) == 2 {
		return strings.TrimSpace(m[1])
	}
	if looksLikeSource(text) {
		return text
	}
	return ""
}

func looksLikeSource(text string) bool {
	q := strings.TrimSpace(text)
	if q == "" {
		return false
	}
	if strings.HasPrefix(q, "#!/") || strings.HasPrefix(q, "\"\"\"") || strings.HasPrefix(q, "'''") {
		return true
	}
	lines := strings.Split(q, "\n")
	hits := 0
	for i, line := range lines {
		if i > 40 {
			break
		}
		trim := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trim, "import ") || strings.HasPrefix(trim, "from "):
			hits += 2
		case strings.HasPrefix(trim, "def ") || strings.HasPrefix(trim, "class "):
			hits += 2
		case strings.HasPrefix(trim, "function ") || strings.HasPrefix(trim, "package "):
			hits += 2
		case strings.HasPrefix(trim, "#") && i < 8:
			hits++
		}
	}
	return hits >= 2
}

func findWriteTool(tools []openaiTool) (openaiTool, bool) {
	prefer := []string{"write", "write_file", "writefile", "create_file", "createfile"}
	byName := map[string]openaiTool{}
	for _, tool := range tools {
		name := strings.ToLower(strings.TrimSpace(tool.Function.Name))
		if name == "" {
			continue
		}
		byName[name] = tool
	}
	for _, name := range prefer {
		if tool, ok := byName[name]; ok {
			return tool, true
		}
	}
	for name, tool := range byName {
		if strings.Contains(name, "todo") {
			continue
		}
		if strings.Contains(name, "write") || strings.Contains(name, "create_file") {
			return tool, true
		}
	}
	return openaiTool{}, false
}

func writeToolArgKeys(tool openaiTool) (pathKey, contentKey string) {
	pathKey, contentKey = "file_path", "content"
	if len(tool.Function.Parameters) == 0 {
		return pathKey, contentKey
	}
	var schema struct {
		Properties map[string]json.RawMessage `json:"properties"`
		Required   []string                   `json:"required"`
	}
	if json.Unmarshal(tool.Function.Parameters, &schema) != nil || len(schema.Properties) == 0 {
		return pathKey, contentKey
	}
	pathAliases := []string{"file_path", "filePath", "path", "filename", "file", "target_file", "targetFile"}
	contentAliases := []string{"content", "contents", "text", "body", "file_text", "fileText"}
	if key := firstSchemaKey(schema.Properties, schema.Required, pathAliases); key != "" {
		pathKey = key
	}
	if key := firstSchemaKey(schema.Properties, schema.Required, contentAliases); key != "" {
		contentKey = key
	}
	return pathKey, contentKey
}

func firstSchemaKey(props map[string]json.RawMessage, required, aliases []string) string {
	lower := map[string]string{}
	for key := range props {
		lower[strings.ToLower(key)] = key
	}
	for _, req := range required {
		if orig, ok := lower[strings.ToLower(req)]; ok {
			for _, alias := range aliases {
				if strings.EqualFold(req, alias) {
					return orig
				}
			}
		}
	}
	for _, alias := range aliases {
		if orig, ok := lower[strings.ToLower(alias)]; ok {
			return orig
		}
	}
	return ""
}

func synthesizeWriteCall(tool openaiTool, path, content string) openaiToolCall {
	pathKey, contentKey := writeToolArgKeys(tool)
	raw, _ := json.Marshal(map[string]string{
		pathKey:    path,
		contentKey: content,
	})
	return openaiToolCall{
		ID:    newOpenAIID("call_"),
		Type:  "function",
		Index: 0,
		Function: openaiToolFunction{
			Name:      tool.Function.Name,
			Arguments: string(raw),
		},
	}
}

func prependHit(hits []RankedChunk, ev RankedChunk) []RankedChunk {
	out := make([]RankedChunk, 0, len(hits)+1)
	out = append(out, ev)
	for _, hit := range hits {
		if hit.Chunk.DOI != "" && hit.Chunk.DOI == ev.Chunk.DOI {
			continue
		}
		out = append(out, hit)
	}
	return out
}

func selectArtifactEvidence(userText, transcript string, hits []RankedChunk) RankedChunk {
	blob := userText + " " + transcript
	var dois []string
	for _, raw := range doiRE.FindAllString(blob, 8) {
		dois = append(dois, strings.TrimRight(raw, ".,;"))
	}
	for _, doi := range dois {
		for _, hit := range hits {
			if hit.Chunk.DOI != "" && (strings.Contains(hit.Chunk.DOI, doi) || strings.Contains(doi, hit.Chunk.DOI)) {
				return hit
			}
		}
	}
	query := strings.ToLower(researchQueryFrom(userText, transcript))
	terms := strings.Fields(query)
	best := RankedChunk{}
	bestScore := -1.0
	for _, hit := range hits {
		blob := strings.ToLower(hit.Chunk.Title + " " + hit.Chunk.Text + " " + hit.Chunk.DOI)
		score := hit.Score
		for _, term := range terms {
			if len(term) >= 4 && strings.Contains(blob, term) {
				score += 0.5
			}
		}
		if strings.EqualFold(hit.Chunk.Source, "scienceopen") && hit.Chunk.DOI != "" {
			score += 0.1
		}
		if score > bestScore {
			bestScore = score
			best = hit
		}
	}
	if bestScore >= 0 {
		return best
	}
	if len(dois) > 0 {
		return RankedChunk{Chunk: KnowledgeChunk{
			DOI:   dois[0],
			Title: "ScienceOpen preprint",
			Text:  singleLine(blob, 700),
			URL:   scienceOpenURL(dois[0]),
		}}
	}
	return RankedChunk{}
}

func artifactAssistantNote(path string, hits []RankedChunk) string {
	if len(hits) > 0 && strings.TrimSpace(hits[0].Chunk.Title+hits[0].Chunk.DOI) != "" {
		hit := hits[0]
		doi := strings.TrimSpace(hit.Chunk.DOI)
		title := singleLine(hit.Chunk.Title, 140)
		if doi == "" {
			doi = "DOI n/a"
		}
		if title == "" {
			title = "ScienceOpen preprint"
		}
		return fmt.Sprintf("Retrieved ScienceOpen preprint %s (%s). Drafting %s from that evidence for OpenCode to write. Computational research only, not a therapy.", title, doi, path)
	}
	return fmt.Sprintf("Drafting %s from retrieved research evidence for OpenCode to write. Computational research only, not a therapy.", path)
}

func (m *StellaModel) BuildCodePrompt(question string, local Prediction, hits []RankedChunk) string {
	var b strings.Builder
	b.WriteString("You are Stella V. The user asked for a file or script. Stella retrieves ScienceOpen preprints and pushes that evidence; you must WRITE THE REQUESTED PROGRAM, not summarize the papers.\n")
	b.WriteString("Rules:\n")
	b.WriteString("1. Output only the complete file in one markdown code fence.\n")
	b.WriteString("2. Ground comments, constants, and model structure in the passages. Cite the DOI in a module docstring.\n")
	b.WriteString("3. This is a computational research prototype. Not a real vaccine, not manufacturing, not diagnosis or treatment.\n")
	b.WriteString("4. If the user asked for Chempy, use the chempy library (Reaction / ReactionSystem, optional ODE kinetics) with placeholder rates.\n")
	b.WriteString("5. Use placeholder sequences and toy kinetics only. No wet-lab protocol.\n")
	b.WriteString("6. If evidence is thin, still write a clearly labeled prototype and say so in comments.\n\n")
	b.WriteString("User request:\n")
	b.WriteString(strings.TrimSpace(question))
	b.WriteByte('\n')
	if strings.TrimSpace(local.Text) != "" && local.Source != "fallback" && local.Source != "empty" {
		b.WriteString("\nLocal tensor-memory draft (optional):\n")
		b.WriteString(singleLine(local.Text, 280))
		b.WriteByte('\n')
	}
	b.WriteString("\nScienceOpen preprint passages:\n")
	b.WriteString(formatRankedPassages(hits, 4))
	b.WriteString("\n\nWrite the file now. Output only code.")
	return b.String()
}

func fallbackResearchScript(userText string, hits []RankedChunk, transcript ...string) string {
	trans := ""
	if len(transcript) > 0 {
		trans = transcript[0]
	}
	ev := selectArtifactEvidence(userText, trans, hits)
	lower := strings.ToLower(userText)
	title, doi, url, passage := "unindexed ScienceOpen preprint", "n/a", "", "No matching preprint was in tensor memory."
	if strings.TrimSpace(ev.Chunk.Title) != "" {
		title = singleLine(ev.Chunk.Title, 180)
	}
	if strings.TrimSpace(ev.Chunk.DOI) != "" {
		doi = strings.TrimSpace(ev.Chunk.DOI)
	}
	url = strings.TrimSpace(ev.Chunk.URL)
	if strings.TrimSpace(ev.Chunk.Text) != "" {
		passage = singleLine(ev.Chunk.Text, 700)
	}
	if strings.Contains(lower, "chempy") || (strings.Contains(lower, "python") && (strings.Contains(lower, "vaccine") || strings.Contains(lower, "kinetic") || strings.Contains(lower, "mrna"))) {
		return chempyResearchStub(title, doi, url, passage)
	}
	return pythonResearchStub(title, doi, url, passage)
}

func pythonResearchStub(title, doi, url, passage string) string {
	return fmt.Sprintf(`#!/usr/bin/env python3
"""Computational research prototype grounded in a ScienceOpen preprint.

Title: %s
DOI: %s
URL: %s

Passage (unreviewed preprint):
    %s

This is NOT a vaccine, NOT a wet-lab protocol, and NOT clinical advice.
Stella V only retrieved evidence; this file is a research sketch for local experimentation.
"""

from __future__ import annotations

EVIDENCE = {
    "title": %q,
    "doi": %q,
    "url": %q,
    "unreviewed": True,
}


def main() -> None:
    print("ScienceOpen DOI:", EVIDENCE["doi"])
    print("Title:", EVIDENCE["title"])
    print("Unreviewed preprint. Computational sketch only.")


if __name__ == "__main__":
    main()
`, title, doi, url, passage, title, doi, url)
}

func chempyResearchStub(title, doi, url, passage string) string {
	return fmt.Sprintf(`#!/usr/bin/env python3
"""Toy Chempy kinetics for a personalized mRNA cancer-vaccine *research sketch*.

Grounded in ScienceOpen preprint:
    %s
    DOI: %s
    URL: %s

Passage (unreviewed):
    %s

This is NOT a vaccine, NOT manufacturing instructions, and NOT medical advice.
Species and rate constants are placeholders for a computational prototype:
    mRNA -> antigen protein -> immune-activation marker
"""

from __future__ import annotations

DOI = %q

try:
    from chempy import Reaction, ReactionSystem
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Install chempy to run this research sketch: pip install chempy") from exc


def build_system() -> ReactionSystem:
    # Placeholder first-order rates (1/hour). Not fitted to data.
    reactions = [
        Reaction({"mRNA": 1}, {"mRNA": 1, "antigen": 1}, param=0.12, name="translation"),
        Reaction({"mRNA": 1}, {}, param=0.05, name="mrna_decay"),
        Reaction({"antigen": 1}, {"signal": 1}, param=0.03, name="presentation"),
        Reaction({"antigen": 1}, {}, param=0.02, name="antigen_clearance"),
        Reaction({"signal": 1}, {}, param=0.01, name="signal_decay"),
    ]
    return ReactionSystem(reactions, "mRNA antigen signal")


def main() -> None:
    rsys = build_system()
    print("ScienceOpen DOI:", DOI)
    print("Unreviewed preprint. Toy ReactionSystem only.")
    print(rsys)
    try:
        import numpy as np
        from chempy.kinetics.ode import get_odesys

        odesys, _extra = get_odesys(rsys)
        tout = np.linspace(0.0, 48.0, 97)
        result, _info = odesys.integrate(tout, {"mRNA": 1.0, "antigen": 0.0, "signal": 0.0})
        print("t_hours\tmRNA\tantigen\tsignal")
        for t, row in zip(result.x, result.yout):
            print(f"{float(t):.2f}\t{row[0]:.6f}\t{row[1]:.6f}\t{row[2]:.6f}")
    except Exception as exc:
        print("ODE integrate skipped:", exc)


if __name__ == "__main__":
    main()
`, title, doi, url, passage, doi)
}
