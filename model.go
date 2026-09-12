package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const (
	modelVersion      = 6
	defaultModelPath  = "stella_model.json"
	maxPromptTokens   = 512
	maxReplyTokens    = 64
	embeddingDim      = 32
	minRetrieveScore  = 0.44
	defaultConfidence = 0.58
	medicalDisclaimer = "This is research information from Stella V, not diagnosis or treatment advice."
)

var (
	tokenRE       = regexp.MustCompile(`[A-Za-z0-9_']+|[^\s]`)
	specialTokens = []string{"<pad>", "<unk>", "<bos>", "<eos>", "user:", "assistant:"}
)

const (
	padID = iota
	unkID
	bosID
	eosID
	userID
	assistantID
)

type Sample struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

type StellaModel struct {
	Version    int              `json:"version"`
	CreatedAt  int64            `json:"created_at"`
	UpdatedAt  int64            `json:"updated_at"`
	Vocab      []string         `json:"vocab"`
	Samples    []Sample         `json:"samples"`
	Chunks     []KnowledgeChunk `json:"chunks"`
	ReasonSize string           `json:"reason_size,omitempty"`

	tokenToID       map[string]int
	transitions     map[int]map[int]int
	answerFrequency map[int]int
}

type Prediction struct {
	Text       string   `json:"text"`
	Confidence float64  `json:"confidence"`
	Tokens     []string `json:"tokens"`
	Source     string   `json:"source"`
}

type ModelStats struct {
	Samples     int    `json:"samples"`
	Vocab       int    `json:"vocab"`
	Transitions int    `json:"transitions"`
	Chunks      int    `json:"chunks"`
	Format      string `json:"format"`
	UpdatedAt   int64  `json:"updated_at"`
}

type ModelTensor struct {
	Name  string
	DType string
	Shape []uint64
	F32   []float32
}

type rankedSample struct {
	Sample Sample
	Score  float64
}

func NewStellaModel() *StellaModel {
	now := time.Now().Unix()
	m := &StellaModel{
		Version:   modelVersion,
		CreatedAt: now,
		UpdatedAt: now,
		Vocab:     append([]string(nil), specialTokens...),
	}
	m.Rebuild()
	return m
}

func LoadModel(path string) (*StellaModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m StellaModel
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if m.Version == 0 {
		m.Version = modelVersion
	}
	if m.CreatedAt == 0 {
		m.CreatedAt = time.Now().Unix()
	}
	if len(m.Vocab) == 0 {
		m.Vocab = append([]string(nil), specialTokens...)
	}
	m.normalizeSpecials()
	m.Rebuild()
	return &m, nil
}

func LoadOrBootstrap(path string) (*StellaModel, error) {
	if strings.TrimSpace(path) == "" {
		path = defaultModelPath
	}
	if m, err := LoadModel(path); err == nil {
		return m, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	m := NewStellaModel()
	for _, s := range BootstrapSamples() {
		m.AddSample(s.Question, s.Answer)
	}
	return m, nil
}

func (m *StellaModel) Save(path string) error {
	if strings.TrimSpace(path) == "" {
		path = defaultModelPath
	}
	m.Version = modelVersion
	m.UpdatedAt = time.Now().Unix()
	if err := os.MkdirAll(filepath.Dir(cleanPathForCreate(path)), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func cleanPathForCreate(path string) string {
	if dir := filepath.Dir(path); dir == "." || dir == "" {
		return filepath.Join(".", filepath.Base(path))
	}
	return path
}

func (m *StellaModel) AddSample(question, answer string) {
	question = strings.TrimSpace(question)
	answer = strings.TrimSpace(answer)
	if question == "" || answer == "" {
		return
	}
	m.Samples = append(m.Samples, Sample{Question: question, Answer: answer})
	m.UpdatedAt = time.Now().Unix()
	m.Rebuild()
}

func (m *StellaModel) AddChunk(id, title, text, rawURL, source string) bool {
	title = strings.TrimSpace(title)
	text = strings.TrimSpace(text)
	if text == "" {
		return false
	}
	id = strings.TrimSpace(id)
	if id == "" {
		id = fmt.Sprintf("chunk-%d", len(m.Chunks)+1)
	}
	for i, existing := range m.Chunks {
		if existing.ID == id || (existing.DOI != "" && existing.DOI == id) {
			m.Chunks[i] = KnowledgeChunk{
				ID:     id,
				Title:  title,
				DOI:    id,
				URL:    strings.TrimSpace(rawURL),
				Text:   text,
				Source: source,
			}
			m.UpdatedAt = time.Now().Unix()
			return false
		}
	}
	m.Chunks = append(m.Chunks, KnowledgeChunk{
		ID:     id,
		Title:  title,
		DOI:    id,
		URL:    strings.TrimSpace(rawURL),
		Text:   text,
		Source: source,
	})
	m.UpdatedAt = time.Now().Unix()
	return true
}

func (m *StellaModel) Rebuild() {
	m.normalizeSpecials()
	m.tokenToID = make(map[string]int, len(m.Vocab))
	for i, tok := range m.Vocab {
		normalized := normalizeToken(tok)
		if normalized == "" {
			normalized = fmt.Sprintf("<empty-%d>", i)
		}
		m.Vocab[i] = normalized
		if _, exists := m.tokenToID[normalized]; !exists {
			m.tokenToID[normalized] = i
		}
	}
	m.transitions = map[int]map[int]int{}
	m.answerFrequency = map[int]int{}
	for _, sample := range m.Samples {
		qIDs := m.encode(tokenize(sample.Question), true)
		aTokens := tokenize(sample.Answer)
		aIDs := m.encode(aTokens, true)
		if len(aIDs) == 0 {
			continue
		}
		promptSeed := assistantID
		if len(qIDs) > 0 {
			promptSeed = qIDs[len(qIDs)-1]
		}
		m.addTransition(promptSeed, aIDs[0])
		prev := bosID
		for _, id := range aIDs {
			m.addTransition(prev, id)
			m.answerFrequency[id]++
			prev = id
		}
		m.addTransition(prev, eosID)
	}
}

func (m *StellaModel) normalizeSpecials() {
	if len(m.Vocab) < len(specialTokens) {
		merged := append([]string(nil), specialTokens...)
		for _, tok := range m.Vocab {
			if !containsString(merged, tok) {
				merged = append(merged, tok)
			}
		}
		m.Vocab = merged
		return
	}
	for i, tok := range specialTokens {
		m.Vocab[i] = tok
	}
}

func (m *StellaModel) addTransition(from, to int) {
	if m.transitions[from] == nil {
		m.transitions[from] = map[int]int{}
	}
	m.transitions[from][to]++
}

func (m *StellaModel) ensureToken(token string) int {
	token = normalizeToken(token)
	if token == "" {
		return unkID
	}
	if id, ok := m.tokenToID[token]; ok {
		return id
	}
	id := len(m.Vocab)
	m.Vocab = append(m.Vocab, token)
	m.tokenToID[token] = id
	return id
}

func (m *StellaModel) encode(tokens []string, allowAdd bool) []int {
	if len(tokens) > maxPromptTokens {
		tokens = tokens[:maxPromptTokens]
	}
	ids := make([]int, 0, len(tokens))
	for _, tok := range tokens {
		tok = normalizeToken(tok)
		if tok == "" {
			continue
		}
		if allowAdd {
			ids = append(ids, m.ensureToken(tok))
			continue
		}
		if id, ok := m.tokenToID[tok]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, unkID)
		}
	}
	return ids
}

func (m *StellaModel) Predict(question string) Prediction {
	question = strings.TrimSpace(question)
	if question == "" {
		return Prediction{Text: "Ask a medical-research question, or teach me a new answer.", Confidence: 0.2, Source: "empty"}
	}
	if hits := NewTensorEngine().Search(question, m.Chunks, nil, 1); len(hits) > 0 && hits[0].Score >= 0.12 {
		hit := hits[0]
		text := hit.Chunk.Text
		if hit.Chunk.Title != "" {
			doi := hit.Chunk.DOI
			if doi == "" {
				doi = "DOI n/a"
			}
			text = fmt.Sprintf("From ScienceOpen preprint %s (%s): %s %s", hit.Chunk.Title, doi, singleLine(hit.Chunk.Text, 700), medicalDisclaimer)
		}
		conf := math.Min(0.93, 0.58+math.Min(hit.Score, 1.0)*0.28)
		return Prediction{Text: text, Confidence: conf, Tokens: tokenize(text), Source: "tensor-scienceopen"}
	}
	if text, confidence := m.retrieve(question); text != "" {
		return Prediction{Text: text, Confidence: confidence, Tokens: tokenize(text), Source: "memory"}
	}
	if text := semanticFallback(question); text != "" {
		return Prediction{Text: text, Confidence: 0.62, Tokens: tokenize(text), Source: "bootstrap"}
	}
	if text := m.generate(question); text != "" {
		return Prediction{Text: text, Confidence: defaultConfidence, Tokens: tokenize(text), Source: "learned-transition"}
	}
	return Prediction{
		Text:       "I do not have a matching ScienceOpen preprint yet. Index a query or teach me the answer. " + medicalDisclaimer,
		Confidence: 0.18,
		Tokens:     nil,
		Source:     "fallback",
	}
}

func (m *StellaModel) BuildConquerorPrompt(question string, local Prediction) string {
	related := m.RelevantSamples(question, 3)
	hits := NewTensorEngine().Search(question, m.Chunks, nil, 4)
	var b strings.Builder
	b.WriteString(reasonerSystemPrompt())
	b.WriteString("\nAnswer as Stella V in clear research prose.\n")
	b.WriteString("Use the retrieved ScienceOpen passages when they are relevant. Keep the answer concise.\n")
	b.WriteString(medicalDisclaimer + "\n\n")
	if len(hits) > 0 {
		b.WriteString("Tensor-ranked ScienceOpen passages:\n")
		b.WriteString(formatRankedPassages(hits, 4))
		b.WriteString("\n\n")
	}
	if len(related) > 0 {
		b.WriteString("Learned memory:\n")
		for _, sample := range related {
			b.WriteString("- User: ")
			b.WriteString(singleLine(sample.Question, 220))
			b.WriteString("\n  Stella: ")
			b.WriteString(singleLine(sample.Answer, 320))
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}
	if strings.TrimSpace(local.Text) != "" && local.Source != "fallback" && local.Source != "empty" {
		b.WriteString("Local Stella draft: ")
		b.WriteString(singleLine(local.Text, 420))
		b.WriteString("\n\n")
	}
	b.WriteString("Latest user message: ")
	b.WriteString(strings.TrimSpace(question))
	b.WriteString("\nStella:")
	return b.String()
}

func (m *StellaModel) RelevantSamples(question string, limit int) []Sample {
	if limit <= 0 || len(m.Samples) == 0 {
		return nil
	}
	qTokens := tokenize(question)
	if len(qTokens) == 0 {
		return nil
	}
	qSet := tokenSet(qTokens)
	qBigrams := bigramSet(qTokens)
	qNorm := strings.Join(qTokens, " ")
	ranked := make([]rankedSample, 0, len(m.Samples))
	for _, sample := range m.Samples {
		score := sampleRelevance(sample, qSet, qBigrams, qNorm)
		if score > 0 {
			ranked = append(ranked, rankedSample{Sample: sample, Score: score})
		}
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].Score == ranked[j].Score {
			return ranked[i].Sample.Question < ranked[j].Sample.Question
		}
		return ranked[i].Score > ranked[j].Score
	})
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}
	out := make([]Sample, len(ranked))
	for i, item := range ranked {
		out[i] = item.Sample
	}
	return out
}

func (m *StellaModel) retrieve(question string) (string, float64) {
	qTokens := tokenize(question)
	if len(qTokens) == 0 {
		return "", 0
	}
	qSet := tokenSet(qTokens)
	qBigrams := bigramSet(qTokens)
	qNorm := strings.Join(qTokens, " ")

	best := Sample{}
	bestScore := 0.0
	for _, sample := range m.Samples {
		score := sampleRelevance(sample, qSet, qBigrams, qNorm)
		if score > bestScore {
			bestScore = score
			best = sample
		}
	}
	if bestScore < minRetrieveScore {
		return "", 0
	}
	confidence := math.Min(0.99, 0.58+math.Min(bestScore, 1.0)*0.37)
	return best.Answer, confidence
}

func sampleRelevance(sample Sample, qSet, qBigrams map[string]struct{}, qNorm string) float64 {
	sTokens := tokenize(sample.Question)
	if len(sTokens) == 0 {
		return 0
	}
	sSet := tokenSet(sTokens)
	overlap := intersectionSize(qSet, sSet)
	if overlap == 0 {
		return 0
	}
	union := len(qSet) + len(sSet) - overlap
	tokenScore := float64(overlap) / float64(maxInt(1, union))
	sBigrams := bigramSet(sTokens)
	bigramOverlap := intersectionSize(qBigrams, sBigrams)
	bigramUnion := len(qBigrams) + len(sBigrams) - bigramOverlap
	bigramScore := 0.0
	if bigramUnion > 0 {
		bigramScore = float64(bigramOverlap) / float64(bigramUnion)
	}
	sNorm := strings.Join(sTokens, " ")
	exactBonus := 0.0
	if qNorm == sNorm {
		exactBonus = 0.45
	}
	containsBonus := 0.0
	if strings.Contains(qNorm, sNorm) || strings.Contains(sNorm, qNorm) {
		containsBonus = 0.18
	}
	return tokenScore + 0.35*bigramScore + exactBonus + containsBonus
}

func (m *StellaModel) generate(question string) string {
	ids := m.encode(tokenize(question), false)
	seed := assistantID
	if len(ids) > 0 {
		seed = ids[len(ids)-1]
	}
	current := seed
	if len(m.transitions[current]) == 0 {
		current = bosID
	}
	var out []int
	recent := map[int]int{}
	for step := 0; step < maxReplyTokens; step++ {
		next := m.pickNext(current, recent)
		if next < 0 || next == eosID {
			break
		}
		if next >= len(m.Vocab) {
			break
		}
		out = append(out, next)
		recent[next]++
		current = next
	}
	if len(out) == 0 {
		return ""
	}
	tokens := make([]string, 0, len(out))
	for _, id := range out {
		if id >= len(specialTokens) && id < len(m.Vocab) {
			tokens = append(tokens, m.Vocab[id])
		}
	}
	return detokenize(tokens)
}

func (m *StellaModel) pickNext(current int, recent map[int]int) int {
	candidates := m.transitions[current]
	if len(candidates) == 0 {
		candidates = m.transitions[bosID]
	}
	bestID := -1
	bestScore := math.Inf(-1)
	for id, count := range candidates {
		score := float64(count)
		if id < len(specialTokens) && id != eosID {
			score -= 8
		}
		if recent[id] > 0 {
			score -= float64(recent[id]) * 1.5
		}
		score += float64(fnv32(fmt.Sprintf("%d:%d", current, id))%997) / 100000.0
		if score > bestScore {
			bestScore = score
			bestID = id
		}
	}
	return bestID
}

func (m *StellaModel) Stats() ModelStats {
	edges := 0
	for _, next := range m.transitions {
		edges += len(next)
	}
	return ModelStats{
		Samples:     len(m.Samples),
		Vocab:       len(m.Vocab),
		Transitions: edges,
		Chunks:      len(m.Chunks),
		Format:      "stella-v-medical/scienceopen+tensor",
		UpdatedAt:   m.UpdatedAt,
	}
}

func (m *StellaModel) Tensors() []ModelTensor {
	emb := m.embeddingTensor()
	output := make([]float32, embeddingDim*len(m.Vocab))
	for token := 0; token < len(m.Vocab); token++ {
		for dim := 0; dim < embeddingDim; dim++ {
			output[dim*len(m.Vocab)+token] = emb[token*embeddingDim+dim]
		}
	}
	bias := make([]float32, len(m.Vocab))
	maxFreq := 1
	for _, count := range m.answerFrequency {
		if count > maxFreq {
			maxFreq = count
		}
	}
	for id, count := range m.answerFrequency {
		if id >= 0 && id < len(bias) {
			bias[id] = float32(count) / float32(maxFreq)
		}
	}
	transition := make([]float32, len(m.Vocab)*embeddingDim)
	for from, nexts := range m.transitions {
		if from < 0 || from >= len(m.Vocab) {
			continue
		}
		total := 0
		for _, count := range nexts {
			total += count
		}
		if total == 0 {
			continue
		}
		for to, count := range nexts {
			if to < 0 || to >= len(m.Vocab) {
				continue
			}
			weight := float32(count) / float32(total)
			for dim := 0; dim < embeddingDim; dim++ {
				transition[from*embeddingDim+dim] += weight * emb[to*embeddingDim+dim]
			}
		}
	}
	tensors := []ModelTensor{
		{Name: "stella.embedding.weight", DType: "F32", Shape: []uint64{uint64(len(m.Vocab)), embeddingDim}, F32: emb},
		{Name: "stella.output.weight", DType: "F32", Shape: []uint64{embeddingDim, uint64(len(m.Vocab))}, F32: output},
		{Name: "stella.output.bias", DType: "F32", Shape: []uint64{uint64(len(m.Vocab))}, F32: bias},
		{Name: "stella.transition.summary", DType: "F32", Shape: []uint64{uint64(len(m.Vocab)), embeddingDim}, F32: transition},
	}
	return append(tensors, NewTensorEngine().RetrievalTensors()...)
}

func (m *StellaModel) embeddingTensor() []float32 {
	out := make([]float32, len(m.Vocab)*embeddingDim)
	for row, token := range m.Vocab {
		seed := fnv32(token)
		for dim := 0; dim < embeddingDim; dim++ {
			seed = seed*1664525 + 1013904223
			centered := (float32((seed>>8)&0xffff) / 32767.5) - 1
			out[row*embeddingDim+dim] = centered
		}
	}
	return out
}

func BootstrapSamples() []Sample {
	return []Sample{
		{Question: "what is stella v", Answer: "Stella V is a local medical-research assistant that retrieves ScienceOpen preprints with a tensor engine and reasons over them with a selectable 1.5B or 3B model."},
		{Question: "what can you export", Answer: "I export PAccel first, safetensors second, and ONNX third, including retrieval projection tensors."},
		{Question: "why paccel", Answer: "PAccel is the preferred tensor container because it keeps quantized weights and runtime metadata compact for inference."},
		{Question: "how do you learn", Answer: "I index ScienceOpen preprints, store teaching pairs, and rebuild linear plus attention retrieval tensors."},
		{Question: "what is scienceopen", Answer: "ScienceOpen is the research source Stella V uses. Preprints there are registered under Crossref prefix 10.14293."},
		{Question: "what is a preprint", Answer: "A preprint is a manuscript shared before journal peer review and should be read as unreviewed evidence."},
		{Question: "what is a randomized trial", Answer: "A randomized trial assigns participants by chance so confounders are balanced in expectation when testing an intervention."},
		{Question: "what is confounding", Answer: "Confounding is distortion of an effect estimate by a third factor associated with both exposure and outcome."},
		{Question: "is this medical advice", Answer: medicalDisclaimer},
		{Question: "what is a tensor", Answer: "A tensor is a typed multidimensional array used to store model weights, activations, or retrieval embeddings."},
		{Question: "what is the reasoning model", Answer: "Stella V can select a 1.5B or 3B local reasoning model. That model uses the tensor engine as a fast backend to pick ScienceOpen passages."},
		{Question: "how should i teach you", Answer: "Give me a medical-research question and the answer you want me to remember, or index ScienceOpen queries."},
	}
}

func tokenize(text string) []string {
	raw := tokenRE.FindAllString(strings.ToLower(text), -1)
	out := make([]string, 0, len(raw))
	for _, tok := range raw {
		tok = normalizeToken(tok)
		if tok != "" {
			out = append(out, tok)
		}
		if len(out) >= maxPromptTokens {
			break
		}
	}
	return out
}

func normalizeToken(token string) string {
	return strings.ToLower(strings.TrimSpace(token))
}

func detokenize(tokens []string) string {
	noSpaceBefore := ".,;:)]}>?!"
	noSpaceAfter := "([{<#"
	var b strings.Builder
	prev := ""
	for _, tok := range tokens {
		if tok == "" {
			continue
		}
		if b.Len() == 0 || strings.Contains(noSpaceBefore, tok) || strings.Contains(noSpaceAfter, prev) {
			b.WriteString(tok)
		} else {
			b.WriteByte(' ')
			b.WriteString(tok)
		}
		prev = tok
	}
	return b.String()
}

func singleLine(text string, limit int) string {
	text = strings.Join(strings.Fields(text), " ")
	if limit > 0 && len(text) > limit {
		if limit <= 3 {
			return text[:limit]
		}
		return text[:limit-3] + "..."
	}
	return text
}

func semanticFallback(question string) string {
	q := strings.ToLower(question)
	switch {
	case strings.Contains(q, "scienceopen"):
		return "ScienceOpen hosts and indexes research, including preprints with Crossref prefix 10.14293. Stella V searches those preprints and ranks the abstracts with a tensor engine."
	case strings.Contains(q, "preprint"):
		return "A preprint is a complete manuscript shared before journal peer review. It speeds communication but is unreviewed evidence."
	case strings.Contains(q, "1.5") || strings.Contains(q, "3b") || strings.Contains(q, "reason"):
		return "Stella V uses a selectable 1.5B or 3B reasoning model. That model calls the tensor engine to fetch and select ScienceOpen passages before answering."
	case strings.Contains(q, "random") && strings.Contains(q, "trial"):
		return "A randomized trial assigns participants by chance so measured and unmeasured confounders are balanced in expectation."
	case strings.Contains(q, "confound"):
		return "Confounding occurs when a third factor associated with both exposure and outcome distorts the estimated effect."
	case strings.Contains(q, "paccel"):
		return "PAccel is Stella's primary export path: learned tensors are written through safetensors and compressed into a PAccel package."
	case strings.Contains(q, "safetensor"):
		return "Safetensors export writes Stella's embedding, output, bias, transition, and retrieval projection tensors."
	case strings.Contains(q, "onnx"):
		return "ONNX export writes a compact feature-to-logits graph that carries Stella's learned output tensors."
	case strings.Contains(q, "medical advice") || strings.Contains(q, "diagnos") || strings.Contains(q, "treat me"):
		return medicalDisclaimer
	case strings.Contains(q, "learn") || strings.Contains(q, "train"):
		return "I learn by indexing ScienceOpen preprints, storing teaching pairs, and rebuilding linear plus attention retrieval tensors."
	default:
		return ""
	}
}

func tokenSet(tokens []string) map[string]struct{} {
	out := make(map[string]struct{}, len(tokens))
	for _, tok := range tokens {
		out[tok] = struct{}{}
	}
	return out
}

func bigramSet(tokens []string) map[string]struct{} {
	out := map[string]struct{}{}
	for i := 0; i+1 < len(tokens); i++ {
		out[tokens[i]+"\x00"+tokens[i+1]] = struct{}{}
	}
	return out
}

func intersectionSize(a, b map[string]struct{}) int {
	if len(a) > len(b) {
		a, b = b, a
	}
	n := 0
	for k := range a {
		if _, ok := b[k]; ok {
			n++
		}
	}
	return n
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func fnv32(text string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(text))
	return h.Sum32()
}

func readJSONLSamples(path string) ([]Sample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []Sample
	for lineNo, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var sample Sample
		if err := json.Unmarshal([]byte(line), &sample); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo+1, err)
		}
		if sample.Question != "" && sample.Answer != "" {
			out = append(out, sample)
		}
	}
	return out, nil
}

func sortedTransitionKeys(m map[int]map[int]int) []int {
	keys := make([]int, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	return keys
}
