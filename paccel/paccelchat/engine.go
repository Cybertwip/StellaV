package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/powerengine/paccel/libpaccel"
)

// Engine is the chat backend seam. EchoEngine needs no model; PaccelEngine binds
// to a loaded .paccel container and generates tokens from the model's real,
// dequantized embedding matrix. A full autoregressive transformer executor would
// implement this same interface — it is the single integration point left open
// by the bare-minimum port (the heavy graph executor is deliberately out of
// scope; libpaccel supplies it dequantized weights on demand).
type Engine interface {
	Name() string
	Reply(prompt string) (string, error)
}

// EchoEngine is the always-available fallback when no neural weights are bound.
type EchoEngine struct{}

func (EchoEngine) Name() string { return "echo (no model bound)" }
func (EchoEngine) Reply(prompt string) (string, error) {
	p := strings.TrimSpace(prompt)
	if p == "" {
		return "Say something and I'll echo it back. Load a .paccel with --model for embedding-grounded replies.", nil
	}
	return "you said: " + p, nil
}

// PaccelEngine generates from a loaded model's token-embedding matrix using a
// nearest-neighbour walk in embedding space. This is a genuine (if minimal)
// neural operation over weights decoded by libpaccel from the .paccel container;
// it is NOT a full language model and does not produce fluent prose. It exists
// to prove the codec round-trips real model weights into usable float tensors.
type PaccelEngine struct {
	modelName string
	embed     []float32 // [vocab][dim], row-major
	vocab     int
	dim       int
	idToTok   []string
	tokToID   map[string]int
	maxTokens int
}

func (e *PaccelEngine) Name() string {
	return fmt.Sprintf("paccel-embedding-sampler [%s, vocab=%d dim=%d]", e.modelName, e.vocab, e.dim)
}

// LoadPaccelEngine reads a .paccel package, locates and dequantizes a rank-2
// token-embedding tensor via libpaccel, and (optionally) loads a vocab.json from
// modelDir for human-readable tokens.
func LoadPaccelEngine(paccelPath, modelDir string, maxTokens int) (*PaccelEngine, error) {
	pkg, err := libpaccel.ReadPackage(paccelPath)
	if err != nil {
		return nil, err
	}
	rec, ok := findEmbeddingRecord(pkg)
	if !ok {
		return nil, fmt.Errorf("no rank-2 embedding tensor found in %s", filepath.Base(paccelPath))
	}
	shape := rec.Shape
	if rec.Compressed() {
		shape = rec.Tensor.Shape
	}
	vocab, dim := int(shape[0]), int(shape[1])

	var embed []float32
	if rec.Compressed() {
		embed = libpaccel.TurboDequantize(rec.Tensor)
	} else {
		raw := rec.RawPayload
		if rec.Encoding == libpaccel.EncodingRawRLE {
			if raw, err = libpaccel.RLEDecode(raw); err != nil {
				return nil, err
			}
		}
		embed = libpaccel.DecodeRawTensor(raw, rec.SourceDataType)
	}
	if len(embed) < vocab*dim {
		return nil, fmt.Errorf("embedding payload too small: have %d need %d", len(embed), vocab*dim)
	}

	eng := &PaccelEngine{
		modelName: filepath.Base(modelDir),
		embed:     embed,
		vocab:     vocab,
		dim:       dim,
		maxTokens: maxTokens,
	}
	eng.loadVocab(modelDir)
	return eng, nil
}

// findEmbeddingRecord prefers tensors whose name marks a token embedding, then
// falls back to the largest rank-2 tensor (which in transformer checkpoints is
// almost always the embedding / tied LM head).
func findEmbeddingRecord(pkg *libpaccel.Package) (libpaccel.Record, bool) {
	var best libpaccel.Record
	var bestRows uint64
	found := false
	for _, rec := range pkg.Records {
		shape := rec.Shape
		if rec.Compressed() {
			shape = rec.Tensor.Shape
		}
		if len(shape) != 2 || shape[0] == 0 || shape[1] == 0 {
			continue
		}
		name := strings.ToLower(rec.Name)
		preferred := strings.Contains(name, "embed") || strings.Contains(name, "wte") ||
			strings.Contains(name, "tok_embeddings") || strings.Contains(name, "word_embeddings")
		if preferred {
			return rec, true
		}
		if shape[0] > bestRows {
			bestRows = shape[0]
			best = rec
			found = true
		}
	}
	return best, found
}

func (e *PaccelEngine) loadVocab(modelDir string) {
	data, err := os.ReadFile(filepath.Join(modelDir, "vocab.json"))
	if err != nil {
		return
	}
	var m map[string]int
	if json.Unmarshal(data, &m) != nil {
		return
	}
	e.tokToID = m
	e.idToTok = make([]string, e.vocab)
	for tok, id := range m {
		if id >= 0 && id < e.vocab {
			e.idToTok[id] = tok
		}
	}
}

// tokenize maps a prompt to embedding-row ids. With a vocab it greedily matches
// whole words (with the GPT-2 leading-space marker) and falls back to a stable
// hash; without a vocab it hashes words directly into the row range.
func (e *PaccelEngine) tokenize(prompt string) []int {
	words := strings.Fields(strings.ToLower(prompt))
	ids := make([]int, 0, len(words))
	for _, w := range words {
		id := -1
		if e.tokToID != nil {
			if v, ok := e.tokToID["Ġ"+w]; ok {
				id = v
			} else if v, ok := e.tokToID[w]; ok {
				id = v
			}
		}
		if id < 0 {
			id = int(fnv32(w)) % e.vocab
		}
		ids = append(ids, id%e.vocab)
	}
	return ids
}

func (e *PaccelEngine) row(id int) []float32 {
	return e.embed[id*e.dim : id*e.dim+e.dim]
}

// Reply runs the nearest-neighbour walk: build a context vector from the prompt
// embeddings, then repeatedly pick the most similar vocabulary row, blending it
// into the context. Candidate rows are capped for bounded latency.
func (e *PaccelEngine) Reply(prompt string) (string, error) {
	ids := e.tokenize(prompt)
	context := make([]float32, e.dim)
	if len(ids) == 0 {
		// Seed from a deterministic row so empty prompts still respond.
		copy(context, e.row(int(fnv32(prompt))%e.vocab))
	} else {
		for _, id := range ids {
			r := e.row(id)
			for i := range context {
				context[i] += r[i]
			}
		}
		for i := range context {
			context[i] /= float32(len(ids))
		}
	}

	candidateCap := e.vocab
	if candidateCap > 20000 {
		candidateCap = 20000
	}
	recent := map[int]bool{}
	var out []string
	for step := 0; step < e.maxTokens; step++ {
		bestID, bestSim := -1, float32(math.Inf(-1))
		for id := 0; id < candidateCap; id++ {
			if recent[id] {
				continue
			}
			if sim := cosine(context, e.row(id)); sim > bestSim {
				bestSim, bestID = sim, id
			}
		}
		if bestID < 0 {
			break
		}
		recent[bestID] = true
		out = append(out, e.detok(bestID))
		// Blend the chosen row into context (autoregressive-style feedback).
		r := e.row(bestID)
		for i := range context {
			context[i] = 0.7*context[i] + 0.3*r[i]
		}
	}
	reply := strings.TrimSpace(strings.Join(out, ""))
	if reply == "" {
		reply = "(model produced no tokens)"
	}
	return reply, nil
}

func (e *PaccelEngine) detok(id int) string {
	if e.idToTok != nil && id < len(e.idToTok) && e.idToTok[id] != "" {
		// GPT-2 byte-level: Ġ marks a leading space.
		return strings.ReplaceAll(e.idToTok[id], "Ġ", " ")
	}
	return fmt.Sprintf(" <%d>", id)
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

func fnv32(s string) uint32 {
	const prime = 16777619
	h := uint32(2166136261)
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= prime
	}
	if h == 0 {
		h = 1
	}
	return h
}

// summarizePackage returns a short human description of a loaded package.
func summarizePackage(pkg *libpaccel.Package) string {
	comp, raw := 0, 0
	var packed uint64
	bitHist := map[uint32]int{}
	for _, rec := range pkg.Records {
		if rec.Compressed() {
			comp++
			packed += uint64(len(rec.Tensor.Packed))
			bitHist[rec.Tensor.BitWidth]++
		} else {
			raw++
			packed += uint64(len(rec.RawPayload))
		}
	}
	bits := make([]int, 0, len(bitHist))
	for b := range bitHist {
		bits = append(bits, int(b))
	}
	sort.Ints(bits)
	parts := make([]string, 0, len(bits))
	for _, b := range bits {
		parts = append(parts, fmt.Sprintf("%db×%d", b, bitHist[uint32(b)]))
	}
	return fmt.Sprintf("v%d, %d tensors (%d quantized, %d raw), payload %s, widths {%s}",
		pkg.Version, len(pkg.Records), comp, raw, humanBytes(packed), strings.Join(parts, " "))
}
