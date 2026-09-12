package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/powerengine/paccel/v7graph"
)

// V7Engine is the optimal inference route: it runs the extracted V7
// transformer_decoder graph directly against the pre-transposed, block-quantized
// weights of a direct-only .paccel package. Generation is greedy with full
// recompute (no KV cache) for clarity; the point is to exercise the real graph.
type V7Engine struct {
	model     *v7graph.Model
	idToTok   []string
	tokToID   map[string]int
	maxTokens int
	eosID     int
}

func (e *V7Engine) Name() string {
	r := e.model.R
	return fmt.Sprintf("v7-direct [%s, %d layers, hidden %d, heads %d/%d, vocab %d]",
		r.ModelType, r.NumHiddenLayers, r.HiddenSize, r.NumAttentionHeads, r.NumKeyValueHeads, r.VocabSize)
}

// LoadV7Engine binds the V7 graph from a recipe + package and loads the tokenizer
// vocabulary from modelDir (for human-readable tokens).
func LoadV7Engine(recipePath, paccelPath, modelDir string, maxTokens int) (*V7Engine, error) {
	m, err := v7graph.Load(recipePath, paccelPath)
	if err != nil {
		return nil, err
	}
	e := &V7Engine{model: m, maxTokens: maxTokens, eosID: -1}
	e.loadVocab(modelDir, m.R.VocabSize)
	return e, nil
}

func (e *V7Engine) loadVocab(modelDir string, vocab int) {
	data, err := os.ReadFile(filepath.Join(modelDir, "vocab.json"))
	if err != nil {
		return
	}
	var m map[string]int
	if json.Unmarshal(data, &m) != nil {
		return
	}
	e.tokToID = m
	e.idToTok = make([]string, vocab)
	for tok, id := range m {
		if id >= 0 && id < vocab {
			e.idToTok[id] = tok
		}
	}
}

func (e *V7Engine) tokenize(prompt string) []int {
	vocab := e.model.R.VocabSize
	words := strings.Fields(strings.ToLower(prompt))
	ids := make([]int, 0, len(words)+1)
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
			id = int(fnv32(w)) % vocab
		}
		ids = append(ids, id%vocab)
	}
	if len(ids) == 0 {
		ids = append(ids, 0)
	}
	return ids
}

func (e *V7Engine) detok(id int) string {
	if e.idToTok != nil && id < len(e.idToTok) && e.idToTok[id] != "" {
		return strings.ReplaceAll(e.idToTok[id], "Ġ", " ")
	}
	return fmt.Sprintf(" <%d>", id)
}

// Reply runs greedy autoregressive decoding through the V7 graph.
func (e *V7Engine) Reply(prompt string) (string, error) {
	tokens := e.tokenize(prompt)
	var out []string
	for step := 0; step < e.maxTokens; step++ {
		next := e.model.NextToken(tokens)
		if next == e.eosID {
			break
		}
		out = append(out, e.detok(next))
		tokens = append(tokens, next)
		if len(tokens) > 256 { // bound the recompute window
			tokens = tokens[len(tokens)-256:]
		}
	}
	reply := strings.TrimSpace(strings.Join(out, ""))
	if reply == "" {
		reply = "(v7 graph produced no tokens)"
	}
	return reply, nil
}

// findV7Recipe returns the path to a model.v7.json sitting next to a package or
// in modelDir, or "" if none is present.
func findV7Recipe(paccelPath, modelDir string) string {
	for _, dir := range []string{filepath.Dir(paccelPath), modelDir} {
		if dir == "" {
			continue
		}
		p := filepath.Join(dir, "model.v7.json")
		if info, err := os.Stat(p); err == nil && !info.IsDir() && info.Size() > 0 {
			return p
		}
	}
	return ""
}
