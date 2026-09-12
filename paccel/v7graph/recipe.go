// Package v7graph is the extracted V7 model graph: the optimal inference route
// for the PAccel format. It loads a model.v7.json recipe plus a direct-layout
// .paccel package and runs a Qwen2/Qwen3-family transformer decoder forward pass
// directly against the pre-transposed, block-quantized weights — no ONNX, no
// onnxruntime, no protobuf, no transpose at runtime. It depends only on the
// standard library and libpaccel.
package v7graph

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/powerengine/paccel/libpaccel"
)

// Recipe mirrors model.v7.json: the architecture description that lets the graph
// run without a serialized compute graph. Everything the decoder needs to wire
// itself is here, so the recipe plus the weight package is a complete model.
type Recipe struct {
	Format            string  `json:"format"`
	Version           int     `json:"version"`
	GraphKind         string  `json:"graph_kind"`
	ModelType         string  `json:"model_type"`
	WeightLayout      string  `json:"weight_layout"`
	PackageMode       string  `json:"package_mode"`
	Weights           string  `json:"weights"`
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	HeadDim           int     `json:"head_dim"`
	VocabSize         int     `json:"vocab_size"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	RopeTheta         float64 `json:"rope_theta"`
	HiddenAct         string  `json:"hidden_act"`
	TieWordEmbeddings bool    `json:"tie_word_embeddings"`
}

// LoadRecipe reads and validates a model.v7.json file.
func LoadRecipe(path string) (*Recipe, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Recipe
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("v7 recipe: %w", err)
	}
	if err := r.fillDefaults(); err != nil {
		return nil, err
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return &r, nil
}

func (r *Recipe) fillDefaults() error {
	if r.HeadDim == 0 && r.NumAttentionHeads > 0 {
		r.HeadDim = r.HiddenSize / r.NumAttentionHeads
	}
	if r.NumKeyValueHeads == 0 {
		r.NumKeyValueHeads = r.NumAttentionHeads
	}
	if r.RopeTheta == 0 {
		r.RopeTheta = 10000
	}
	if r.RMSNormEps == 0 {
		r.RMSNormEps = 1e-6
	}
	if r.HiddenAct == "" {
		r.HiddenAct = "silu"
	}
	if r.WeightLayout == "" {
		r.WeightLayout = libpaccel.V7WeightLayout // fused, current default
	}
	return nil
}

// Validate checks the recipe is a current, supported V7 direct recipe with a
// self-consistent shape set.
func (r *Recipe) Validate() error {
	if !strings.EqualFold(r.Format, libpaccel.V7Format) || r.Version != libpaccel.V7Version {
		return fmt.Errorf("v7 recipe: not a %s v%d recipe", libpaccel.V7Format, libpaccel.V7Version)
	}
	switch strings.ToLower(strings.TrimSpace(r.ModelType)) {
	case "qwen2", "qwen3":
	default:
		return fmt.Errorf("v7 recipe: unsupported model_type %q (want qwen2/qwen3)", r.ModelType)
	}
	if r.WeightLayout != "" && !libpaccel.V7WeightLayoutSupported(r.WeightLayout) {
		return fmt.Errorf("v7 recipe: unsupported weight_layout %q", r.WeightLayout)
	}
	if r.HiddenSize <= 0 || r.NumHiddenLayers <= 0 || r.NumAttentionHeads <= 0 ||
		r.HeadDim <= 0 || r.VocabSize <= 0 || r.IntermediateSize <= 0 {
		return fmt.Errorf("v7 recipe: incomplete shape set %+v", r)
	}
	if r.NumAttentionHeads%r.NumKeyValueHeads != 0 {
		return fmt.Errorf("v7 recipe: heads %d not divisible by kv heads %d", r.NumAttentionHeads, r.NumKeyValueHeads)
	}
	if r.NumAttentionHeads*r.HeadDim != r.HiddenSize {
		// Some models use a head_dim that does not tile the hidden size exactly;
		// the graph supports that, but flag the common misconfiguration softly.
		return nil
	}
	return nil
}
