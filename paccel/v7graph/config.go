package v7graph

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/powerengine/paccel/libpaccel"
)

// hfConfig is the subset of a Hugging Face config.json the V7 recipe needs.
type hfConfig struct {
	ModelType         string  `json:"model_type"`
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

// RecipeFromHFConfig builds a V7 recipe from a Hugging Face config.json, filling
// the format/layout metadata that marks it a direct-only V7 snapshot.
func RecipeFromHFConfig(configPath string) (*Recipe, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	var c hfConfig
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	r := &Recipe{
		Format:            libpaccel.V7Format,
		Version:           libpaccel.V7Version,
		GraphKind:         libpaccel.V7GraphKind,
		ModelType:         c.ModelType,
		WeightLayout:      libpaccel.V7WeightLayout,
		PackageMode:       libpaccel.V7PackageMode,
		Weights:           "model.paccel",
		HiddenSize:        c.HiddenSize,
		IntermediateSize:  c.IntermediateSize,
		NumHiddenLayers:   c.NumHiddenLayers,
		NumAttentionHeads: c.NumAttentionHeads,
		NumKeyValueHeads:  c.NumKeyValueHeads,
		HeadDim:           c.HeadDim,
		VocabSize:         c.VocabSize,
		RMSNormEps:        c.RMSNormEps,
		RopeTheta:         c.RopeTheta,
		HiddenAct:         c.HiddenAct,
		TieWordEmbeddings: c.TieWordEmbeddings,
	}
	if err := r.fillDefaults(); err != nil {
		return nil, err
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return r, nil
}

// Save writes the recipe as model.v7.json.
func (r *Recipe) Save(path string) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}
