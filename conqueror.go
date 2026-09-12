package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	defaultConquerorModel   = defaultConqueror1p5
	defaultConquerorBackend = "splashml-plus"
	defaultConquerorTimeout = 90 * time.Second
	defaultNumPredict       = 256
)

type RuntimeMode string

const (
	RuntimeAuto      RuntimeMode = "auto"
	RuntimeLocalOnly RuntimeMode = "local"
	RuntimeConqueror RuntimeMode = "conqueror"
)

type ConquerorConfig struct {
	Mode       RuntimeMode
	Binary     string
	Model      string
	ModelsDir  string
	Backend    string
	Timeout    time.Duration
	NumPredict int
	ReasonSize string
	LiveSearch bool
}

type ReplyEngine struct {
	model     *StellaModel
	conqueror ConquerorConfig
}

func NewReplyEngine(model *StellaModel, cfg ConquerorConfig) ReplyEngine {
	cfg = cfg.normalized()
	return ReplyEngine{model: model, conqueror: cfg}
}

func (e ReplyEngine) Reply(prompt string) Prediction {
	cfg := e.conqueror.normalized()
	searchQ := prompt
	artifact := looksLikeArtifactRequest(prompt)
	if artifact {
		searchQ = researchQueryFrom(prompt, "")
	}
	if cfg.LiveSearch && cfg.Mode != RuntimeLocalOnly {
		_, _ = indexScienceOpenIntoModel(e.model, searchQ, 6)
	}
	local := e.model.Predict(searchQ)
	hits := NewTensorEngine().Search(searchQ, e.model.Chunks, nil, 6)
	if len(hits) == 0 {
		hits = NewTensorEngine().Search(searchQ, e.model.Chunks, e.model.Samples, 6)
	}
	if len(hits) == 0 && searchQ != prompt {
		hits = NewTensorEngine().Search(prompt, e.model.Chunks, e.model.Samples, 6)
	}
	if cfg.Mode == RuntimeLocalOnly {
		if artifact {
			text := fallbackResearchScript(prompt, hits)
			return Prediction{Text: text, Confidence: 0.7, Tokens: tokenize(text), Source: "artifact-fallback"}
		}
		return local
	}
	if artifact {
		return e.replyArtifact(prompt, local, hits)
	}
	if local.Source == "memory" && local.Confidence >= 0.97 {
		return local
	}
	reasonPrompt := e.model.BuildReasonPrompt(prompt, local, hits)
	if reply, err := e.queryReasoner(reasonPrompt); err == nil && strings.TrimSpace(reply) != "" {
		return Prediction{
			Text:       reply,
			Confidence: maxFloat(local.Confidence, 0.82),
			Tokens:     tokenize(reply),
			Source:     "reasoner:" + cfg.ReasonSize + ":" + ollamaModelFor(cfg.ReasonSize),
		}
	}
	reply, err := e.queryConqueror(prompt, local)
	if err != nil {
		if cfg.Mode == RuntimeConqueror {
			return Prediction{
				Text:       fmt.Sprintf("%s\n\n(Reasoner error: %v)", local.Text, err),
				Confidence: local.Confidence,
				Tokens:     local.Tokens,
				Source:     local.Source + "+reasoner-error",
			}
		}
		return local
	}
	if reply == "" {
		return local
	}
	return Prediction{
		Text:       reply,
		Confidence: maxFloat(local.Confidence, 0.72),
		Tokens:     tokenize(reply),
		Source:     "conqueror:" + cfg.Model,
	}
}

func (e ReplyEngine) queryReasoner(prompt string) (string, error) {
	return e.queryReasonerPredict(prompt, 512)
}

func (e ReplyEngine) queryReasonerPredict(prompt string, numPredict int) (string, error) {
	cfg := e.conqueror.normalized()
	if !ollamaAlive() {
		return "", errors.New("ollama is not reachable")
	}
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()
	return queryOllamaPredict(ctx, ollamaModelFor(cfg.ReasonSize), prompt, numPredict)
}

func (e ReplyEngine) replyArtifact(prompt string, local Prediction, hits []RankedChunk) Prediction {
	codePrompt := e.model.BuildCodePrompt(prompt, local, hits)
	if reply, err := e.queryReasonerPredict(codePrompt, 1536); err == nil && strings.TrimSpace(reply) != "" {
		code := extractFencedCode(reply)
		if code == "" && looksLikeSource(reply) {
			code = strings.TrimSpace(reply)
		}
		if code != "" {
			return Prediction{
				Text:       code,
				Confidence: maxFloat(local.Confidence, 0.8),
				Tokens:     tokenize(code),
				Source:     "reasoner-artifact:" + e.conqueror.normalized().ReasonSize,
			}
		}
	}
	text := fallbackResearchScript(prompt, hits)
	return Prediction{
		Text:       text,
		Confidence: 0.72,
		Tokens:     tokenize(text),
		Source:     "artifact-fallback",
	}
}

func (e ReplyEngine) Status() map[string]any {
	cfg := e.conqueror.normalized()
	return map[string]any{
		"mode":          string(cfg.Mode),
		"enabled":       cfg.reasonerEnabled(),
		"binary":        cfg.Binary,
		"model":         ollamaModelFor(cfg.ReasonSize),
		"reason_size":   cfg.ReasonSize,
		"models_dir":    cfg.ModelsDir,
		"backend":       cfg.Backend,
		"timeout_ms":    cfg.Timeout.Milliseconds(),
		"num_predict":   cfg.NumPredict,
		"live_search":   cfg.LiveSearch,
		"tensor_device": tensorDeviceLabel(),
		"chunks":        len(e.model.Chunks),
		"ollama":        ollamaAlive(),
	}
}

func (e ReplyEngine) queryConqueror(userPrompt string, local Prediction) (string, error) {
	cfg := e.conqueror.normalized()
	if cfg.Binary == "" {
		return "", errors.New("conqueror binary was not found")
	}
	if cfg.ModelsDir == "" {
		return "", errors.New("conqueror models directory was not found")
	}
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()
	wrappedPrompt := e.model.BuildConquerorPrompt(userPrompt, local)
	args := []string{
		"query",
		"--model", cfg.Model,
		"--models-dir", cfg.ModelsDir,
		"--backend", cfg.Backend,
		"--prompt", wrappedPrompt,
		"--raw",
	}
	cmd := exec.CommandContext(ctx, cfg.Binary, args...)
	cmd.Dir = defaultCommandDir()
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("INFERENCE_NUM_PREDICT=%d", cfg.NumPredict),
		"INFERENCE_TEMPERATURE=0",
	)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return "", fmt.Errorf("query timed out after %s", cfg.Timeout)
		}
		return "", fmt.Errorf("%w: %s", err, strings.TrimSpace(out.String()))
	}
	return cleanConquerorOutput(out.String()), nil
}

func (cfg ConquerorConfig) normalized() ConquerorConfig {
	if cfg.Mode == "" {
		cfg.Mode = RuntimeMode(envOr("STELLA_BACKEND_MODE", string(RuntimeAuto)))
	}
	switch RuntimeMode(strings.ToLower(strings.TrimSpace(string(cfg.Mode)))) {
	case RuntimeLocalOnly:
		cfg.Mode = RuntimeLocalOnly
	case RuntimeConqueror:
		cfg.Mode = RuntimeConqueror
	default:
		cfg.Mode = RuntimeAuto
	}
	cfg.ReasonSize = NormalizeReasonSize(firstNonEmpty(cfg.ReasonSize, envOr("STELLAV_REASON_SIZE", Reason1p5B)))
	if cfg.Model == "" {
		cfg.Model = envOr("STELLA_CONQUEROR_MODEL", conquerorModelFor(cfg.ReasonSize))
	}
	cfg.Model = normalizeConquerorModel(cfg.Model)
	if cfg.Backend == "" {
		cfg.Backend = envOr("STELLA_CONQUEROR_BACKEND", defaultConquerorBackend)
	}
	if strings.TrimSpace(os.Getenv("STELLAV_LIVE_SEARCH")) == "0" || cfg.Mode == RuntimeLocalOnly {
		cfg.LiveSearch = false
	} else {
		cfg.LiveSearch = true
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultConquerorTimeout
	}
	if cfg.NumPredict <= 0 {
		cfg.NumPredict = defaultNumPredict
	}
	if cfg.Binary == "" {
		cfg.Binary = envOr("STELLA_CONQUEROR", findConquerorBinary())
	}
	if cfg.ModelsDir == "" {
		cfg.ModelsDir = envOr("STELLA_CONQUEROR_MODELS_DIR", findConquerorModelsDir(cfg.Binary))
	}
	return cfg
}

func (cfg ConquerorConfig) enabled() bool {
	return cfg.reasonerEnabled()
}

func (cfg ConquerorConfig) reasonerEnabled() bool {
	cfg = cfg.normalized()
	if cfg.Mode == RuntimeLocalOnly {
		return false
	}
	if ollamaAlive() {
		return true
	}
	return cfg.Binary != "" && cfg.ModelsDir != ""
}

func normalizeConquerorModel(model string) string {
	key := strings.ToLower(strings.TrimSpace(model))
	key = strings.ReplaceAll(key, "_", "-")
	switch key {
	case "", "qwen", "1.5b", "qwen2.5:1.5b", "qwen2.5-1.5b":
		return conquerorModelFor(Reason1p5B)
	case "3b", "qwen2.5:3b", "qwen2.5-3b":
		return conquerorModelFor(Reason3B)
	case "qwen3", "qwen3-0.6b", "qwen3-0-6b", "qwen-3-0.6b", "qwen/qwen3-0.6b":
		return "hf-qwen-qwen3-0-6b"
	default:
		return key
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func cleanConquerorOutput(raw string) string {
	var lines []string
	for _, line := range strings.Split(raw, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "localruntime:") ||
			strings.HasPrefix(trimmed, "paccel:") ||
			strings.HasPrefix(trimmed, "conqueror:") {
			continue
		}
		lines = append(lines, strings.TrimRight(line, "\r"))
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func findConquerorBinary() string {
	names := []string{"conqueror"}
	if runtime.GOOS == "windows" {
		names = []string{"conqueror.exe", "Conqueror.exe"}
	}
	for _, candidate := range candidateConquerorRoots() {
		for _, name := range names {
			path := filepath.Join(candidate, name)
			if fileExistsExecutable(path) {
				return path
			}
		}
	}
	if path, err := exec.LookPath("conqueror"); err == nil {
		return path
	}
	return ""
}

func findConquerorModelsDir(binary string) string {
	if binary != "" {
		dir := filepath.Join(filepath.Dir(binary), "models")
		if dirExists(dir) {
			return dir
		}
	}
	for _, root := range candidateConquerorRoots() {
		dir := filepath.Join(root, "models")
		if dirExists(dir) {
			return dir
		}
	}
	return ""
}

func candidateConquerorRoots() []string {
	var out []string
	add := func(path string) {
		path = filepath.Clean(path)
		if path != "." && !containsString(out, path) {
			out = append(out, path)
		}
	}
	platform := conquerorPlatformDir()
	if cwd, err := os.Getwd(); err == nil {
		add(filepath.Join(cwd, "build", "Release", "package", "bin", "Release", platform))
		add(filepath.Join(cwd, "package", "bin", "Release", platform))
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		for _, rel := range []string{
			filepath.Join("..", "..", "package", "bin", "Release", platform),
			filepath.Join("..", "..", "..", "package", "bin", "Release", platform),
			filepath.Join("..", "package", "bin", "Release", platform),
		} {
			add(filepath.Join(exeDir, rel))
		}
	}
	return out
}

func conquerorPlatformDir() string {
	osPart := "linux"
	switch runtime.GOOS {
	case "darwin":
		osPart = "osx"
	case "windows":
		osPart = "win"
	}
	archPart := "x64"
	if runtime.GOARCH == "arm64" {
		archPart = "arm64"
	}
	return "conqueror-" + osPart + "-" + archPart
}

func defaultCommandDir() string {
	if cwd, err := os.Getwd(); err == nil {
		return cwd
	}
	return "."
}

func fileExistsExecutable(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir() && info.Mode()&0o111 != 0
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func parseDurationFlag(raw string) (time.Duration, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return defaultConquerorTimeout, nil
	}
	if d, err := time.ParseDuration(raw); err == nil {
		return d, nil
	}
	return 0, fmt.Errorf("invalid duration %q", raw)
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
