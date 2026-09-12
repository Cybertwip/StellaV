package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/powerengine/paccel/libpaccel"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Stella: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return nil
	}
	switch args[0] {
	case "bootstrap":
		return cmdBootstrap(args[1:])
	case "train":
		return cmdTrain(args[1:])
	case "ask":
		return cmdAsk(args[1:])
	case "chat":
		return cmdChat(args[1:])
	case "teach":
		return cmdTeach(args[1:])
	case "gui", "serve":
		return cmdGUI(args[1:])
	case "index":
		return cmdIndex(args[1:])
	case "export":
		return cmdExport(args[1:])
	case "info":
		return cmdInfo(args[1:])
	case "-h", "--help", "help":
		usage()
		return nil
	default:
		usage()
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func usage() {
	fmt.Print(`Stella V - medical-research assistant (ScienceOpen + 1.5B/3B reasoner)

usage:
  Stella bootstrap [--model stella_model.json]
  Stella train     [--model stella_model.json] [--data samples.jsonl]
  Stella ask       [--model stella_model.json] [--reason 1.5b|3b] [--backend-mode auto] <question>
  Stella chat      [--model stella_model.json] [--reason 1.5b|3b]
  Stella teach     [--model stella_model.json] <question> <answer>
  Stella index     [--model stella_model.json] [--rows 6] <query or DOI>
  Stella gui|serve [--model stella_model.json] [--addr 127.0.0.1:8765] [--reason 1.5b|3b]
  Stella export    <paccel|safetensors|onnx> [--model stella_model.json] [--bits 4] [--block 32] <output>
  Stella info      [--model stella_model.json] [file.paccel]

serve/gui is the self-contained agent: browser UI plus OpenAI-compatible
endpoints at http://ADDR/v1 for OpenCode and other OpenAI clients.
Models: stella-v, stella-v-1.5b, stella-v-3b, stella-v-local.

ScienceOpen preprints are retrieved with a tensor engine (linear + multi-head
attention). A selectable 1.5B or 3B model reasons over the ranked passages.
Export order: PAccel, then safetensors, then ONNX.
`)
}

func modelFlag(fs *flag.FlagSet) *string {
	return fs.String("model", envOr("STELLA_MODEL", defaultModelPath), "Stella model JSON path")
}

type runtimeFlagSet struct {
	mode         *string
	llm          *string
	reason       *string
	conquerorBin *string
	modelsDir    *string
	backend      *string
	timeout      *string
	numPredict   *int
}

func addRuntimeFlags(fs *flag.FlagSet) runtimeFlagSet {
	return runtimeFlagSet{
		mode:         fs.String("backend-mode", envOr("STELLA_BACKEND_MODE", string(RuntimeAuto)), "Reply backend: auto, conqueror, or local"),
		llm:          fs.String("llm", envOr("STELLA_CONQUEROR_MODEL", ""), "Optional Conqueror model id"),
		reason:       fs.String("reason", envOr("STELLAV_REASON_SIZE", Reason1p5B), "Reasoning model size: 1.5b or 3b"),
		conquerorBin: fs.String("conqueror", envOr("STELLA_CONQUEROR", ""), "Path to the conqueror tool"),
		modelsDir:    fs.String("models-dir", envOr("STELLA_CONQUEROR_MODELS_DIR", ""), "Conqueror managed models directory"),
		backend:      fs.String("conqueror-backend", envOr("STELLA_CONQUEROR_BACKEND", defaultConquerorBackend), "Conqueror inference backend"),
		timeout:      fs.String("timeout", envOr("STELLA_CONQUEROR_TIMEOUT", defaultConquerorTimeout.String()), "Reasoner query timeout"),
		numPredict:   fs.Int("num-predict", defaultNumPredict, "Generated token budget per reply"),
	}
}

func (f runtimeFlagSet) config() (ConquerorConfig, error) {
	timeout := defaultConquerorTimeout
	if f.timeout != nil {
		parsed, err := parseDurationFlag(*f.timeout)
		if err != nil {
			return ConquerorConfig{}, err
		}
		timeout = parsed
	}
	cfg := ConquerorConfig{
		Mode:       RuntimeMode(strings.ToLower(strings.TrimSpace(valueOf(f.mode)))),
		Binary:     valueOf(f.conquerorBin),
		Model:      valueOf(f.llm),
		ModelsDir:  valueOf(f.modelsDir),
		Backend:    valueOf(f.backend),
		Timeout:    timeout,
		ReasonSize: NormalizeReasonSize(valueOf(f.reason)),
		LiveSearch: true,
	}
	if f.numPredict != nil {
		cfg.NumPredict = *f.numPredict
	}
	return cfg.normalized(), nil
}

func valueOf(ptr *string) string {
	if ptr == nil {
		return ""
	}
	return strings.TrimSpace(*ptr)
}

func cmdBootstrap(args []string) error {
	fs := flag.NewFlagSet("bootstrap", flag.ContinueOnError)
	modelPath := modelFlag(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m := NewStellaModel()
	for _, sample := range BootstrapSamples() {
		m.AddSample(sample.Question, sample.Answer)
	}
	if err := m.Save(*modelPath); err != nil {
		return err
	}
	fmt.Printf("wrote %s (%d samples, %d tokens)\n", *modelPath, len(m.Samples), len(m.Vocab))
	return nil
}

func cmdTrain(args []string) error {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	modelPath := modelFlag(fs)
	dataPath := fs.String("data", "", "JSONL samples with question/answer fields")
	if err := fs.Parse(args); err != nil {
		return err
	}
	m := NewStellaModel()
	for _, sample := range BootstrapSamples() {
		m.AddSample(sample.Question, sample.Answer)
	}
	if *dataPath != "" {
		samples, err := readJSONLSamples(*dataPath)
		if err != nil {
			return err
		}
		for _, sample := range samples {
			m.AddSample(sample.Question, sample.Answer)
		}
	}
	if err := m.Save(*modelPath); err != nil {
		return err
	}
	stats := m.Stats()
	fmt.Printf("trained %s: samples=%d vocab=%d transitions=%d\n", *modelPath, stats.Samples, stats.Vocab, stats.Transitions)
	return nil
}

func cmdAsk(args []string) error {
	fs := flag.NewFlagSet("ask", flag.ContinueOnError)
	modelPath := modelFlag(fs)
	runtimeFlags := addRuntimeFlags(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("ask requires a question")
	}
	m, err := LoadOrBootstrap(*modelPath)
	if err != nil {
		return err
	}
	cfg, err := runtimeFlags.config()
	if err != nil {
		return err
	}
	reply := NewReplyEngine(m, cfg).Reply(strings.Join(fs.Args(), " "))
	fmt.Printf("%s\nconfidence=%.4f source=%s\n", reply.Text, reply.Confidence, reply.Source)
	return nil
}

func cmdTeach(args []string) error {
	fs := flag.NewFlagSet("teach", flag.ContinueOnError)
	modelPath := modelFlag(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 {
		return fmt.Errorf("teach requires <question> <answer>")
	}
	m, err := LoadOrBootstrap(*modelPath)
	if err != nil {
		return err
	}
	m.AddSample(fs.Arg(0), strings.Join(fs.Args()[1:], " "))
	if err := m.Save(*modelPath); err != nil {
		return err
	}
	stats := m.Stats()
	fmt.Printf("saved %s: samples=%d vocab=%d transitions=%d\n", *modelPath, stats.Samples, stats.Vocab, stats.Transitions)
	return nil
}

func cmdChat(args []string) error {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	modelPath := modelFlag(fs)
	runtimeFlags := addRuntimeFlags(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m, err := LoadOrBootstrap(*modelPath)
	if err != nil {
		return err
	}
	cfg, err := runtimeFlags.config()
	if err != nil {
		return err
	}
	engine := NewReplyEngine(m, cfg)
	fmt.Printf("Stella V medical chat (%s). reason=%s model=%s device=%s. Commands: /teach question => answer, /index query, /export paccel path, /save, /exit\n",
		*modelPath, engine.Status()["reason_size"], engine.Status()["model"], engine.Status()["tensor_device"])
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for {
		fmt.Print("\nyou> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "/exit" || line == "/quit" {
			break
		}
		if line == "/save" {
			if err := m.Save(*modelPath); err != nil {
				fmt.Printf("error: %v\n", err)
			} else {
				fmt.Println("saved.")
			}
			continue
		}
		if strings.HasPrefix(line, "/teach ") {
			body := strings.TrimSpace(strings.TrimPrefix(line, "/teach "))
			parts := strings.SplitN(body, "=>", 2)
			if len(parts) != 2 {
				fmt.Println("use: /teach question => answer")
				continue
			}
			m.AddSample(parts[0], parts[1])
			_ = m.Save(*modelPath)
			engine = NewReplyEngine(m, cfg)
			fmt.Println("learned.")
			continue
		}
		if strings.HasPrefix(line, "/index ") {
			query := strings.TrimSpace(strings.TrimPrefix(line, "/index "))
			n, err := indexScienceOpenIntoModel(m, query, 8)
			if err != nil {
				fmt.Printf("error: %v\n", err)
			} else {
				_ = m.Save(*modelPath)
				fmt.Printf("indexed %d ScienceOpen preprint(s).\n", n)
			}
			engine = NewReplyEngine(m, cfg)
			continue
		}
		if strings.HasPrefix(line, "/export ") {
			fields := strings.Fields(line)
			if len(fields) != 3 {
				fmt.Println("use: /export paccel|safetensors|onnx path")
				continue
			}
			if err := exportModel(m, fields[1], fields[2], 4, 32); err != nil {
				fmt.Printf("error: %v\n", err)
			} else {
				fmt.Printf("wrote %s\n", fields[2])
			}
			continue
		}
		reply := engine.Reply(line)
		fmt.Printf("stella> %s\n", reply.Text)
		fmt.Printf("        confidence=%.3f source=%s\n", reply.Confidence, reply.Source)
	}
	return m.Save(*modelPath)
}

func cmdGUI(args []string) error {
	fs := flag.NewFlagSet("gui", flag.ContinueOnError)
	modelPath := modelFlag(fs)
	addr := fs.String("addr", envOr("STELLA_ADDR", "127.0.0.1:8765"), "HTTP listen address")
	runtimeFlags := addRuntimeFlags(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	m, err := LoadOrBootstrap(*modelPath)
	if err != nil {
		return err
	}
	if err := m.Save(*modelPath); err != nil {
		return err
	}
	cfg, err := runtimeFlags.config()
	if err != nil {
		return err
	}
	fmt.Printf("Stella V agent: http://%s\n", *addr)
	fmt.Printf("OpenAI API:     http://%s/v1\n", *addr)
	fmt.Printf("OpenCode:       baseURL=http://%s/v1  model=stella-v-%s\n", *addr, NormalizeReasonSize(cfg.ReasonSize))
	return RunGUI(*addr, *modelPath, m, cfg)
}

func cmdIndex(args []string) error {
	fs := flag.NewFlagSet("index", flag.ContinueOnError)
	modelPath := modelFlag(fs)
	rows := fs.Int("rows", 8, "ScienceOpen hits per query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("index requires a ScienceOpen query or DOI")
	}
	m, err := LoadOrBootstrap(*modelPath)
	if err != nil {
		return err
	}
	query := strings.Join(fs.Args(), " ")
	n, err := indexScienceOpenIntoModel(m, query, *rows)
	if err != nil {
		return err
	}
	if err := m.Save(*modelPath); err != nil {
		return err
	}
	fmt.Printf("indexed %d ScienceOpen preprint(s) into %s (chunks=%d)\n", n, *modelPath, len(m.Chunks))
	return nil
}

func cmdExport(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("export requires paccel, safetensors, or onnx")
	}
	format := strings.ToLower(args[0])
	fs := flag.NewFlagSet("export "+format, flag.ContinueOnError)
	modelPath := modelFlag(fs)
	bits := fs.Uint("bits", libpaccel.DefaultBits, "PAccel quantization bits")
	block := fs.Uint("block", libpaccel.DefaultBlockSize, "PAccel block size")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("export %s requires <output>", format)
	}
	m, err := LoadOrBootstrap(*modelPath)
	if err != nil {
		return err
	}
	output := fs.Arg(0)
	if err := exportModel(m, format, output, uint32(*bits), uint32(*block)); err != nil {
		return err
	}
	fmt.Printf("wrote %s\n", output)
	return nil
}

func exportModel(m *StellaModel, format, output string, bits, block uint32) error {
	switch strings.ToLower(format) {
	case "paccel":
		return m.ExportPAccel(output, bits, block)
	case "safe", "safetensor", "safetensors":
		return m.ExportSafetensors(output)
	case "onnx":
		return m.ExportONNX(output)
	default:
		return fmt.Errorf("unsupported export format %q", format)
	}
}

func cmdInfo(args []string) error {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	modelPath := modelFlag(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 1 && strings.EqualFold(filepath.Ext(fs.Arg(0)), ".paccel") {
		pkg, err := libpaccel.ReadPackage(fs.Arg(0))
		if err != nil {
			return err
		}
		fmt.Printf("%s: version=%d bits=%d block=%d records=%d source=%s\n",
			fs.Arg(0), pkg.Version, pkg.BitWidth, pkg.BlockSize, len(pkg.Records), humanBytes(pkg.ModelSize))
		return nil
	}
	m, err := LoadOrBootstrap(*modelPath)
	if err != nil {
		return err
	}
	stats := m.Stats()
	fmt.Printf("%s: samples=%d vocab=%d transitions=%d chunks=%d updated=%d\n",
		*modelPath, stats.Samples, stats.Vocab, stats.Transitions, stats.Chunks, stats.UpdatedAt)
	return nil
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func humanBytes(n uint64) string {
	const unit = 1024.0
	v := float64(n)
	suffix := "B"
	for _, s := range []string{"KiB", "MiB", "GiB", "TiB"} {
		if v < unit {
			break
		}
		v /= unit
		suffix = s
	}
	if suffix == "B" {
		return fmt.Sprintf("%d B", n)
	}
	return fmt.Sprintf("%.1f %s", v, suffix)
}
