// Command paccelchat is the bare-minimum PAccel frontend: a simple chatbot with
// Hugging Face model-download capabilities. It depends only on the standard
// library and libpaccel — no onnxruntime, no protobuf, no cgo. It can pull a
// model from the Hub, compress it into a .paccel container, inspect a container,
// and run an interactive chat grounded in the model's dequantized embeddings.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/powerengine/paccel/libpaccel"
	"github.com/powerengine/paccel/v7graph"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "pull":
		err = cmdPull(os.Args[2:])
	case "compress":
		err = cmdCompress(os.Args[2:])
	case "info":
		err = cmdInfo(os.Args[2:])
	case "chat":
		err = cmdChat(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "paccelchat: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Print(`paccelchat — bare-minimum PAccel chatbot (no onnxruntime)

usage:
  paccelchat pull     <repo> [--rev main] [--out ./models]
  paccelchat compress <model-dir|file.safetensors> <out.paccel> [--bits 4] [--block 32] [--v7]
  paccelchat info     <file.paccel>
  paccelchat chat     [--paccel file.paccel] [--model model-dir] [--recipe model.v7.json] [--max-tokens 24]

examples:
  paccelchat pull Qwen/Qwen2.5-0.5B-Instruct --out ./models
  paccelchat compress ./models/qwen ./models/qwen/model.paccel --bits 4 --v7   # emits model.v7.json
  paccelchat chat --paccel ./models/qwen/model.paccel --model ./models/qwen     # auto-uses V7 direct graph

environment:
  HF_TOKEN     bearer token for gated/private repos
  HF_ENDPOINT  alternate hub host (default https://huggingface.co)
`)
}

func cmdPull(args []string) error {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	rev := fs.String("rev", "main", "git revision/branch/tag")
	out := fs.String("out", "./models", "download directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("pull requires exactly one <repo> argument")
	}
	dir, err := PullModel(fs.Arg(0), *rev, *out)
	if err != nil {
		return err
	}
	fmt.Printf("done: %s\n", dir)
	return nil
}

func cmdCompress(args []string) error {
	fs := flag.NewFlagSet("compress", flag.ContinueOnError)
	bits := fs.Uint("bits", libpaccel.DefaultBits, "quantization bits (2..8)")
	block := fs.Uint("block", libpaccel.DefaultBlockSize, "quantization block size")
	v7 := fs.Bool("v7", false, "also emit the V7 direct graph: pre-transposed aliases + model.v7.json (optimal route)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("compress requires <input> <out.paccel>")
	}
	input, output := fs.Arg(0), fs.Arg(1)
	files := libpaccel.CollectSafetensorsFiles(input)
	if len(files) == 0 {
		return fmt.Errorf("no .safetensors found under %s", input)
	}
	fmt.Printf("compressing %d shard(s) at %d-bit (block %d)...\n", len(files), *bits, *block)
	pkg, err := libpaccel.CompressSafetensors(files, uint32(*bits), uint32(*block))
	if err != nil {
		return err
	}
	if *v7 {
		if err := libpaccel.AppendV7Aliases(pkg, uint32(*bits), uint32(*block)); err != nil {
			return fmt.Errorf("v7 aliases: %w", err)
		}
		configDir := input
		if info, statErr := os.Stat(input); statErr == nil && !info.IsDir() {
			configDir = filepath.Dir(input)
		}
		recipe, rErr := v7graph.RecipeFromHFConfig(filepath.Join(configDir, "config.json"))
		if rErr != nil {
			fmt.Fprintf(os.Stderr, "v7: config.json not usable (%v); package has aliases but no recipe\n", rErr)
		} else if wErr := recipe.Save(filepath.Join(filepath.Dir(output), libpaccel.V7RecipeFileName)); wErr != nil {
			return fmt.Errorf("write v7 recipe: %w", wErr)
		} else {
			fmt.Printf("wrote V7 recipe %s (%s, %d layers)\n",
				filepath.Join(filepath.Dir(output), libpaccel.V7RecipeFileName), recipe.ModelType, recipe.NumHiddenLayers)
		}
	}
	if err := libpaccel.WritePackage(output, pkg); err != nil {
		return err
	}
	src := pkg.ModelSize
	var packed uint64
	for _, rec := range pkg.Records {
		if rec.Compressed() {
			packed += uint64(len(rec.Tensor.Packed)) + uint64(len(rec.Tensor.Scales))*2
		} else {
			packed += uint64(len(rec.RawPayload))
		}
	}
	ratio := 0.0
	if src > 0 {
		ratio = float64(packed) / float64(src)
	}
	fmt.Printf("wrote %s\n  %s\n  source %s -> payload %s (%.3fx)\n",
		output, summarizePackage(pkg), humanBytes(src), humanBytes(packed), ratio)
	return nil
}

func cmdInfo(args []string) error {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	verbose := fs.Bool("v", false, "list every tensor")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("info requires <file.paccel>")
	}
	pkg, err := libpaccel.ReadPackage(fs.Arg(0))
	if err != nil {
		return err
	}
	fmt.Printf("%s\n%s\n", fs.Arg(0), summarizePackage(pkg))
	limit := 12
	for i, rec := range pkg.Records {
		if !*verbose && i >= limit {
			fmt.Printf("  ... and %d more (use -v)\n", len(pkg.Records)-limit)
			break
		}
		shape := rec.Shape
		kind := "raw"
		if rec.Compressed() {
			shape = rec.Tensor.Shape
			kind = fmt.Sprintf("%d-bit", rec.Tensor.BitWidth)
			if rec.Tensor.Layout == libpaccel.PackedLayoutOutputTileInt4 {
				kind += " tiled"
			}
		} else if rec.Encoding == libpaccel.EncodingRawRLE {
			kind = "raw+rle"
		}
		fmt.Printf("  %-40s %-9s %v\n", truncate(rec.Name, 40), kind, shape)
	}
	return nil
}

func cmdChat(args []string) error {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	paccelPath := fs.String("paccel", "", "path to a .paccel container")
	modelDir := fs.String("model", "", "model dir (for vocab.json; auto-compresses safetensors if --paccel omitted)")
	recipe := fs.String("recipe", "", "path to a model.v7.json (forces the V7 direct graph)")
	maxTokens := fs.Int("max-tokens", 24, "tokens to generate per reply")
	if err := fs.Parse(args); err != nil {
		return err
	}

	var engine Engine = EchoEngine{}
	switch {
	case *paccelPath != "":
		// Prefer the V7 direct graph (the optimal route) when a recipe is present.
		recipePath := *recipe
		if recipePath == "" {
			recipePath = findV7Recipe(*paccelPath, *modelDir)
		}
		if recipePath != "" {
			eng, err := LoadV7Engine(recipePath, *paccelPath, *modelDir, *maxTokens)
			if err != nil {
				return err
			}
			engine = eng
			break
		}
		eng, err := LoadPaccelEngine(*paccelPath, *modelDir, *maxTokens)
		if err != nil {
			return err
		}
		engine = eng
	case *modelDir != "":
		// Convenience: compress the model dir to a temp .paccel, then load it.
		files := libpaccel.CollectSafetensorsFiles(*modelDir)
		if len(files) == 0 {
			return fmt.Errorf("no .safetensors under %s and no --paccel given", *modelDir)
		}
		fmt.Println("no --paccel given; compressing model dir to a temporary container...")
		pkg, err := libpaccel.CompressSafetensors(files, libpaccel.DefaultBits, libpaccel.DefaultBlockSize)
		if err != nil {
			return err
		}
		tmp := filepath.Join(*modelDir, "model.paccel")
		if err := libpaccel.WritePackage(tmp, pkg); err != nil {
			return err
		}
		eng, err := LoadPaccelEngine(tmp, *modelDir, *maxTokens)
		if err != nil {
			return err
		}
		engine = eng
	}

	fmt.Printf("engine: %s\n", engine.Name())
	fmt.Println("type your message; 'exit' or Ctrl-D to quit.")
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
		if line == "exit" || line == "quit" {
			break
		}
		reply, err := engine.Reply(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  error: %v\n", err)
			continue
		}
		fmt.Printf("bot> %s\n", reply)
	}
	fmt.Println("\nbye.")
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}
