package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// hfEndpoint is the Hugging Face Hub host. Override with HF_ENDPOINT to point at
// a mirror or a private hub.
func hfEndpoint() string {
	if v := strings.TrimSpace(os.Getenv("HF_ENDPOINT")); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "https://huggingface.co"
}

// hfSibling is one file entry returned by the model-info API.
type hfSibling struct {
	RFilename string `json:"rfilename"`
	Size      int64  `json:"size"`
}

type hfModelInfo struct {
	ID       string      `json:"id"`
	Siblings []hfSibling `json:"siblings"`
}

func hfClient() *http.Client {
	return &http.Client{Timeout: 0} // large weight files: no overall deadline
}

func hfAuth(req *http.Request) {
	if tok := strings.TrimSpace(os.Getenv("HF_TOKEN")); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	req.Header.Set("User-Agent", "paccelchat/0.1 (+libpaccel)")
}

// fetchModelInfo lists the files in a repo at a given revision.
func fetchModelInfo(repo, revision string) (*hfModelInfo, error) {
	url := fmt.Sprintf("%s/api/models/%s/revision/%s", hfEndpoint(), repo, revision)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	hfAuth(req)
	resp, err := hfClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("hf model info %s: HTTP %d %s", repo, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var info hfModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, err
	}
	return &info, nil
}

// wantFile selects the minimal file set a chatbot needs: weights, model config,
// and tokenizer assets. Everything else (images, eval logs, ONNX exports) is
// skipped to keep "bare minimum" downloads small.
func wantFile(name string) bool {
	lower := strings.ToLower(name)
	if strings.HasSuffix(lower, ".safetensors") {
		return true
	}
	switch filepath.Base(lower) {
	case "config.json", "generation_config.json",
		"tokenizer.json", "tokenizer_config.json",
		"vocab.json", "merges.txt", "special_tokens_map.json":
		return true
	}
	return false
}

// PullModel downloads the selected files of repo@revision into destDir, skipping
// files that already exist with a non-zero size. Returns the local directory.
func PullModel(repo, revision, destDir string) (string, error) {
	if revision == "" {
		revision = "main"
	}
	info, err := fetchModelInfo(repo, revision)
	if err != nil {
		return "", err
	}
	target := filepath.Join(destDir, strings.ReplaceAll(repo, "/", "__"))
	if err := os.MkdirAll(target, 0o755); err != nil {
		return "", err
	}

	var selected []hfSibling
	for _, s := range info.Siblings {
		if wantFile(s.RFilename) {
			selected = append(selected, s)
		}
	}
	if len(selected) == 0 {
		return "", fmt.Errorf("repo %s has no downloadable weight/tokenizer files", repo)
	}
	fmt.Printf("pulling %s@%s -> %s (%d files)\n", repo, revision, target, len(selected))
	for _, s := range selected {
		if err := downloadFile(repo, revision, s.RFilename, target); err != nil {
			return "", fmt.Errorf("download %s: %w", s.RFilename, err)
		}
	}
	return target, nil
}

func downloadFile(repo, revision, rfilename, target string) error {
	dst := filepath.Join(target, filepath.FromSlash(rfilename))
	if info, err := os.Stat(dst); err == nil && info.Size() > 0 {
		fmt.Printf("  = %s (cached)\n", rfilename)
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	url := fmt.Sprintf("%s/%s/resolve/%s/%s", hfEndpoint(), repo, revision, rfilename)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	hfAuth(req)
	resp, err := hfClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("HTTP %d %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	tmp := dst + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	start := time.Now()
	n, err := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if err != nil {
		os.Remove(tmp)
		return err
	}
	if closeErr != nil {
		os.Remove(tmp)
		return closeErr
	}
	if err := os.Rename(tmp, dst); err != nil {
		return err
	}
	fmt.Printf("  + %s (%s in %s)\n", rfilename, humanBytes(uint64(n)), time.Since(start).Truncate(time.Millisecond))
	return nil
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
