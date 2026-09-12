package main

import (
	"hash/fnv"
	"math"
	"strings"
)

const (
	retrievalDim   = 64
	retrievalHeads = 4
)

type KnowledgeChunk struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	DOI    string `json:"doi"`
	URL    string `json:"url"`
	Text   string `json:"text"`
	Topic  string `json:"topic"`
	Source string `json:"source"`
}

type RankedChunk struct {
	Chunk KnowledgeChunk
	Score float64
}

type TensorEngine struct {
	dim   int
	heads int
	qProj []float32
	kProj []float32
	vProj []float32
}

func NewTensorEngine() *TensorEngine {
	e := &TensorEngine{dim: retrievalDim, heads: retrievalHeads}
	e.qProj = hashedSquare("stella.query.proj", e.dim)
	e.kProj = hashedSquare("stella.key.proj", e.dim)
	e.vProj = hashedSquare("stella.value.proj", e.dim)
	return e
}

func hashedSquare(seed string, dim int) []float32 {
	out := make([]float32, dim*dim)
	h := fnv.New32a()
	_, _ = h.Write([]byte(seed))
	state := h.Sum32()
	scale := float32(1.0 / math.Sqrt(float64(dim)))
	for i := range out {
		state = state*1664525 + 1013904223
		centered := (float32((state>>8)&0xffff) / 32767.5) - 1
		out[i] = centered * scale
	}
	return out
}

func (e *TensorEngine) Embed(text string) []float32 {
	vec := make([]float32, e.dim)
	blob := strings.ToLower(strings.Join(strings.Fields(text), " "))
	if blob == "" {
		return vec
	}
	features := strings.Fields(blob)
	compact := strings.ReplaceAll(blob, " ", "")
	for n := 2; n <= 4; n++ {
		if len(compact) >= n {
			for i := 0; i+n <= len(compact); i += n {
				features = append(features, compact[i:i+n])
			}
		}
	}
	for _, feat := range features {
		h := fnv32(feat)
		idx := int(h % uint32(e.dim))
		sign := float32(1)
		if h>>31 == 1 {
			sign = -1
		}
		vec[idx] += sign
		vec[int((h/uint32(e.dim))%uint32(e.dim))] += 0.35 * sign
	}
	projected := matVec(e.qProj, e.dim, vec)
	for i := range vec {
		vec[i] = 0.65*vec[i] + 0.35*projected[i]
	}
	return l2norm(vec)
}

func (e *TensorEngine) Search(query string, chunks []KnowledgeChunk, samples []Sample, k int) []RankedChunk {
	q := e.Embed(query)
	cands := make([]KnowledgeChunk, 0, len(chunks)+len(samples))
	cands = append(cands, chunks...)
	for i, sample := range samples {
		cands = append(cands, KnowledgeChunk{
			ID:     "sample-" + itoa(i),
			Title:  sample.Question,
			Text:   sample.Answer,
			Source: "memory",
		})
	}
	if len(cands) == 0 {
		return nil
	}
	keys := make([][]float32, len(cands))
	for i, chunk := range cands {
		keys[i] = e.Embed(chunk.Title + " " + chunk.Text)
	}
	linear := make([]float64, len(cands))
	attn := e.attentionScores(q, keys)
	for i, key := range keys {
		linear[i] = float64(dot(q, key))
	}
	ranked := make([]RankedChunk, 0, len(cands))
	for i, chunk := range cands {
		ranked = append(ranked, RankedChunk{Chunk: chunk, Score: 0.45*linear[i] + 0.55*attn[i]})
	}
	sortRanked(ranked)
	if k <= 0 || k > len(ranked) {
		k = len(ranked)
	}
	return ranked[:k]
}

func (e *TensorEngine) attentionScores(query []float32, keys [][]float32) []float64 {
	headDim := e.dim / e.heads
	qHeads := matVec(e.qProj, e.dim, query)
	out := make([]float64, len(keys))
	scale := 1.0 / math.Sqrt(float64(headDim))
	for i, key := range keys {
		k := matVec(e.kProj, e.dim, key)
		sum := 0.0
		for h := 0; h < e.heads; h++ {
			off := h * headDim
			sum += float64(dot(qHeads[off:off+headDim], k[off:off+headDim])) * scale
		}
		out[i] = sum / float64(e.heads)
	}
	return out
}

func (e *TensorEngine) RetrievalTensors() []ModelTensor {
	return []ModelTensor{
		{Name: "stella.retrieval.query_proj", DType: "F32", Shape: []uint64{uint64(e.dim), uint64(e.dim)}, F32: e.qProj},
		{Name: "stella.retrieval.key_proj", DType: "F32", Shape: []uint64{uint64(e.dim), uint64(e.dim)}, F32: e.kProj},
		{Name: "stella.retrieval.value_proj", DType: "F32", Shape: []uint64{uint64(e.dim), uint64(e.dim)}, F32: e.vProj},
	}
}

func matVec(square []float32, dim int, vec []float32) []float32 {
	out := make([]float32, dim)
	for r := 0; r < dim; r++ {
		sum := float32(0)
		row := r * dim
		for c := 0; c < dim && c < len(vec); c++ {
			sum += square[row+c] * vec[c]
		}
		out[r] = sum
	}
	return out
}

func dot(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func l2norm(v []float32) []float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(float64(sum)))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func sortRanked(items []RankedChunk) {
	for i := 1; i < len(items); i++ {
		j := i
		for j > 0 && items[j].Score > items[j-1].Score {
			items[j], items[j-1] = items[j-1], items[j]
			j--
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [16]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
