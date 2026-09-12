package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/powerengine/paccel/libpaccel"
)

type safetensorsMeta struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

func (m *StellaModel) ExportSafetensors(path string) error {
	if path == "" {
		return fmt.Errorf("safetensors output path is required")
	}
	if err := os.MkdirAll(filepath.Dir(cleanPathForCreate(path)), 0o755); err != nil {
		return err
	}
	tensors := m.Tensors()
	payloads := make([][]byte, 0, len(tensors))
	header := map[string]any{
		"__metadata__": map[string]string{
			"format":        "stella-v-go",
			"model_version": strconv.Itoa(modelVersion),
			"samples":       strconv.Itoa(len(m.Samples)),
			"vocab_size":    strconv.Itoa(len(m.Vocab)),
			"vocab_json":    mustJSONString(m.Vocab),
		},
	}
	var offset uint64
	for _, tensor := range tensors {
		raw, err := tensorRaw(tensor)
		if err != nil {
			return err
		}
		payloads = append(payloads, raw)
		begin := offset
		offset += uint64(len(raw))
		header[tensor.Name] = safetensorsMeta{
			DType:       tensor.DType,
			Shape:       append([]uint64(nil), tensor.Shape...),
			DataOffsets: []uint64{begin, offset},
		}
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	if _, err = f.Write(lenBuf[:]); err == nil {
		_, err = f.Write(headerBytes)
	}
	for _, raw := range payloads {
		if err != nil {
			break
		}
		_, err = f.Write(raw)
	}
	closeErr := f.Close()
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, path)
}

func (m *StellaModel) ExportPAccel(path string, bits, block uint32) error {
	if path == "" {
		return fmt.Errorf("paccel output path is required")
	}
	if bits == 0 {
		bits = libpaccel.DefaultBits
	}
	if block == 0 {
		block = libpaccel.DefaultBlockSize
	}
	dir := filepath.Dir(cleanPathForCreate(path))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".stella-*.safetensors")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	_ = tmp.Close()
	defer os.Remove(tmpPath)
	if err := m.ExportSafetensors(tmpPath); err != nil {
		return err
	}
	pkg, err := libpaccel.CompressSafetensors([]string{tmpPath}, bits, block)
	if err != nil {
		return err
	}
	return libpaccel.WritePackage(path, pkg)
}

func (m *StellaModel) ExportONNX(path string) error {
	if path == "" {
		return fmt.Errorf("onnx output path is required")
	}
	if err := os.MkdirAll(filepath.Dir(cleanPathForCreate(path)), 0o755); err != nil {
		return err
	}
	tensors := m.Tensors()
	var weight, bias ModelTensor
	for _, tensor := range tensors {
		switch tensor.Name {
		case "stella.output.weight":
			weight = tensor
		case "stella.output.bias":
			bias = tensor
		}
	}
	if len(weight.F32) == 0 || len(bias.F32) == 0 {
		return fmt.Errorf("stella output tensors are empty")
	}
	graph := onnxGraphDef{
		name: "stella_v_chatbot",
		nodes: []onnxNodeDef{
			{name: "stella.matmul", opType: "MatMul", inputs: []string{"features", "stella.output.weight"}, outputs: []string{"scores"}},
			{name: "stella.bias", opType: "Add", inputs: []string{"scores", "stella.output.bias"}, outputs: []string{"logits"}},
		},
		initializers: []onnxTensorDef{
			{name: "stella.output.weight", dims: []int64{embeddingDim, int64(len(m.Vocab))}, dataType: onnxDTFloat, rawData: f32Raw(weight.F32)},
			{name: "stella.output.bias", dims: []int64{int64(len(m.Vocab))}, dataType: onnxDTFloat, rawData: f32Raw(bias.F32)},
		},
		inputs:  []onnxValueDef{{name: "features", elemType: onnxDTFloat, dims: []int64{1, embeddingDim}}},
		outputs: []onnxValueDef{{name: "logits", elemType: onnxDTFloat, dims: []int64{1, int64(len(m.Vocab))}}},
	}
	raw := marshalONNXModel(graph)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func tensorRaw(t ModelTensor) ([]byte, error) {
	switch t.DType {
	case "F32":
		return f32Raw(t.F32), nil
	default:
		return nil, fmt.Errorf("unsupported tensor dtype %s", t.DType)
	}
}

func f32Raw(values []float32) []byte {
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return raw
}

func mustJSONString(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		return "null"
	}
	return string(data)
}

const (
	pbWireVarint      = 0
	pbWireLengthDelim = 2
	pbWireFixed32     = 5
)

const (
	onnxDTFloat = 1
)

type onnxNodeDef struct {
	name    string
	opType  string
	inputs  []string
	outputs []string
}

type onnxTensorDef struct {
	name     string
	dims     []int64
	dataType int64
	rawData  []byte
}

type onnxValueDef struct {
	name     string
	elemType int64
	dims     []int64
}

type onnxGraphDef struct {
	name         string
	nodes        []onnxNodeDef
	initializers []onnxTensorDef
	inputs       []onnxValueDef
	outputs      []onnxValueDef
}

func marshalONNXModel(graph onnxGraphDef) []byte {
	var msg []byte
	msg = pbAppendVarintField(msg, 1, 9)
	msg = pbAppendBytesField(msg, 2, []byte("stella-v-go"))
	msg = pbAppendBytesField(msg, 7, marshalONNXGraph(graph))
	msg = pbAppendBytesField(msg, 8, marshalOpset("", 13))
	return msg
}

func marshalONNXGraph(g onnxGraphDef) []byte {
	var msg []byte
	for _, node := range g.nodes {
		msg = pbAppendBytesField(msg, 1, marshalNode(node))
	}
	msg = pbAppendBytesField(msg, 2, []byte(g.name))
	for _, init := range g.initializers {
		msg = pbAppendBytesField(msg, 5, marshalTensor(init))
	}
	for _, input := range g.inputs {
		msg = pbAppendBytesField(msg, 11, marshalValueInfo(input))
	}
	for _, output := range g.outputs {
		msg = pbAppendBytesField(msg, 12, marshalValueInfo(output))
	}
	return msg
}

func marshalOpset(domain string, version int64) []byte {
	var msg []byte
	if domain != "" {
		msg = pbAppendBytesField(msg, 1, []byte(domain))
	}
	msg = pbAppendVarintField(msg, 2, uint64(version))
	return msg
}

func marshalNode(node onnxNodeDef) []byte {
	var msg []byte
	for _, input := range node.inputs {
		msg = pbAppendBytesField(msg, 1, []byte(input))
	}
	for _, output := range node.outputs {
		msg = pbAppendBytesField(msg, 2, []byte(output))
	}
	msg = pbAppendBytesField(msg, 3, []byte(node.name))
	msg = pbAppendBytesField(msg, 4, []byte(node.opType))
	return msg
}

func marshalTensor(t onnxTensorDef) []byte {
	var msg []byte
	for _, dim := range t.dims {
		msg = pbAppendVarintField(msg, 1, uint64(dim))
	}
	msg = pbAppendVarintField(msg, 2, uint64(t.dataType))
	msg = pbAppendBytesField(msg, 8, []byte(t.name))
	if len(t.rawData) > 0 {
		msg = pbAppendBytesField(msg, 9, t.rawData)
	}
	return msg
}

func marshalValueInfo(v onnxValueDef) []byte {
	var msg []byte
	msg = pbAppendBytesField(msg, 1, []byte(v.name))
	msg = pbAppendBytesField(msg, 2, marshalTypeProto(v.elemType, v.dims))
	return msg
}

func marshalTypeProto(elemType int64, dims []int64) []byte {
	var tensor []byte
	tensor = pbAppendVarintField(tensor, 1, uint64(elemType))
	tensor = pbAppendBytesField(tensor, 2, marshalTensorShape(dims))
	var msg []byte
	msg = pbAppendBytesField(msg, 1, tensor)
	return msg
}

func marshalTensorShape(dims []int64) []byte {
	var msg []byte
	for _, dim := range dims {
		var d []byte
		d = pbAppendVarintField(d, 1, uint64(dim))
		msg = pbAppendBytesField(msg, 1, d)
	}
	return msg
}

func pbAppendVarintField(dst []byte, field int, value uint64) []byte {
	dst = pbAppendKey(dst, field, pbWireVarint)
	return pbAppendVarint(dst, value)
}

func pbAppendBytesField(dst []byte, field int, value []byte) []byte {
	dst = pbAppendKey(dst, field, pbWireLengthDelim)
	dst = pbAppendVarint(dst, uint64(len(value)))
	return append(dst, value...)
}

func pbAppendKey(dst []byte, field int, wire int) []byte {
	return pbAppendVarint(dst, uint64(uint32(field<<3|wire)))
}

func pbAppendVarint(dst []byte, value uint64) []byte {
	for value >= 0x80 {
		dst = append(dst, byte(value)|0x80)
		value >>= 7
	}
	return append(dst, byte(value))
}
