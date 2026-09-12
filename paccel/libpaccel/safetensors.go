package libpaccel

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type safetensorsMeta struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

func safetensorsDTypeToWire(dtype string) (uint32, bool) {
	switch dtype {
	case "BOOL":
		return WireBool, true
	case "U8":
		return WireUint8, true
	case "I8":
		return WireInt8, true
	case "U16":
		return WireUint16, true
	case "I16":
		return WireInt16, true
	case "I32":
		return WireInt32, true
	case "I64":
		return WireInt64, true
	case "F16":
		return WireFloat16, true
	case "BF16":
		return WireBFloat16, true
	case "F32":
		return WireFloat, true
	case "F64":
		return WireDouble, true
	case "U32":
		return WireUint32, true
	case "U64":
		return WireUint64, true
	default:
		return 0, false
	}
}

func safetensorsElementBytes(dtype string) uint32 {
	switch dtype {
	case "BOOL", "U8", "I8":
		return 1
	case "F16", "BF16", "U16", "I16":
		return 2
	case "F32", "I32", "U32":
		return 4
	case "F64", "I64", "U64":
		return 8
	default:
		return 0
	}
}

// CollectSafetensorsFiles returns every .safetensors file under input (a file or
// a directory), sorted deterministically and skipping dot-directories.
func CollectSafetensorsFiles(input string) []string {
	var out []string
	if info, err := os.Stat(input); err == nil && info.IsDir() {
		_ = filepath.WalkDir(input, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				if path != input && strings.HasPrefix(d.Name(), ".") {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.EqualFold(filepath.Ext(path), ".safetensors") {
				out = append(out, path)
			}
			return nil
		})
	} else if strings.EqualFold(filepath.Ext(input), ".safetensors") {
		out = append(out, input)
	}
	sort.Strings(out)
	return out
}

// CompressSafetensors ingests one or more .safetensors shards and returns a
// fully-built Package. Float tensors are block-quantized (with mixed-precision
// promotion for sensitive tensors); non-float tensors are stored raw, RLE-encoded
// when that shrinks them. This is the dependency-free ingest path the frontend
// uses to turn a downloaded HF checkpoint into a .paccel container.
func CompressSafetensors(files []string, bits, blockSize uint32) (*Package, error) {
	if bits < 2 || bits > 8 {
		return nil, fmt.Errorf("libpaccel: bits must be 2..8")
	}
	if blockSize == 0 {
		return nil, fmt.Errorf("libpaccel: block size must be positive")
	}
	pkg := &Package{BitWidth: bits, BlockSize: blockSize}
	seen := map[string]struct{}{}

	for _, path := range files {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		info, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, err
		}
		pkg.ModelSize += uint64(info.Size())
		var hdrLen [8]byte
		if _, err := io.ReadFull(f, hdrLen[:]); err != nil {
			f.Close()
			return nil, err
		}
		headerLen := binary.LittleEndian.Uint64(hdrLen[:])
		header := make([]byte, headerLen)
		if _, err := io.ReadFull(f, header); err != nil {
			f.Close()
			return nil, err
		}
		var root map[string]json.RawMessage
		if err := json.Unmarshal(header, &root); err != nil {
			f.Close()
			return nil, fmt.Errorf("safetensors header: %w", err)
		}
		dataBase := int64(8 + headerLen)

		names := make([]string, 0, len(root))
		for name := range root {
			if name != "__metadata__" {
				names = append(names, name)
			}
		}
		sort.Strings(names)

		for _, name := range names {
			var meta safetensorsMeta
			if err := json.Unmarshal(root[name], &meta); err != nil {
				continue
			}
			if meta.DType == "" || len(meta.DataOffsets) != 2 {
				continue
			}
			wire, ok := safetensorsDTypeToWire(meta.DType)
			if !ok {
				continue
			}
			begin, end := meta.DataOffsets[0], meta.DataOffsets[1]
			if end < begin {
				continue
			}
			raw := make([]byte, end-begin)
			if _, err := f.ReadAt(raw, dataBase+int64(begin)); err != nil {
				f.Close()
				return nil, err
			}
			recName := name
			if _, dup := seen[recName]; dup {
				recName = filepath.Base(path) + ":" + name
			}
			seen[recName] = struct{}{}

			rec := Record{
				Name:               recName,
				SourceDataType:     wire,
				Shape:              append([]uint64(nil), meta.Shape...),
				SourcePayloadBytes: end - begin,
			}

			values := DecodeRawTensor(raw, wire)
			if len(values) > 0 && compressWorthwhile(meta.Shape, meta.DType, bits, blockSize) {
				shape := NormalizedShape(meta.Shape, len(values))
				tensorBits := ResolveMixedBitWidth(recName, shape, bits)
				q, err := TurboQuantize(values, shape, tensorBits, blockSize, recName, false)
				if err != nil {
					f.Close()
					return nil, err
				}
				rec.Encoding = EncodingCompressed
				rec.Tensor = q
			} else {
				rec.Encoding = EncodingRaw
				if enc := RLEEncode(raw, safetensorsElementBytes(meta.DType)); enc != nil {
					rec.Encoding = EncodingRawRLE
					rec.RawPayload = enc
				} else {
					rec.RawPayload = raw
				}
			}
			pkg.Records = append(pkg.Records, rec)
		}
		f.Close()
	}
	if len(pkg.Records) == 0 {
		return nil, fmt.Errorf("libpaccel: no supported tensors found")
	}
	return pkg, nil
}

// compressWorthwhile keeps tiny constants raw but compresses real float weights,
// requiring the quantized size (plus per-record overhead) to beat the raw size.
func compressWorthwhile(shape []uint64, dtype string, bits, blockSize uint32) bool {
	if dtype != "F32" && dtype != "F16" && dtype != "BF16" {
		return false
	}
	valueCount := ShapeValueCount(shape)
	if valueCount == 0 {
		return false
	}
	rawBytes := valueCount * uint64(safetensorsElementBytes(dtype))
	effBits := ResolveMixedBitWidth("", shape, bits)
	quantBytes := EstimatedQuantizedBytes(valueCount, effBits, blockSize)
	return rawBytes > 0 && quantBytes > 0 && quantBytes+RecordOverheadBytes < rawBytes
}
