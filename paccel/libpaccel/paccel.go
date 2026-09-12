package libpaccel

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Magic is the 8-byte container signature. The trailing "1" is a container
// generation marker, not the package version (which follows as a u32).
const Magic = "PACONNX1"

// Package version history. Versions are forward-compatible at the reader: a
// loader accepts [RequiredPackageVersion, LatestPackageVersion].
const (
	FP16ScaleVersion       = 4 // block scales stored as fp16 from here on
	RequiredPackageVersion = 5 // self-contained zero-byte raw tensors
	LatestPackageVersion   = 6 // adds the AVX-friendly tiled int4 layout
)

// Per-record payload encodings.
const (
	EncodingRaw                = 0
	EncodingCompressed         = 1
	EncodingRawRLE             = 2
	EncodingCompressedTileInt4 = 3
)

// ONNX/safetensors source element types, retained so a tensor can be rebuilt in
// its original precision when an encoding leaves it uncompressed.
const (
	WireFloat    = 1
	WireUint8    = 2
	WireInt8     = 3
	WireUint16   = 4
	WireInt16    = 5
	WireInt32    = 6
	WireInt64    = 7
	WireBool     = 9
	WireFloat16  = 10
	WireDouble   = 11
	WireUint32   = 12
	WireUint64   = 13
	WireBFloat16 = 16
)

// Record is a single weight tensor inside a package.
type Record struct {
	Name               string
	SourceDataType     uint32
	Encoding           uint32
	Shape              []uint64
	SourcePayloadBytes uint64

	// Tensor is populated for EncodingCompressed / EncodingCompressedTileInt4.
	Tensor QuantizedTensor
	// RawPayload is populated for EncodingRaw / EncodingRawRLE.
	RawPayload []byte
}

// Compressed reports whether the record carries a quantized tensor.
func (r Record) Compressed() bool {
	return r.Encoding == EncodingCompressed || r.Encoding == EncodingCompressedTileInt4
}

// Package is a decoded PAccel container.
type Package struct {
	Version   uint32
	BitWidth  uint32
	BlockSize uint32
	ModelSize uint64
	Records   []Record
}

// WritePackage serializes a package to path using LatestPackageVersion and fp16
// scales. The byte layout is identical to the reference C++/Go compressors so a
// file produced here loads unmodified in the engine and vice versa.
func WritePackage(path string, pkg *Package) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	bw := bufio.NewWriterSize(f, 1<<20)

	if _, err := bw.WriteString(Magic); err != nil {
		return err
	}
	var scratch [8]byte
	u32 := func(v uint32) error {
		binary.LittleEndian.PutUint32(scratch[:4], v)
		_, e := bw.Write(scratch[:4])
		return e
	}
	u64 := func(v uint64) error {
		binary.LittleEndian.PutUint64(scratch[:8], v)
		_, e := bw.Write(scratch[:8])
		return e
	}
	if err := u32(LatestPackageVersion); err != nil {
		return err
	}
	if err := u32(pkg.BitWidth); err != nil {
		return err
	}
	if err := u32(pkg.BlockSize); err != nil {
		return err
	}
	if err := u64(pkg.ModelSize); err != nil {
		return err
	}
	if err := u32(uint32(len(pkg.Records))); err != nil {
		return err
	}

	for _, rec := range pkg.Records {
		if err := u32(uint32(len(rec.Name))); err != nil {
			return err
		}
		if _, err := bw.WriteString(rec.Name); err != nil {
			return err
		}
		if err := u32(rec.SourceDataType); err != nil {
			return err
		}
		encoding := rec.Encoding
		if rec.Compressed() && rec.Tensor.Layout == PackedLayoutOutputTileInt4 {
			encoding = EncodingCompressedTileInt4
		}
		if err := u32(encoding); err != nil {
			return err
		}
		shape := rec.Shape
		if rec.Compressed() {
			shape = rec.Tensor.Shape
		}
		if err := u32(uint32(len(shape))); err != nil {
			return err
		}
		for _, dim := range shape {
			if err := u64(dim); err != nil {
				return err
			}
		}
		if err := u64(rec.SourcePayloadBytes); err != nil {
			return err
		}
		if rec.Compressed() {
			if err := u32(rec.Tensor.BitWidth); err != nil {
				return err
			}
			if err := u32(rec.Tensor.BlockSize); err != nil {
				return err
			}
			if encoding == EncodingCompressedTileInt4 {
				if err := u32(rec.Tensor.Layout); err != nil {
					return err
				}
				if err := u32(rec.Tensor.TileOutputs); err != nil {
					return err
				}
			}
			if err := u32(uint32(len(rec.Tensor.Scales))); err != nil {
				return err
			}
			if err := u64(uint64(len(rec.Tensor.Packed))); err != nil {
				return err
			}
			for _, scale := range rec.Tensor.Scales {
				binary.LittleEndian.PutUint16(scratch[:2], Float32ToFp16(scale))
				if _, err := bw.Write(scratch[:2]); err != nil {
					return err
				}
			}
			if _, err := bw.Write(rec.Tensor.Packed); err != nil {
				return err
			}
			continue
		}
		if err := u64(uint64(len(rec.RawPayload))); err != nil {
			return err
		}
		if _, err := bw.Write(rec.RawPayload); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// ReadPackage fully decodes a package file, reconstructing each QuantizedTensor
// (codes are kept packed; call TurboDequantize to materialize floats).
func ReadPackage(path string) (*Package, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	magic := make([]byte, 8)
	if _, err := io.ReadFull(r, magic); err != nil || string(magic) != Magic {
		return nil, fmt.Errorf("paccel: bad magic")
	}
	u32 := func() (uint32, error) {
		var b [4]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(b[:]), nil
	}
	u64 := func() (uint64, error) {
		var b [8]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint64(b[:]), nil
	}

	pkg := &Package{}
	if pkg.Version, err = u32(); err != nil {
		return nil, err
	}
	if pkg.Version < RequiredPackageVersion || pkg.Version > LatestPackageVersion {
		return nil, fmt.Errorf("paccel: unsupported version %d (want %d..%d)", pkg.Version, RequiredPackageVersion, LatestPackageVersion)
	}
	scaleBytes := 4
	if pkg.Version >= FP16ScaleVersion {
		scaleBytes = 2
	}
	if pkg.BitWidth, err = u32(); err != nil {
		return nil, err
	}
	if pkg.BlockSize, err = u32(); err != nil {
		return nil, err
	}
	if pkg.ModelSize, err = u64(); err != nil {
		return nil, err
	}
	count, err := u32()
	if err != nil {
		return nil, err
	}

	for i := uint32(0); i < count; i++ {
		var rec Record
		nameLen, err := u32()
		if err != nil {
			return nil, err
		}
		nameBytes := make([]byte, nameLen)
		if _, err := io.ReadFull(r, nameBytes); err != nil {
			return nil, err
		}
		rec.Name = string(nameBytes)
		if rec.SourceDataType, err = u32(); err != nil {
			return nil, err
		}
		if rec.Encoding, err = u32(); err != nil {
			return nil, err
		}
		rank, err := u32()
		if err != nil {
			return nil, err
		}
		shape := make([]uint64, 0, rank)
		for j := uint32(0); j < rank; j++ {
			dim, err := u64()
			if err != nil {
				return nil, err
			}
			shape = append(shape, dim)
		}
		rec.Shape = shape
		if rec.SourcePayloadBytes, err = u64(); err != nil {
			return nil, err
		}

		if rec.Encoding == EncodingCompressed || rec.Encoding == EncodingCompressedTileInt4 {
			t := QuantizedTensor{Shape: shape, Layout: PackedLayoutLinear}
			if t.BitWidth, err = u32(); err != nil {
				return nil, err
			}
			if t.BlockSize, err = u32(); err != nil {
				return nil, err
			}
			if rec.Encoding == EncodingCompressedTileInt4 {
				if t.Layout, err = u32(); err != nil {
					return nil, err
				}
				if t.TileOutputs, err = u32(); err != nil {
					return nil, err
				}
			}
			scaleCount, err := u32()
			if err != nil {
				return nil, err
			}
			packedLen, err := u64()
			if err != nil {
				return nil, err
			}
			t.Scales = make([]float32, scaleCount)
			sbuf := make([]byte, int(scaleCount)*scaleBytes)
			if _, err := io.ReadFull(r, sbuf); err != nil {
				return nil, err
			}
			for s := uint32(0); s < scaleCount; s++ {
				if scaleBytes == 2 {
					t.Scales[s] = Fp16ToFloat32(binary.LittleEndian.Uint16(sbuf[int(s)*2:]))
				} else {
					t.Scales[s] = float32FromBits(binary.LittleEndian.Uint32(sbuf[int(s)*4:]))
				}
			}
			t.Packed = make([]byte, packedLen)
			if _, err := io.ReadFull(r, t.Packed); err != nil {
				return nil, err
			}
			rec.Tensor = t
		} else {
			payloadLen, err := u64()
			if err != nil {
				return nil, err
			}
			rec.RawPayload = make([]byte, payloadLen)
			if _, err := io.ReadFull(r, rec.RawPayload); err != nil {
				return nil, err
			}
		}
		pkg.Records = append(pkg.Records, rec)
	}
	return pkg, nil
}

// ReadPackageHeader peeks only the fixed header (magic, version, defaults,
// model size, tensor count) without decoding tensors — cheap for capability
// gating and snapshot validation.
func ReadPackageHeader(path string) (*Package, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	var magic [8]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil || string(magic[:]) != Magic {
		return nil, 0, fmt.Errorf("paccel: bad magic")
	}
	rd := func(n int) ([]byte, error) {
		b := make([]byte, n)
		_, e := io.ReadFull(f, b)
		return b, e
	}
	b, err := rd(4)
	if err != nil {
		return nil, 0, err
	}
	pkg := &Package{Version: binary.LittleEndian.Uint32(b)}
	if b, err = rd(4); err != nil {
		return nil, 0, err
	}
	pkg.BitWidth = binary.LittleEndian.Uint32(b)
	if b, err = rd(4); err != nil {
		return nil, 0, err
	}
	pkg.BlockSize = binary.LittleEndian.Uint32(b)
	if b, err = rd(8); err != nil {
		return nil, 0, err
	}
	pkg.ModelSize = binary.LittleEndian.Uint64(b)
	if b, err = rd(4); err != nil {
		return nil, 0, err
	}
	return pkg, int(binary.LittleEndian.Uint32(b)), nil
}
