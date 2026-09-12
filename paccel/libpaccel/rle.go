package libpaccel

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

// RLEEncode applies element-wise run-length encoding to a raw payload. The
// element width (e.g. 8 for int64) defines the unit; identical consecutive
// elements collapse into a (count, element) run. The stream starts with a small
// header: u32 elementSize, u64 elementCount, u64 runCount. Encoding is abandoned
// (returns nil) if it fails to beat the raw size by a safety margin, so the
// caller stores the original bytes instead.
func RLEEncode(raw []byte, elementSize uint32) []byte {
	if elementSize == 0 || len(raw) == 0 || len(raw)%int(elementSize) != 0 {
		return nil
	}
	encoded := make([]byte, 0, minInt(len(raw), 1<<20))
	encoded = binary.LittleEndian.AppendUint32(encoded, elementSize)
	encoded = binary.LittleEndian.AppendUint64(encoded, uint64(len(raw)/int(elementSize)))
	runCountPos := len(encoded)
	encoded = binary.LittleEndian.AppendUint64(encoded, 0)

	runCount := uint64(0)
	elements := len(raw) / int(elementSize)
	es := int(elementSize)
	for index := 0; index < elements; {
		runBegin := index
		index++
		for index < elements &&
			bytes.Equal(raw[runBegin*es:(runBegin+1)*es], raw[index*es:(index+1)*es]) {
			index++
		}
		encoded = binary.LittleEndian.AppendUint64(encoded, uint64(index-runBegin))
		encoded = append(encoded, raw[runBegin*es:(runBegin+1)*es]...)
		runCount++
		if len(encoded)+96 >= len(raw) {
			return nil // not worth it
		}
	}
	binary.LittleEndian.PutUint64(encoded[runCountPos:], runCount)
	return encoded
}

// RLEDecode reconstructs the original raw payload from an RLEEncode stream.
func RLEDecode(encoded []byte) ([]byte, error) {
	if len(encoded) < 20 {
		return nil, fmt.Errorf("libpaccel: rle stream too short")
	}
	elementSize := int(binary.LittleEndian.Uint32(encoded[0:4]))
	elementCount := binary.LittleEndian.Uint64(encoded[4:12])
	runCount := binary.LittleEndian.Uint64(encoded[12:20])
	if elementSize == 0 {
		return nil, fmt.Errorf("libpaccel: rle zero element size")
	}
	out := make([]byte, 0, int(elementCount)*elementSize)
	pos := 20
	for r := uint64(0); r < runCount; r++ {
		if pos+8+elementSize > len(encoded) {
			return nil, fmt.Errorf("libpaccel: truncated rle run")
		}
		count := binary.LittleEndian.Uint64(encoded[pos : pos+8])
		pos += 8
		element := encoded[pos : pos+elementSize]
		pos += elementSize
		for c := uint64(0); c < count; c++ {
			out = append(out, element...)
		}
	}
	return out, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
