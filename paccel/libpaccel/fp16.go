// Package libpaccel is a dependency-free (no onnxruntime, no protobuf) Go
// implementation of the PAccel model-weight container format and its
// block-wise quantization algebra.
//
// This file implements the IEEE-754 half-precision (binary16) <-> single
// precision (binary32) conversions used to store per-block quantization
// scales in PAccel package versions >= 4. The bit manipulation is a direct,
// clean-room port of the reference accelerator so that scales decode
// bit-for-bit identically to the C++ engine.
package libpaccel

import "math"

// Fp16ToFloat32 expands an IEEE-754 binary16 value (carried in a uint16) to
// binary32. Subnormals, infinities and NaNs are all handled so that any scale
// written by the compressor round-trips exactly.
func Fp16ToFloat32(value uint16) float32 {
	sign := (uint32(value) & 0x8000) << 16
	exp := (uint32(value) >> 10) & 0x1f
	mant := uint32(value) & 0x03ff
	var bits uint32
	switch {
	case exp == 0 && mant == 0:
		// Signed zero.
		bits = sign
	case exp == 0:
		// Subnormal: normalize the mantissa, tracking the implied exponent shift.
		shift := uint32(0)
		for mant&0x0400 == 0 {
			mant <<= 1
			shift++
		}
		mant &= 0x03ff
		bits = sign | ((127 - 15 - shift) << 23) | (mant << 13)
	case exp == 0x1f:
		// Inf / NaN.
		bits = sign | 0x7f800000 | (mant << 13)
	default:
		// Normalized: rebias exponent from 15 to 127, left-justify the mantissa.
		bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(bits)
}

// Float32ToFp16 narrows a binary32 value to IEEE-754 binary16 with
// round-to-nearest-even on the truncated mantissa bits (the +0x1000 carry),
// matching the reference encoder.
func Float32ToFp16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := (bits >> 16) & 0x8000
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x007fffff
	if exp <= 0 {
		if exp < -10 {
			// Magnitude rounds to signed zero.
			return uint16(sign)
		}
		// Subnormal half: restore the implied leading 1 then shift into place.
		mant = (mant | 0x00800000) >> uint32(1-exp)
		return uint16(sign | ((mant + 0x00001000) >> 13))
	}
	if exp >= 31 {
		// Overflow saturates to signed infinity.
		return uint16(sign | 0x7c00)
	}
	return uint16(sign | (uint32(exp) << 10) | ((mant + 0x00001000) >> 13))
}

// BFloat16ToFloat32 expands a truncated-mantissa bfloat16 (the upper 16 bits of
// a binary32) back to binary32. Used when ingesting BF16 source weights.
func BFloat16ToFloat32(value uint16) float32 {
	return math.Float32frombits(uint32(value) << 16)
}

// float32FromBits decodes a little-endian fp32 scale used by legacy (pre-v4)
// packages that stored block scales at full precision.
func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}
