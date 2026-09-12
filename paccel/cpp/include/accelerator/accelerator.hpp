#pragma once

// libpaccel C++ accelerator — the minimal, dependency-free quantization core
// extracted from the PowerEngine inference accelerator. It has NO dependency on
// onnxruntime, protobuf, Eigen, abseil, or any BLAS: just the C++20 standard
// library. It is the exact algebra the Go libpaccel mirrors, provided in C++ for
// engines that want a native hot path. Link this alone to read/write the
// quantized payloads of a .paccel container.

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace power::inference::accelerator {

// QuantizedTensor holds the result of block-wise quantization: a shape, the bit
// width, the per-block scales, and the packed low-bit code stream.
struct QuantizedTensor {
    std::vector<std::uint64_t> shape;
    std::uint32_t bit_width = 0;
    std::uint32_t block_size = 0;
    std::vector<float> scales;
    std::vector<std::uint8_t> packed;
};

// IEEE-754 half <-> single precision (used for stored block scales, v4+).
float fp16_to_float(std::uint16_t value);
std::uint16_t float_to_fp16(float value);
void convert_fp16_to_fp32(const std::uint16_t* input, float* output, std::size_t count);
void convert_fp32_to_fp16(const float* input, std::uint16_t* output, std::size_t count);

// Per-element symmetric uint8 affine quant with an explicit zero point.
std::uint8_t quantize_u8(float value, float scale, std::uint8_t zero_point);
float dequantize_u8(std::uint8_t value, float scale, std::uint8_t zero_point);

// Low-bit (1..8) LSB-first packing of one code per element.
std::vector<std::uint8_t> pack_low_bits(std::span<const std::uint8_t> codes, std::uint32_t bit_width);
std::vector<std::uint8_t> unpack_low_bits(std::span<const std::uint8_t> packed, std::size_t value_count,
                                          std::uint32_t bit_width);

// Block-wise symmetric mid-rise quantization and its inverse. bit_width is 2..8.
QuantizedTensor turbo_quantize(std::span<const float> values, std::span<const std::uint64_t> shape,
                               std::uint32_t bit_width, std::uint32_t block_size = 32);
std::vector<float> turbo_dequantize(const QuantizedTensor& tensor);

// Portable row-major single-precision GEMM (output = lhs * rhs). Always
// available — a scalar fallback, no BLAS required. Returns false on bad shapes.
bool native_matmul_f32_available();
bool native_matmul_f32(const float* lhs, const float* rhs, float* output, std::size_t rows,
                       std::size_t inner, std::size_t output_cols);

} // namespace power::inference::accelerator
