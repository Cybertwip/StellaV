#include "accelerator/accelerator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace power::inference::accelerator {
namespace {

// Round half away from zero, matching the Go and reference implementations so
// every code is identical across languages.
long round_to_long(float value)
{
    constexpr float kMax = static_cast<float>(std::numeric_limits<long>::max());
    constexpr float kMin = static_cast<float>(std::numeric_limits<long>::min());
    if (value >= kMax) return std::numeric_limits<long>::max();
    if (value <= kMin) return std::numeric_limits<long>::min();
    return static_cast<long>(value + (value >= 0.0f ? 0.5f : -0.5f));
}

} // namespace

float fp16_to_float(std::uint16_t value)
{
    const std::uint32_t sign = (static_cast<std::uint32_t>(value) & 0x8000u) << 16;
    const std::uint32_t exp = (static_cast<std::uint32_t>(value) >> 10) & 0x1fu;
    std::uint32_t mant = static_cast<std::uint32_t>(value) & 0x03ffu;
    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03ffu;
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t float_to_fp16(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant = bits & 0x007fffffu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant = (mant | 0x00800000u) >> static_cast<std::uint32_t>(1 - exp);
        return static_cast<std::uint16_t>(sign | ((mant + 0x00001000u) >> 13));
    }
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | ((mant + 0x00001000u) >> 13));
}

void convert_fp16_to_fp32(const std::uint16_t* input, float* output, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) output[i] = fp16_to_float(input[i]);
}

void convert_fp32_to_fp16(const float* input, std::uint16_t* output, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) output[i] = float_to_fp16(input[i]);
}

std::uint8_t quantize_u8(float value, float scale, std::uint8_t zero_point)
{
    if (!(scale > 0.0f) || !std::isfinite(scale)) return zero_point;
    const long rounded = round_to_long(value / scale) + static_cast<long>(zero_point);
    return static_cast<std::uint8_t>(std::clamp<long>(rounded, 0, 255));
}

float dequantize_u8(std::uint8_t value, float scale, std::uint8_t zero_point)
{
    return (static_cast<int>(value) - static_cast<int>(zero_point)) * scale;
}

std::vector<std::uint8_t> pack_low_bits(std::span<const std::uint8_t> codes, std::uint32_t bit_width)
{
    if (bit_width == 0 || bit_width > 8) throw std::invalid_argument("pack_low_bits: bit width must be 1..8");
    const std::uint32_t mask = (1u << bit_width) - 1u;
    const std::size_t bit_count = codes.size() * bit_width;
    std::vector<std::uint8_t> packed((bit_count + 7u) / 8u, 0);
    std::size_t bit_pos = 0;
    for (std::uint8_t code : codes) {
        std::uint32_t value = static_cast<std::uint32_t>(code) & mask;
        for (std::uint32_t b = 0; b < bit_width; ++b) {
            if (value & (1u << b)) packed[bit_pos / 8u] |= static_cast<std::uint8_t>(1u << (bit_pos % 8u));
            ++bit_pos;
        }
    }
    return packed;
}

std::vector<std::uint8_t> unpack_low_bits(std::span<const std::uint8_t> packed, std::size_t value_count,
                                          std::uint32_t bit_width)
{
    if (bit_width == 0 || bit_width > 8) throw std::invalid_argument("unpack_low_bits: bit width must be 1..8");
    std::vector<std::uint8_t> codes(value_count, 0);
    std::size_t bit_pos = 0;
    for (std::size_t i = 0; i < value_count; ++i) {
        std::uint32_t value = 0;
        for (std::uint32_t b = 0; b < bit_width; ++b) {
            if ((bit_pos / 8u) < packed.size() && (packed[bit_pos / 8u] & (1u << (bit_pos % 8u)))) {
                value |= 1u << b;
            }
            ++bit_pos;
        }
        codes[i] = static_cast<std::uint8_t>(value);
    }
    return codes;
}

QuantizedTensor turbo_quantize(std::span<const float> values, std::span<const std::uint64_t> shape,
                               std::uint32_t bit_width, std::uint32_t block_size)
{
    if (bit_width < 2 || bit_width > 8) throw std::invalid_argument("turbo_quantize: bit width must be 2..8");
    if (block_size == 0) throw std::invalid_argument("turbo_quantize: block size must be positive");

    QuantizedTensor result;
    result.shape.assign(shape.begin(), shape.end());
    result.bit_width = bit_width;
    result.block_size = block_size;

    const std::uint32_t max_code = (1u << bit_width) - 1u;
    std::vector<std::uint8_t> codes(values.size(), 0);
    result.scales.reserve((values.size() + block_size - 1u) / block_size);

    for (std::size_t begin = 0; begin < values.size(); begin += block_size) {
        const std::size_t end = std::min<std::size_t>(values.size(), begin + block_size);
        float max_abs = 0.0f;
        for (std::size_t i = begin; i < end; ++i) {
            if (!std::isfinite(values[i])) throw std::invalid_argument("turbo_quantize: non-finite input value");
            max_abs = std::max(max_abs, std::fabs(values[i]));
        }
        const float scale = max_abs > 0.0f ? max_abs / static_cast<float>(max_code / 2u) : 1.0f;
        result.scales.push_back(scale);
        for (std::size_t i = begin; i < end; ++i) {
            const long centered = round_to_long(values[i] / scale) + static_cast<long>(max_code / 2u);
            codes[i] = static_cast<std::uint8_t>(std::clamp<long>(centered, 0, max_code));
        }
    }
    result.packed = pack_low_bits(codes, bit_width);
    return result;
}

std::vector<float> turbo_dequantize(const QuantizedTensor& tensor)
{
    std::size_t value_count = 1;
    for (std::uint64_t dim : tensor.shape) value_count *= static_cast<std::size_t>(dim);
    const std::vector<std::uint8_t> codes = unpack_low_bits(tensor.packed, value_count, tensor.bit_width);
    const std::uint32_t max_code = (1u << tensor.bit_width) - 1u;
    std::vector<float> values(value_count, 0.0f);
    for (std::size_t i = 0; i < value_count; ++i) {
        const std::size_t block = i / tensor.block_size;
        const float scale = block < tensor.scales.size() ? tensor.scales[block] : 1.0f;
        values[i] = (static_cast<int>(codes[i]) - static_cast<int>(max_code / 2u)) * scale;
    }
    return values;
}

bool native_matmul_f32_available() { return true; }

bool native_matmul_f32(const float* lhs, const float* rhs, float* output, std::size_t rows,
                       std::size_t inner, std::size_t output_cols)
{
    if (!lhs || !rhs || !output || rows == 0 || inner == 0 || output_cols == 0) return false;
    // Portable cache-friendly scalar GEMM (ikj order). No BLAS dependency.
    for (std::size_t i = 0; i < rows; ++i) {
        float* out_row = output + i * output_cols;
        for (std::size_t c = 0; c < output_cols; ++c) out_row[c] = 0.0f;
        const float* lhs_row = lhs + i * inner;
        for (std::size_t k = 0; k < inner; ++k) {
            const float a = lhs_row[k];
            const float* rhs_row = rhs + k * output_cols;
            for (std::size_t c = 0; c < output_cols; ++c) out_row[c] += a * rhs_row[c];
        }
    }
    return true;
}

} // namespace power::inference::accelerator
