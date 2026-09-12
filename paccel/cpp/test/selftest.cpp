// Self-test for the extracted libpaccel C++ accelerator. Verifies the same
// invariants as the Go test suite: quantize/dequantize error bound, pack/unpack
// bijection, fp16 round-trip, and a small matmul identity. Exits non-zero on
// failure so CTest/CI can gate on it.

#include "accelerator/accelerator.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace acc = power::inference::accelerator;

namespace {
int g_failures = 0;
void check(const char* name, bool ok)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++g_failures;
}

std::vector<float> synth(std::size_t n)
{
    std::vector<float> out(n);
    std::uint64_t s = 0x9e3779b97f4a7c15ull;
    for (auto& v : out) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const double u = static_cast<double>(s >> 11) / static_cast<double>(1ull << 53);
        v = static_cast<float>((u - 0.5) * 6.0);
    }
    return out;
}
} // namespace

int main()
{
    const std::vector<float> values = synth(4096);

    for (std::uint32_t bits : {2u, 4u, 8u}) {
        const acc::QuantizedTensor q =
            acc::turbo_quantize(values, std::vector<std::uint64_t>{values.size()}, bits, 32);
        const std::vector<float> deq = acc::turbo_dequantize(q);
        bool ok = deq.size() == values.size();
        for (std::size_t i = 0; ok && i < values.size(); ++i) {
            const float scale = q.scales[i / 32];
            if (std::fabs(values[i] - deq[i]) > scale * 0.5f + 1e-5f) ok = false;
        }
        char name[64];
        std::snprintf(name, sizeof(name), "quant/dequant within scale/2 (bits=%u)", bits);
        check(name, ok);
    }

    for (std::uint32_t bits : {2u, 3u, 4u, 5u, 8u}) {
        std::vector<std::uint8_t> codes(1000);
        for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<std::uint8_t>(i & ((1u << bits) - 1u));
        const auto packed = acc::pack_low_bits(codes, bits);
        const auto back = acc::unpack_low_bits(packed, codes.size(), bits);
        bool ok = back.size() == codes.size();
        for (std::size_t i = 0; ok && i < codes.size(); ++i) {
            if (back[i] != codes[i]) ok = false;
        }
        char name[48];
        std::snprintf(name, sizeof(name), "pack/unpack identity (bits=%u)", bits);
        check(name, ok);
    }

    {
        bool ok = true;
        for (float v : {0.0f, 1.0f, -2.25f, 3.1415927f, 1e-4f, 65504.0f}) {
            const float r = acc::fp16_to_float(acc::float_to_fp16(v));
            const float tol = std::fabs(v) * 0.01f + 1e-6f;
            if (std::fabs(r - v) > tol) ok = false;
        }
        check("fp16 round trip", ok);
    }

    {
        // [2x3] * [3x2] identity-ish check against a hand-computed result.
        const float lhs[6] = {1, 2, 3, 4, 5, 6};
        const float rhs[6] = {1, 0, 0, 1, 1, 1};
        float out[4] = {0, 0, 0, 0};
        const bool ran = acc::native_matmul_f32(lhs, rhs, out, 2, 3, 2);
        // row0 = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        // row1 = [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
        const bool ok = ran && out[0] == 4 && out[1] == 5 && out[2] == 10 && out[3] == 11;
        check("portable matmul identity", ok);
    }

    std::printf("\n%s\n", g_failures == 0 ? "ALL VERIFIED" : "VERIFICATION FAILED");
    return g_failures == 0 ? 0 : 1;
}
