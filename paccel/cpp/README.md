# libpaccel C++ accelerator

The minimal native quantization core, extracted from
`External/inference/accelerator`. It is the **only** C++ the PAccel format
requires. It has zero third-party dependencies — no onnxruntime, protobuf,
Eigen, abseil, or BLAS — only the C++20 standard library.

## API (`power::inference::accelerator`)

| Function | Purpose |
| --- | --- |
| `turbo_quantize` / `turbo_dequantize` | block-wise symmetric quantization |
| `pack_low_bits` / `unpack_low_bits`   | LSB-first low-bit (1–8) packing |
| `fp16_to_float` / `float_to_fp16`     | IEEE-754 half codec for stored scales |
| `quantize_u8` / `dequantize_u8`       | per-element uint8 affine quant |
| `native_matmul_f32`                   | portable scalar GEMM (no BLAS) |

The algebra is bit-for-bit identical to the Go `libpaccel` package, so a tensor
quantized in one language dequantizes in the other.

## Build

```bash
cmake -B build && cmake --build build && ctest --test-dir build
# or directly:
g++ -std=c++20 -O2 -Iinclude src/accelerator.cpp test/selftest.cpp -o selftest && ./selftest
```

The `power::inference::accelerator` CMake alias is drop-in compatible with the
original compressor target, so `External/inference/compressor` can link this
extraction unchanged.
