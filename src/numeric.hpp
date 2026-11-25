#pragma once
#include "host_device.hpp"

#include <cstdint>
#include <cstring>

// BF16 on the host and on the device. nvcc compiles this header: C++20 only.
namespace mk {
// BF16 is the top 16 bits of an F32, so widening is exact.
MK_HOST_DEVICE inline float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = std::uint32_t{bits} << 16;
    float               f;
    std::memcpy(&f, &wide, sizeof f);
    return f;
}

// f rounded to the nearest BF16 value, ties to even: ggml's ggml_compute_fp32_to_bf16,
// widened again. llama.cpp's CPU backend does this to a vector before a BF16 dot product.
MK_HOST_DEVICE inline float round_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        u = ((u >> 16) | 64u) << 16; // NaN: made quiet, as ggml does
    } else {
        u = ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16) << 16;
    }
    std::memcpy(&f, &u, sizeof f);
    return f;
}
} // namespace mk
