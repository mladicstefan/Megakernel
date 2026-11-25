#pragma once
#include "numeric.hpp"

#include <concepts>
#include <cstdint>

// How the megakernel multiplies one BF16 row by a vector. A row kernel is a type, so a
// configuration can name another one without a change to the megakernel. nvcc compiles
// this header: C++20 only.
namespace mk {
// K::dot<Cols>(row, xs) is the dot product of the Cols weights at row with xs, computed by
// one whole warp. Every lane gets the result. Cols is a multiple of K::step.
template <class K>
concept RowKernel = requires(const std::uint16_t* row, const float* xs) {
    { K::step } -> std::convertible_to<unsigned>;
    { K::template dot<256>(row, xs) } -> std::same_as<float>;
};

#ifdef __CUDACC__
// The sum of v over the warp's lanes. Every lane gets it.
__device__ inline float warp_sum(float v) {
    for (unsigned lanes = 16; lanes > 0; lanes /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, lanes);
    }
    return v;
}
#endif

// Each lane takes 8 consecutive weights (16 bytes) at a time, so the warp reads 512
// contiguous bytes per step.
struct RowDot8 {
    static constexpr unsigned step = 256; // weights the warp reads per step
#ifdef __CUDACC__
    template <unsigned Cols>
    __device__ static float dot(const std::uint16_t* row, const float* xs) {
        static_assert(Cols % step == 0);
        float sum = 0;
        for (unsigned c = (threadIdx.x % 32) * 8; c < Cols; c += step) {
            // Aligned: tensors start on 32 bytes and a row is a multiple of 16 bytes.
            const uint4  w = *reinterpret_cast<const uint4*>(row + c);
            const float* x = xs + c;
            sum += bf16_to_float(static_cast<std::uint16_t>(w.x)) * x[0] +
                   bf16_to_float(static_cast<std::uint16_t>(w.x >> 16)) * x[1] +
                   bf16_to_float(static_cast<std::uint16_t>(w.y)) * x[2] +
                   bf16_to_float(static_cast<std::uint16_t>(w.y >> 16)) * x[3] +
                   bf16_to_float(static_cast<std::uint16_t>(w.z)) * x[4] +
                   bf16_to_float(static_cast<std::uint16_t>(w.z >> 16)) * x[5] +
                   bf16_to_float(static_cast<std::uint16_t>(w.w)) * x[6] +
                   bf16_to_float(static_cast<std::uint16_t>(w.w >> 16)) * x[7];
        }
        return warp_sum(sum);
    }
#else
    // Host code can name the kernel, so the type is a RowKernel there too, but only device
    // code can call it.
    template <unsigned Cols>
    static float dot(const std::uint16_t* row, const float* xs);
#endif
};
} // namespace mk
