#pragma once
#include "base/numeric.hpp"
#include "config/platform.hpp"

#include <concepts>
#include <cstdint>

// How the megakernel multiplies one BF16 row by a vector. A row kernel is a type, so a
// configuration can name another one without a change to the megakernel. nvcc compiles
// this header: C++20 only.
namespace mk {
// K::dot<Cols>(row, xs) is the dot product of the Cols weights at row with xs, computed by
// one whole warp. Every lane gets the result. Cols is a multiple of K::step, and row starts
// on a multiple of K::alignment bytes.
template <class K>
concept RowKernel = requires(const std::uint16_t* row, const float* xs) {
    { K::step } -> std::convertible_to<unsigned>;
    { K::alignment } -> std::convertible_to<unsigned>;
    { K::template dot<K::step>(row, xs) } -> std::same_as<float>;
};

#ifdef __CUDACC__
// The sum of v over the warp's lanes. Every lane gets it.
__device__ inline float warp_sum(float v) {
    for (unsigned lanes = warp_size / 2; lanes > 0; lanes /= 2) {
        v += __shfl_xor_sync(full_warp_mask, v, lanes);
    }
    return v;
}
#endif

// Each lane takes 8 consecutive weights (16 bytes) at a time, so the warp reads 512
// contiguous bytes per step.
struct RowDot8 {
    static constexpr unsigned lane_weights = 8; // a lane loads them as one uint4
    static constexpr unsigned step         = warp_size * lane_weights; // weights per warp step
    static constexpr unsigned alignment    = lane_weights * sizeof(std::uint16_t); // bytes
#ifdef __CUDACC__
    template <unsigned Cols>
    __device__ static float dot(const std::uint16_t* row, const float* xs) {
        static_assert(Cols % step == 0);
        static_assert(lane_weights == 8 && lane_weights * sizeof(std::uint16_t) == sizeof(uint4),
                      "the body below is unrolled for the 8 weights of one uint4");
        float sum = 0;
        for (unsigned c = (threadIdx.x % warp_size) * lane_weights; c < Cols; c += step) {
            // Aligned: row starts on `alignment` bytes, and c is a multiple of lane_weights.
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
