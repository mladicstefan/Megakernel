#pragma once
#include "host_device.hpp"
#include "numeric.hpp"

#include <concepts>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

// How the KV cache stores one value. A format is a type: Bits is what a cache slot holds,
// store() rounds a float into it and load() widens it again. Kernels are templates over the
// format, and a configuration names one as its Kv. nvcc compiles this header: C++20.
namespace mk {
template <class F>
concept KvFormat = requires(float value, typename F::Bits bits) {
    { F::store(value) } -> std::same_as<typename F::Bits>;
    { F::load(bits) } -> std::same_as<float>;
};

struct KvF32 {
    using Bits = float;
    MK_HOST_DEVICE static Bits store(float value) {
        return value;
    }
    MK_HOST_DEVICE static float load(Bits bits) {
        return bits;
    }
};

struct KvBF16 {
    using Bits = std::uint16_t;
    MK_HOST_DEVICE static Bits store(float value) {
        return float_to_bf16(value);
    }
    MK_HOST_DEVICE static float load(Bits bits) {
        return bf16_to_float(bits);
    }
};

// IEEE half precision, through the GPU's own conversions.
struct KvF16 {
    using Bits = std::uint16_t;
#ifdef __CUDACC__
    __device__ static Bits store(float value) {
        return __half_as_ushort(__float2half(value));
    }
    __device__ static float load(Bits bits) {
        return __half2float(__ushort_as_half(bits));
    }
#else
    // Host code can name the conversions, so the type is a KvFormat there too, but only
    // device code can call them.
    static Bits  store(float value);
    static float load(Bits bits);
#endif
};
} // namespace mk
