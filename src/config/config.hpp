#pragma once
#include "config/gpu.hpp"
#include "config/kv_format.hpp"
#include "config/model_spec.hpp"
#include "config/platform.hpp"
#include "config/row_kernel.hpp"

#include <algorithm>
#include <concepts>
#include <cstdint>

// What a build is compiled for. A configuration is a type, and the decoders and the
// megakernel are templates over it: nothing here is decided at run time. nvcc compiles this
// header: C++20 only.
#ifndef MK_N_CTX
#error "MK_N_CTX is not defined: CMake's MK_N_CTX caps the positions of the KV cache"
#endif

namespace mk {
// What a decoder needs:
// - Model, the model it decodes
// - Kv, the KV cache's format
// - n_ctx, the positions the KV cache holds
// - llama_cpu_rounding, whether the ops also round where llama.cpp's CPU backend rounds
// - reference_threads, the threads in a block of the reference pass's kernels
template <class C>
concept DecoderConfig = ModelSpec<typename C::Model> && KvFormat<typename C::Kv> && requires {
    { C::n_ctx } -> std::convertible_to<std::uint32_t>;
    { C::llama_cpu_rounding } -> std::convertible_to<bool>;
    { C::reference_threads } -> std::convertible_to<unsigned>;
};

// What the megakernel needs on top:
// - Gpu, the GPU the grid is sized for, with the megakernel's measured blocks_per_sm
// - threads, the grid: every SM runs Gpu::blocks_per_sm blocks of threads
// - RowKernel, how a warp multiplies a BF16 row by a vector
// - barrier_spin_limit, how many times a block polls a grid barrier before it gives up
template <class C>
concept MegakernelConfig =
    DecoderConfig<C> && GpuSpec<typename C::Gpu> && RowKernel<typename C::RowKernel> && requires {
        { C::Gpu::blocks_per_sm } -> std::convertible_to<unsigned>;
        { C::threads } -> std::convertible_to<unsigned>;
        { C::barrier_spin_limit } -> std::convertible_to<unsigned>;
    };

// The engine's own choices for model M on GPU G: FP32 arithmetic throughout, and a BF16
// cache like the weights. config/build.hpp names the one a build is for.
template <ModelSpec M, GpuSpec G>
struct EngineConfig {
    using Model = M;
    using Kv    = KvBF16;
    // CMake's MK_N_CTX, 4,096 by default as the roadmap measures up to that, and a model
    // trained for fewer positions gets those. Qwen3-0.6B allows 32,768, which would be 3.8 GB
    // of cache.
    static constexpr std::uint32_t n_ctx = std::min<std::uint32_t>(MK_N_CTX, M::context_length);
    static constexpr bool llama_cpu_rounding = false;
    // The reference pass's block size. Its kernels take one thread per element or stride by
    // it, so it does not change a result.
    static constexpr unsigned reference_threads = 256;

    using Gpu = G;
    // One thread per dimension of each query head of a group: how attention splits its work.
    static constexpr unsigned threads = M::n_head / M::n_head_kv * M::head_dim;
    using RowKernel = RowDot8;
    // About a second of spinning: a barrier that waits this long will never be met.
    static constexpr unsigned barrier_spin_limit = 1u << 27;
};

// llama.cpp's CPU backend, reproduced: its F16 cache and its rounding. For the op-by-op
// comparison of the reference pass with its dumps.
template <ModelSpec M>
struct LlamaCpuConfig {
    using Model = M;
    using Kv    = KvF16;
    static constexpr std::uint32_t n_ctx              = EngineConfig<M, BuildGpu>::n_ctx;
    static constexpr bool          llama_cpu_rounding = true;
    static constexpr unsigned      reference_threads =
        EngineConfig<M, BuildGpu>::reference_threads;
};

// The megakernel's grid for a configuration, derived from its GPU's SM count and blocks per
// SM.
template <MegakernelConfig C>
struct Grid {
    static constexpr unsigned threads         = C::threads;
    static constexpr unsigned blocks          = C::Gpu::sms * C::Gpu::blocks_per_sm;
    static constexpr unsigned warps_per_block = threads / warp_size;
    static constexpr unsigned warps           = blocks * warps_per_block;
    static_assert(threads % warp_size == 0, "a block is a whole number of warps");
};
} // namespace mk
