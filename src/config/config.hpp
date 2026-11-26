#pragma once
#include "config/gpu.hpp"
#include "config/kv_format.hpp"
#include "config/model_spec.hpp"
#include "config/row_kernel.hpp"

#include <algorithm>
#include <concepts>
#include <cstdint>

// What a build is compiled for. A configuration is a type, and the decoders and the
// megakernel are templates over it: nothing here is decided at run time. nvcc compiles this
// header: C++20 only.
namespace mk {
// What a decoder needs:
// - Model, the model it decodes
// - Kv, the KV cache's format
// - n_ctx, the positions the KV cache holds
// - llama_cpu_rounding, whether the ops also round where llama.cpp's CPU backend rounds
template <class C>
concept DecoderConfig = ModelSpec<typename C::Model> && KvFormat<typename C::Kv> && requires {
    { C::n_ctx } -> std::convertible_to<std::uint32_t>;
    { C::llama_cpu_rounding } -> std::convertible_to<bool>;
};

// What the megakernel needs on top:
// - Gpu, the GPU the grid is sized for, with the megakernel's measured blocks_per_sm
// - threads, the grid: every SM runs Gpu::blocks_per_sm blocks of threads
// - RowKernel, how a warp multiplies a BF16 row by a vector
template <class C>
concept MegakernelConfig =
    DecoderConfig<C> && GpuSpec<typename C::Gpu> && RowKernel<typename C::RowKernel> && requires {
        { C::Gpu::blocks_per_sm } -> std::convertible_to<unsigned>;
        { C::threads } -> std::convertible_to<unsigned>;
    };

// The engine's own choices for model M on GPU G: FP32 arithmetic throughout, and a BF16
// cache like the weights. config/build.hpp names the one a build is for.
template <ModelSpec M, GpuSpec G>
struct EngineConfig {
    using Model = M;
    using Kv    = KvBF16;
    // The roadmap measures up to 4,096 positions, and a model trained for fewer gets those.
    // Qwen3-0.6B allows 32,768, which would be 3.8 GB of cache.
    static constexpr std::uint32_t n_ctx = std::min<std::uint32_t>(4096, M::context_length);
    static constexpr bool llama_cpu_rounding = false;

    using Gpu = G;
    // One thread per dimension of each query head of a group: how attention splits its work.
    static constexpr unsigned threads = M::n_head / M::n_head_kv * M::head_dim;
    using RowKernel = RowDot8;
};

// llama.cpp's CPU backend, reproduced: its F16 cache and its rounding. For the op-by-op
// comparison of the reference pass with its dumps.
template <ModelSpec M>
struct LlamaCpuConfig {
    using Model = M;
    using Kv    = KvF16;
    static constexpr std::uint32_t n_ctx              = EngineConfig<M, BuildGpu>::n_ctx;
    static constexpr bool          llama_cpu_rounding = true;
};

// The megakernel's grid for a configuration, derived from its GPU's SM count and blocks per
// SM.
template <MegakernelConfig C>
struct Grid {
    static constexpr unsigned threads         = C::threads;
    static constexpr unsigned blocks          = C::Gpu::sms * C::Gpu::blocks_per_sm;
    static constexpr unsigned warps_per_block = threads / 32;
    static constexpr unsigned warps           = blocks * warps_per_block;
    static_assert(threads % 32 == 0, "a block is a whole number of warps");
};
} // namespace mk
