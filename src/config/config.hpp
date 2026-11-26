#pragma once
#include "config/gpu.hpp"
#include "config/kv_format.hpp"
#include "config/row_kernel.hpp"

#include <concepts>
#include <cstdint>

// What a build is compiled for. A configuration is a type, and the decoders and the
// megakernel are templates over it: nothing here is decided at run time. nvcc compiles this
// header: C++20 only.
namespace mk {
// What a decoder needs:
// - Kv, the KV cache's format
// - n_ctx, the positions the KV cache holds
// - llama_cpu_rounding, whether the ops also round where llama.cpp's CPU backend rounds
template <class C>
concept DecoderConfig = KvFormat<typename C::Kv> && requires {
    { C::n_ctx } -> std::convertible_to<std::uint32_t>;
    { C::llama_cpu_rounding } -> std::convertible_to<bool>;
};

// What the megakernel needs on top:
// - Gpu, the GPU the grid is sized for
// - blocks_per_sm and threads, the grid: every SM runs blocks_per_sm blocks of threads
// - RowKernel, how a warp multiplies a BF16 row by a vector
template <class C>
concept MegakernelConfig =
    DecoderConfig<C> && GpuSpec<typename C::Gpu> && RowKernel<typename C::RowKernel> && requires {
        { C::blocks_per_sm } -> std::convertible_to<unsigned>;
        { C::threads } -> std::convertible_to<unsigned>;
    };

// The engine's own choices: FP32 arithmetic throughout, and a BF16 cache like the weights.
struct EngineConfig {
    using Kv = KvBF16;
    // The roadmap measures up to 4,096 positions. The model allows 32,768, which would be
    // 3.8 GB of cache.
    static constexpr std::uint32_t n_ctx              = 4096;
    static constexpr bool          llama_cpu_rounding = false;

    using Gpu = Rtx4070TiSuper;
    // Measured with the megakernel over 141 positions: 2.27 ms per step with 2 blocks per
    // SM, 2.29 with 1, 2.30 with 3, 2.43 with 4. More than 4 do not fit on an SM.
    static constexpr unsigned blocks_per_sm = 2;
    static constexpr unsigned threads       = 256;
    using RowKernel = RowDot8;
};
static_assert(MegakernelConfig<EngineConfig>);

// llama.cpp's CPU backend, reproduced: its F16 cache and its rounding. For the op-by-op
// comparison of the reference pass with its dumps.
struct LlamaCpuConfig {
    using Kv = KvF16;
    static constexpr std::uint32_t n_ctx              = EngineConfig::n_ctx;
    static constexpr bool          llama_cpu_rounding = true;
};
static_assert(DecoderConfig<LlamaCpuConfig>);

// The megakernel's grid for a configuration, derived from its GPU's SM count.
template <MegakernelConfig C>
struct Grid {
    static constexpr unsigned threads         = C::threads;
    static constexpr unsigned blocks          = C::Gpu::sms * C::blocks_per_sm;
    static constexpr unsigned warps_per_block = threads / 32;
    static constexpr unsigned warps           = blocks * warps_per_block;
    static_assert(threads % 32 == 0, "a block is a whole number of warps");
};
} // namespace mk
