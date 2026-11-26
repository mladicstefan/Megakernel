#include "config/checksum_kernel.hpp"
#include "config/config.hpp"
#include "config/model_spec.hpp"

#include <algorithm>
#include <cstdint>

#include <gtest/gtest.h>

namespace {
// A model's members by the small concept they belong to, so a test model can leave one out.
struct Shape {
    static constexpr std::uint32_t n_layer = 2, n_embd = 512, n_ff = 1536;
};
struct Attention {
    static constexpr std::uint32_t n_head = 8, n_head_kv = 2, head_dim = 64;
};
struct Vocabulary {
    static constexpr std::uint32_t n_vocab = 1024;
};
struct Constants {
    static constexpr float norm_eps = 1e-6f, rope_base = 1e6f;
};
struct Layer {
    mk::Tensor q;
};
struct Tensors {
    static constexpr mk::Tensor token_embd{}, output_norm{}, output{};
    static constexpr Layer      layers[2]{};
};

template <std::uint32_t Positions>
struct Model : Shape, Attention, Vocabulary, Constants, Tensors {
    static constexpr std::uint32_t context_length = Positions;
};
static_assert(mk::ModelSpec<Model<40960>>);

// One requirement missing each.
struct NoHeadDim : Shape, Vocabulary, Constants, Tensors {
    static constexpr std::uint32_t n_head = 8, n_head_kv = 2, context_length = 40960;
};
struct NoContextLength : Shape, Attention, Vocabulary, Constants, Tensors {};
struct NoRopeBase : Shape, Attention, Vocabulary, Tensors {
    static constexpr float         norm_eps       = 1e-6f;
    static constexpr std::uint32_t context_length = 40960;
};
struct NoLayers : Shape, Attention, Vocabulary, Constants {
    static constexpr mk::Tensor    token_embd{}, output_norm{}, output{};
    static constexpr std::uint32_t context_length = 40960;
};
static_assert(!mk::ModelSpec<NoHeadDim>);
static_assert(!mk::ModelSpec<NoContextLength>);
static_assert(!mk::ModelSpec<NoRopeBase>);
static_assert(!mk::ModelSpec<NoLayers>);

using Long  = mk::EngineConfig<Model<40960>, mk::Rtx4070TiSuper>;
using Short = mk::EngineConfig<Model<2048>, mk::Rtx4070TiSuper>;
static_assert(mk::MegakernelConfig<Long>);
static_assert(mk::DecoderConfig<mk::LlamaCpuConfig<Model<40960>>>);

// A configuration by the members each concept asks for, so a test configuration can leave one
// out: a build that does so fails at the concept, not somewhere in a kernel.
struct DecoderParts {
    using Model = ::Model<40960>;
    using Kv    = mk::KvBF16;
    static constexpr std::uint32_t n_ctx              = 4096;
    static constexpr bool          llama_cpu_rounding = false;
};
struct ReferenceThreads {
    static constexpr unsigned reference_threads = 256;
};
struct GridParts {
    using Gpu       = mk::Rtx4070TiSuper;
    using RowKernel = mk::RowDot8;
    static constexpr unsigned threads = 256;
};
struct SpinLimit {
    static constexpr unsigned barrier_spin_limit = 1u << 20;
};
struct Complete : DecoderParts, ReferenceThreads, GridParts, SpinLimit {};
struct NoReferenceThreads : DecoderParts, GridParts, SpinLimit {};
struct NoSpinLimit : DecoderParts, ReferenceThreads, GridParts {};
static_assert(mk::MegakernelConfig<Complete>);
static_assert(!mk::DecoderConfig<NoReferenceThreads>);
static_assert(mk::DecoderConfig<NoSpinLimit> && !mk::MegakernelConfig<NoSpinLimit>);

struct NoThreads {
    static constexpr unsigned blocks_per_sm = 4;
};
struct NoBlocksPerSm {
    static constexpr unsigned threads = 256;
};
static_assert(mk::ChecksumKernel<mk::Checksum256x4>);
static_assert(!mk::ChecksumKernel<NoThreads>);
static_assert(!mk::ChecksumKernel<NoBlocksPerSm>);

TEST(EngineConfig, ThreadsAreOnePerDimensionOfAGroupOfQueryHeads) {
    EXPECT_EQ(Long::threads, 8u / 2u * 64u);
}

TEST(EngineConfig, TheCacheHoldsAtMostTheModelsContext) {
    // MK_N_CTX is CMake's cap on the positions, 4,096 unless the build says otherwise.
    constexpr std::uint32_t cap = MK_N_CTX;
    EXPECT_EQ(Long::n_ctx, std::min<std::uint32_t>(cap, 40960));
    EXPECT_EQ(Short::n_ctx, std::min<std::uint32_t>(cap, 2048));
    EXPECT_EQ(mk::LlamaCpuConfig<Model<2048>>::n_ctx, Short::n_ctx);
}

TEST(EngineConfig, TheReferencePassTakesItsBlockSizeFromTheConfiguration) {
    EXPECT_EQ(Long::reference_threads, 256u);
    EXPECT_EQ(mk::LlamaCpuConfig<Model<2048>>::reference_threads, Long::reference_threads);
}
} // namespace
