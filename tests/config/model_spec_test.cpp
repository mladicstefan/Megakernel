#include "config/config.hpp"
#include "config/model_spec.hpp"

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

TEST(EngineConfig, ThreadsAreOnePerDimensionOfAGroupOfQueryHeads) {
    EXPECT_EQ(Long::threads, 8u / 2u * 64u);
}

TEST(EngineConfig, TheCacheHoldsAtMostTheModelsContext) {
    EXPECT_EQ(Long::n_ctx, 4096u);
    EXPECT_EQ(Short::n_ctx, 2048u);
    EXPECT_EQ(mk::LlamaCpuConfig<Model<2048>>::n_ctx, 2048u);
}
} // namespace
