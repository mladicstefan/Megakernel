#pragma once
#include "config/model_spec.hpp"
#include "model.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>

// The qwen3 architecture: what the numbers in model.hpp mean for a qwen3 model. Every member
// is read from the generated header, so another qwen3 model is another MODEL_GGUF and no
// edit. CMake includes this header only when the GGUF's general.architecture is qwen3. nvcc
// compiles this header: C++20 only.
namespace mk {
static_assert(model::meta::general::architecture == "qwen3",
              "arch/qwen3.hpp describes a qwen3 model, and model.hpp is another architecture");

struct Qwen3 {
    static constexpr std::uint32_t n_layer   = model::meta::qwen3::block_count;
    static constexpr std::uint32_t n_embd    = model::meta::qwen3::embedding_length;
    static constexpr std::uint32_t n_ff      = model::meta::qwen3::feed_forward_length;
    static constexpr std::uint32_t n_head    = model::meta::qwen3::attention::head_count;
    static constexpr std::uint32_t n_head_kv = model::meta::qwen3::attention::head_count_kv;
    static constexpr std::uint32_t head_dim  = model::meta::qwen3::attention::key_length;
    // The positions it was trained for.
    static constexpr std::uint32_t context_length = model::meta::qwen3::context_length;
    static constexpr float norm_eps  = model::meta::qwen3::attention::layer_norm_rms_epsilon;
    static constexpr float rope_base = model::meta::qwen3::rope::freq_base;

    static constexpr Tensor      token_embd  = find(model::tensors, "token_embd.weight");
    static constexpr Tensor      output_norm = find(model::tensors, "output_norm.weight");
    static constexpr Tensor      output      = find(model::tensors, "output.weight");
    static constexpr const auto& layers      = model::layers;
    // The embedding has one row per token. is_matrix below checks that the cast kept it.
    static constexpr auto n_vocab = static_cast<std::uint32_t>(token_embd.dims[1]);
};
static_assert(ModelSpec<Qwen3>);

static_assert(std::size(Qwen3::layers) == Qwen3::n_layer);
static_assert(model::meta::qwen3::attention::value_length == Qwen3::head_dim);
static_assert(Qwen3::n_head % Qwen3::n_head_kv == 0 && Qwen3::head_dim % 2 == 0);

// A matrix is BF16 with dims {columns, rows}, a norm weight is F32. Anything else in the
// file is a model the decoders were not written for.
constexpr bool is_matrix(const Tensor& t, std::size_t cols, std::size_t rows) {
    return t.dtype == DType::BF16 && t.n_dims == 2 && t.dims[0] == cols && t.dims[1] == rows;
}
constexpr bool is_norm(const Tensor& t, std::size_t n) {
    return t.dtype == DType::F32 && t.n_dims == 1 && t.dims[0] == n;
}
constexpr bool layers_have_qwen3_shapes() {
    using M = Qwen3;
    for (const auto& l : M::layers) {
        if (!(is_norm(l.attn_norm, M::n_embd) &&
              is_matrix(l.q, M::n_embd, std::size_t{M::n_head} * M::head_dim) &&
              is_matrix(l.k, M::n_embd, std::size_t{M::n_head_kv} * M::head_dim) &&
              is_matrix(l.v, M::n_embd, std::size_t{M::n_head_kv} * M::head_dim) &&
              is_norm(l.q_norm, M::head_dim) && is_norm(l.k_norm, M::head_dim) &&
              is_matrix(l.o, std::size_t{M::n_head} * M::head_dim, M::n_embd) &&
              is_norm(l.ffn_norm, M::n_embd) && is_matrix(l.gate, M::n_embd, M::n_ff) &&
              is_matrix(l.up, M::n_embd, M::n_ff) && is_matrix(l.down, M::n_ff, M::n_embd))) {
            return false;
        }
    }
    return true;
}
static_assert(layers_have_qwen3_shapes());
static_assert(is_matrix(Qwen3::token_embd, Qwen3::n_embd, Qwen3::n_vocab) &&
              is_matrix(Qwen3::output, Qwen3::n_embd, Qwen3::n_vocab) &&
              is_norm(Qwen3::output_norm, Qwen3::n_embd));
} // namespace mk
