#pragma once
#include "model/model_types.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>

// What a decoder needs to know about a model. A model is a type that says what the numbers
// of the generated model.hpp mean for its architecture (src/arch/), and the decoders are
// templates over it, so every width is a constant in the kernels. nvcc compiles this
// header: C++20 only.
namespace mk {
// How deep and how wide the decoder is.
template <class M>
concept ModelShape = requires {
    { M::n_layer } -> std::convertible_to<unsigned>;
    { M::n_embd } -> std::convertible_to<unsigned>;
    { M::n_ff } -> std::convertible_to<unsigned>; // the feed-forward width
};

// Its grouped-query attention: n_head query heads share n_head_kv KV heads.
template <class M>
concept ModelAttention = requires {
    { M::n_head } -> std::convertible_to<unsigned>;
    { M::n_head_kv } -> std::convertible_to<unsigned>;
    { M::head_dim } -> std::convertible_to<unsigned>;
};

// What it reads: tokens of its vocabulary, up to the context length it was trained for.
template <class M>
concept ModelTokens = requires {
    { M::n_vocab } -> std::convertible_to<unsigned>;
    { M::context_length } -> std::convertible_to<std::uint32_t>;
};

// The constants of its RMSNorm and its RoPE.
template <class M>
concept ModelConstants = requires {
    { M::norm_eps } -> std::convertible_to<float>;
    { M::rope_base } -> std::convertible_to<float>;
};

// The tensors a decoder reads: the embedding, the output norm and matrix, and a table of
// every layer's tensors by their GGUF names.
template <class M>
concept ModelTensors = requires(std::size_t layer) {
    { M::token_embd } -> std::convertible_to<Tensor>;
    { M::output_norm } -> std::convertible_to<Tensor>;
    { M::output } -> std::convertible_to<Tensor>;
    M::layers[layer];
};

template <class M>
concept ModelSpec = ModelShape<M> && ModelAttention<M> && ModelTokens<M> && ModelConstants<M> &&
                    ModelTensors<M>;
} // namespace mk
