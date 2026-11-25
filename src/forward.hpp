#pragma once
#include "kv_format.hpp"
#include "model.hpp"
#include "weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <variant>

// The reference forward pass: one decode step of Qwen3 as a sequence of plain kernels, one
// per op, with FP32 arithmetic over the BF16 weights. It is the correctness oracle for the
// megakernel, so it keeps every intermediate and favours clarity over speed. nvcc compiles
// this header: C++20 only.
namespace mk {
static_assert(model::meta::general::architecture == "qwen3",
              "the forward pass is written for the qwen3 graph only");

namespace arch = model::meta::arch;

inline constexpr std::size_t n_layer   = layer_count;
inline constexpr std::size_t n_embd    = arch::embedding_length;
inline constexpr std::size_t n_ff      = arch::feed_forward_length;
inline constexpr std::size_t n_head    = arch::attention::head_count;
inline constexpr std::size_t n_head_kv = arch::attention::head_count_kv;
inline constexpr std::size_t head_dim  = arch::attention::key_length;

inline constexpr Tensor token_embd  = find(model::tensors, "token_embd.weight");
inline constexpr Tensor output_norm = find(model::tensors, "output_norm.weight");
inline constexpr Tensor output      = find(model::tensors, "output.weight");
inline constexpr std::size_t n_vocab = token_embd.dims[1];

static_assert(n_layer == arch::block_count);
static_assert(arch::attention::value_length == head_dim);
static_assert(n_head % n_head_kv == 0 && head_dim % 2 == 0);

// A matrix is BF16 with dims {columns, rows}, a norm weight is F32. Anything else in the
// file is a model this pass was not written for.
constexpr bool is_matrix(const Tensor& t, std::size_t cols, std::size_t rows) {
    return t.dtype == DType::BF16 && t.n_dims == 2 && t.dims[0] == cols && t.dims[1] == rows;
}
constexpr bool is_norm(const Tensor& t, std::size_t n) {
    return t.dtype == DType::F32 && t.n_dims == 1 && t.dims[0] == n;
}
constexpr bool layers_have_qwen3_shapes() {
    for (const auto& l : model::layers) {
        if (!(is_norm(l.attn_norm, n_embd) && is_matrix(l.q, n_embd, n_head * head_dim) &&
              is_matrix(l.k, n_embd, n_head_kv * head_dim) &&
              is_matrix(l.v, n_embd, n_head_kv * head_dim) && is_norm(l.q_norm, head_dim) &&
              is_norm(l.k_norm, head_dim) && is_matrix(l.o, n_head * head_dim, n_embd) &&
              is_norm(l.ffn_norm, n_embd) && is_matrix(l.gate, n_embd, n_ff) &&
              is_matrix(l.up, n_embd, n_ff) && is_matrix(l.down, n_ff, n_embd))) {
            return false;
        }
    }
    return true;
}
static_assert(layers_have_qwen3_shapes());
static_assert(is_matrix(token_embd, n_embd, n_vocab) && is_matrix(output, n_embd, n_vocab) &&
              is_norm(output_norm, n_embd));

// Every intermediate of one layer of one step. The fields carry the names of the tensors
// that tools/reference dumps from llama.cpp, so the two can be compared name by name.
struct LayerTrace {
    float attn_norm[n_embd];
    float Qcur[n_head * head_dim];
    float Kcur[n_head_kv * head_dim];
    float Vcur[n_head_kv * head_dim];
    float Qcur_normed[n_head * head_dim];
    float Kcur_normed[n_head_kv * head_dim];
    float Qcur_rope[n_head * head_dim];
    float Kcur_rope[n_head_kv * head_dim];
    float kqv_out[n_head * head_dim]; // attention, before the output projection
    float attn_out[n_embd];
    float ffn_inp[n_embd];
    float ffn_norm[n_embd];
    float ffn_gate[n_ff];
    float ffn_up[n_ff];
    float ffn_swiglu[n_ff];
    float ffn_out[n_embd];
    float l_out[n_embd];
};

// The working memory of one decode step. Each kernel reads some fields and writes one.
struct Trace {
    float      embd[n_embd];
    LayerTrace layers[n_layer];
    float      result_norm[n_embd];
    float      result_output[n_vocab]; // the logits
};

// Calls f(layer, name, values) for every tensor of a trace but the logits, in the order a
// step computes them. layer is -1 for a tensor that belongs to no layer. The names are the
// file names of a tools/reference dump.
template <class TraceT, class F>
void for_each_tensor(TraceT& t, F&& f) {
    f(-1, std::string_view{"embd"}, std::span{t.embd});
    for (std::size_t l = 0; l < n_layer; ++l) {
        auto&      a     = t.layers[l];
        const auto layer = [&](std::string_view name, auto values) {
            f(static_cast<int>(l), name, values);
        };
        layer("attn_norm", std::span{a.attn_norm});
        layer("Qcur", std::span{a.Qcur});
        layer("Kcur", std::span{a.Kcur});
        layer("Vcur", std::span{a.Vcur});
        layer("Qcur_normed", std::span{a.Qcur_normed});
        layer("Kcur_normed", std::span{a.Kcur_normed});
        layer("Qcur_rope", std::span{a.Qcur_rope});
        layer("Kcur_rope", std::span{a.Kcur_rope});
        layer("kqv_out", std::span{a.kqv_out});
        layer("attn_out", std::span{a.attn_out});
        layer("ffn_inp", std::span{a.ffn_inp});
        layer("ffn_norm", std::span{a.ffn_norm});
        layer("ffn_gate", std::span{a.ffn_gate});
        layer("ffn_up", std::span{a.ffn_up});
        layer("ffn_swiglu", std::span{a.ffn_swiglu});
        layer("ffn_out", std::span{a.ffn_out});
        layer("l_out", std::span{a.l_out});
    }
    f(-1, std::string_view{"result_norm"}, std::span{t.result_norm});
}

// A token id that is inside the vocabulary.
struct Token {
    std::uint32_t id;

    static constexpr std::optional<Token> checked(std::uint64_t id) {
        if (id >= n_vocab) {
            return std::nullopt;
        }
        return Token{static_cast<std::uint32_t>(id)};
    }
};

enum class Arithmetic {
    exact,     // FP32 throughout. What the megakernel is checked against.
    llama_cpu, // also rounds where llama.cpp's CPU backend rounds, to compare with its dumps
};

// One sequence being decoded: its KV cache and the working memory of the current step, all
// on the device. It borrows the weights, so the DeviceWeights must outlive it.
struct Decoder {
    Weights       weights;
    Arithmetic    arithmetic;
    std::uint32_t n_ctx;      // positions the KV cache holds
    std::uint32_t n_past = 0; // positions decoded so far

    std::unique_ptr<Trace, CudaFree> trace;
    // [layer][position][KV head][head_dim], in the build's cache format.
    std::unique_ptr<Kv::Bits, CudaFree> k_cache;
    std::unique_ptr<Kv::Bits, CudaFree> v_cache;
    std::unique_ptr<float, CudaFree>         rope;   // [position]: head_dim/2 cosines, then sines
    std::unique_ptr<float, CudaFree>         scores; // attention scratch, [query head][position]
    std::unique_ptr<Trace, CudaFree>         operands; // for decode_from; null until it is used
};

std::variant<Decoder, CudaError> make_decoder(Weights weights, std::uint32_t n_ctx,
                                              Arithmetic arithmetic);

// Runs token at position n_past, which must be below n_ctx. Afterwards the device trace
// holds every intermediate of this step.
std::optional<CudaError> decode(Decoder& d, Token token);

// The same step, but every op takes its operands from operands: another implementation's
// trace of this position. Each op is then checked on its own, on inputs the other
// implementation saw, and an error in one op cannot hide in the next. The KV cache takes
// that trace's K and V too.
std::optional<CudaError> decode_from(Decoder& d, Token token, const Trace& operands);

std::optional<CudaError> read_trace(const Decoder& d, Trace& out);
std::optional<CudaError> read_logits(const Decoder& d, std::span<float, n_vocab> out);
} // namespace mk
