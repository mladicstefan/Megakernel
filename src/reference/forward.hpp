#pragma once
#include "config/build.hpp"
#include "weights/weights.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
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
// Every intermediate of one layer of one step. The fields carry the names of the tensors
// that tools/reference dumps from llama.cpp, so the two can be compared name by name.
template <ModelSpec M>
struct LayerTrace {
    float attn_norm[M::n_embd];
    float Qcur[M::n_head * M::head_dim];
    float Kcur[M::n_head_kv * M::head_dim];
    float Vcur[M::n_head_kv * M::head_dim];
    float Qcur_normed[M::n_head * M::head_dim];
    float Kcur_normed[M::n_head_kv * M::head_dim];
    float Qcur_rope[M::n_head * M::head_dim];
    float Kcur_rope[M::n_head_kv * M::head_dim];
    float kqv_out[M::n_head * M::head_dim]; // attention, before the output projection
    float attn_out[M::n_embd];
    float ffn_inp[M::n_embd];
    float ffn_norm[M::n_embd];
    float ffn_gate[M::n_ff];
    float ffn_up[M::n_ff];
    float ffn_swiglu[M::n_ff];
    float ffn_out[M::n_embd];
    float l_out[M::n_embd];
};

// The working memory of one decode step. Each kernel reads some fields and writes one.
template <ModelSpec M>
struct Trace {
    float         embd[M::n_embd];
    LayerTrace<M> layers[M::n_layer];
    float         result_norm[M::n_embd];
    float         result_output[M::n_vocab]; // the logits
};

// Calls f(layer, name, values) for every tensor of a trace but the logits, in the order a
// step computes them. layer is -1 for a tensor that belongs to no layer. The names are the
// file names of a tools/reference dump.
template <class TraceT, class F>
void for_each_tensor(TraceT& t, F&& f) {
    f(-1, std::string_view{"embd"}, std::span{t.embd});
    for (std::size_t l = 0; l < std::size(t.layers); ++l) {
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

// A token id that is inside model M's vocabulary.
template <ModelSpec M>
struct Token {
    std::uint32_t id;

    static constexpr std::optional<Token> checked(std::uint64_t id) {
        if (id >= M::n_vocab) {
            return std::nullopt;
        }
        return Token{static_cast<std::uint32_t>(id)};
    }
};

// One sequence being decoded: its KV cache and the working memory of the current step, all
// on the device. It borrows the weights, so the DeviceWeights must outlive it.
template <DecoderConfig C>
struct Decoder {
    using Model = typename C::Model;

    Weights       weights;
    std::uint32_t n_past = 0; // positions decoded so far

    std::unique_ptr<Trace<Model>, CudaFree> trace;
    // [layer][position][KV head][head_dim].
    std::unique_ptr<typename C::Kv::Bits, CudaFree> k_cache;
    std::unique_ptr<typename C::Kv::Bits, CudaFree> v_cache;
    std::unique_ptr<float, CudaFree> rope;   // [position]: head_dim/2 cosines, then sines
    std::unique_ptr<float, CudaFree> scores; // attention scratch, [query head][position]
    std::unique_ptr<Trace<Model>, CudaFree> operands; // for decode_from; null until it is used

    static std::variant<Decoder, CudaError> make(Weights weights);

    // Runs token at position n_past, which must be below C::n_ctx. Afterwards the device
    // trace holds every intermediate of this step.
    std::optional<CudaError> decode(Token<Model> token);

    // The same step, but every op takes its operands from operands: another
    // implementation's trace of this position. Each op is then checked on its own, on
    // inputs the other implementation saw, and an error in one op cannot hide in the next.
    // The KV cache takes that trace's K and V too.
    std::optional<CudaError> decode_from(Token<Model> token, const Trace<Model>& operands);

    std::optional<CudaError> read_trace(Trace<Model>& out) const;
    std::optional<CudaError> read_logits(std::span<float, Model::n_vocab> out) const;
};

// forward.cu defines the members for these configurations.
extern template struct Decoder<BuildConfig>;
extern template struct Decoder<LlamaCpuConfig<BuildModel>>;
} // namespace mk
