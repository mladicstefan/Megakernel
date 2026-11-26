#include "reference/forward.hpp"

#include "base/cuda_try.hpp"
#include "base/numeric.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mk {
namespace {
constexpr unsigned kThreads = 256;

// RoPE rotates element i with i + kHalf.
template <ModelSpec M>
constexpr unsigned kHalf = M::head_dim / 2;
// Cache entries per layer and position.
template <ModelSpec M>
constexpr std::size_t kKvPerPos = std::size_t{M::n_head_kv} * M::head_dim;

constexpr unsigned blocks(std::size_t threads) {
    return static_cast<unsigned>((threads + kThreads - 1) / kThreads);
}

// Tensor data on the device. Aligned: weights.hpp asserts it for the offsets.
const std::uint16_t* bf16(Weights w, const Tensor& t) {
    return reinterpret_cast<const std::uint16_t*>(w.base + t.offset);
}
const float* f32(Weights w, const Tensor& t) {
    return reinterpret_cast<const float*>(w.base + t.offset);
}

__device__ unsigned thread_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// The embedding row of one token.
template <ModelSpec M>
__global__ void embed_kernel(const std::uint16_t* row, float* out) {
    const unsigned i = thread_index();
    if (i < M::n_embd) {
        out[i] = bf16_to_float(row[i]);
    }
}

// RMSNorm of each vector of n in x: y = x / sqrt(mean(x^2) + eps) * weight. One block per
// vector, and every vector uses the same weight.
template <ModelSpec M>
__global__ void rms_norm_kernel(const float* x, const float* weight, float* y, unsigned n) {
    __shared__ float scale;
    const float*     in = x + blockIdx.x * n;
    if (threadIdx.x == 0) {
        float sum = 0;
        for (unsigned i = 0; i < n; ++i) {
            sum += in[i] * in[i];
        }
        scale = 1.0f / sqrtf(sum / static_cast<float>(n) + M::norm_eps);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        y[blockIdx.x * n + i] = in[i] * scale * weight[i];
    }
}

// y = W x, with W rows x cols of BF16 in row-major order. One thread per row, summing in
// column order.
template <DecoderConfig C>
__global__ void matvec_kernel(const std::uint16_t* w, const float* x, float* y, unsigned rows,
                              unsigned cols) {
    const unsigned r = thread_index();
    if (r >= rows) {
        return;
    }
    const std::uint16_t* row = w + std::size_t{r} * cols;
    float                sum = 0;
    for (unsigned c = 0; c < cols; ++c) {
        // llama.cpp's CPU backend rounds the vector to BF16 before a BF16 dot product.
        const float xc = C::llama_cpu_rounding ? round_to_bf16(x[c]) : x[c];
        sum += bf16_to_float(row[c]) * xc;
    }
    y[r] = sum;
}

// RoPE on each head of x, pairing element i with element i + head_dim/2 (ggml's NEOX type).
// cos_sin is this position's row of the table: the cosines, then the sines.
template <ModelSpec M>
__global__ void rope_kernel(const float* x, const float* cos_sin, float* y, unsigned heads) {
    const unsigned t = thread_index();
    if (t >= heads * kHalf<M>) {
        return;
    }
    const unsigned i   = t % kHalf<M>;
    const float*   in  = x + (t / kHalf<M>) * M::head_dim;
    float*         out = y + (t / kHalf<M>) * M::head_dim;
    const float    c   = cos_sin[i];
    const float    s   = cos_sin[kHalf<M> + i];
    out[i]             = in[i] * c - in[i + kHalf<M>] * s;
    out[i + kHalf<M>]  = in[i] * s + in[i + kHalf<M>] * c;
}

// Writes this position's K and V into its slot of the cache, in the cache's format.
template <ModelSpec M, KvFormat F>
__global__ void store_kv_kernel(const float* k, const float* v, typename F::Bits* k_slot,
                                typename F::Bits* v_slot) {
    const unsigned i = thread_index();
    if (i < kKvPerPos<M>) {
        k_slot[i] = F::store(k[i]);
        v_slot[i] = F::store(v[i]);
    }
}

// What ggml's CPU flash attention computes for one output element with an F16 cache: an
// online softmax over the positions, with the running weighted sum kept in F16.
template <ModelSpec M, KvFormat F>
__device__ float online_softmax_f16(const float* scores, const typename F::Bits* v,
                                    unsigned n_kv) {
    float       max = -INFINITY;
    float       sum = 0;
    KvF16::Bits acc = KvF16::store(0.0f);
    for (unsigned j = 0; j < n_kv; ++j) {
        float rescale = 1.0f; // of what is summed so far, when a new maximum arrives
        float weight  = 1.0f; // exp(score - max) of this position
        if (scores[j] > max) {
            rescale = expf(max - scores[j]);
            max     = scores[j];
            acc     = KvF16::store(KvF16::load(acc) * rescale);
        } else {
            weight = expf(scores[j] - max);
        }
        acc = KvF16::store(KvF16::load(acc) + F::load(v[j * kKvPerPos<M>]) * weight);
        sum = sum * rescale + weight;
    }
    return KvF16::load(acc) * (1.0f / sum);
}

// Grouped-query attention of one step over the n_kv cached positions. One block per query
// head: the scores against every cached K, their softmax, then the weighted sum of the
// cached V. scores is scratch memory for n_head * n_kv values.
template <DecoderConfig C>
__global__ void attention_kernel(const float* q, const typename C::Kv::Bits* k_cache,
                                 const typename C::Kv::Bits* v_cache, unsigned n_kv, float* scores,
                                 float* out) {
    using F = typename C::Kv;
    using M = typename C::Model;
    const unsigned head = blockIdx.x;
    // Consecutive query heads share a KV head, as in ggml.
    const unsigned          kv_head  = head / (M::n_head / M::n_head_kv);
    const float*            qh       = q + head * M::head_dim;
    const typename F::Bits* k        = k_cache + kv_head * M::head_dim; // this head at position 0
    const typename F::Bits* v        = v_cache + kv_head * M::head_dim;
    float*                  s        = scores + std::size_t{head} * n_kv;
    const float             scale    = 1.0f / sqrtf(static_cast<float>(M::head_dim));

    for (unsigned j = threadIdx.x; j < n_kv; j += blockDim.x) {
        float dot = 0;
        for (unsigned i = 0; i < M::head_dim; ++i) {
            // ggml rounds Q to its cache's F16 before the dot product.
            const float qi = C::llama_cpu_rounding ? KvF16::load(KvF16::store(qh[i])) : qh[i];
            dot += F::load(k[j * kKvPerPos<M> + i]) * qi;
        }
        s[j] = dot * scale;
    }
    __syncthreads();

    if constexpr (C::llama_cpu_rounding) {
        for (unsigned i = threadIdx.x; i < M::head_dim; i += blockDim.x) {
            out[head * M::head_dim + i] = online_softmax_f16<M, F>(s, v + i, n_kv);
        }
        return;
    }

    __shared__ float max_score;
    __shared__ float total;
    if (threadIdx.x == 0) {
        float max = s[0];
        for (unsigned j = 1; j < n_kv; ++j) {
            max = fmaxf(max, s[j]);
        }
        float sum = 0;
        for (unsigned j = 0; j < n_kv; ++j) {
            sum += expf(s[j] - max);
        }
        max_score = max;
        total     = sum;
    }
    __syncthreads();
    for (unsigned j = threadIdx.x; j < n_kv; j += blockDim.x) {
        s[j] = expf(s[j] - max_score) / total; // now the attention weights
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < M::head_dim; i += blockDim.x) {
        float sum = 0;
        for (unsigned j = 0; j < n_kv; ++j) {
            sum += s[j] * F::load(v[j * kKvPerPos<M> + i]);
        }
        out[head * M::head_dim + i] = sum;
    }
}

__global__ void add_kernel(const float* a, const float* b, float* y, unsigned n) {
    const unsigned i = thread_index();
    if (i < n) {
        y[i] = a[i] + b[i];
    }
}

// y = silu(gate) * up.
__global__ void swiglu_kernel(const float* gate, const float* up, float* y, unsigned n) {
    const unsigned i = thread_index();
    if (i < n) {
        y[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
    }
}

template <DecoderConfig C>
void matvec(Weights w, const Tensor& m, const float* x, float* y) {
    const auto cols = static_cast<unsigned>(m.dims[0]);
    const auto rows = static_cast<unsigned>(m.dims[1]);
    matvec_kernel<C><<<blocks(rows), kThreads>>>(bf16(w, m), x, y, rows, cols);
}

template <ModelSpec M>
void rms_norm(Weights w, const Tensor& weight, const float* x, float* y, unsigned vectors) {
    rms_norm_kernel<M><<<vectors, kThreads>>>(x, f32(w, weight), y,
                                              static_cast<unsigned>(weight.dims[0]));
}

template <class T>
cudaError_t device_new(std::unique_ptr<T, CudaFree>& out, std::size_t count) {
    void*             p = nullptr;
    const cudaError_t e = cudaMalloc(&p, count * sizeof(T));
    out.reset(static_cast<T*>(p));
    return e;
}

// One step. Every op writes its result into the trace, and takes its operands from in: the
// trace itself in a normal step, or another implementation's trace of the same position.
template <DecoderConfig C>
std::optional<CudaError> step(Decoder<C>& d, Token<typename C::Model> token,
                              const Trace<typename C::Model>* in) {
    assert(d.n_past < C::n_ctx);
    using Kv = typename C::Kv;
    using M  = typename C::Model;
    constexpr std::size_t kKvPerLayer = std::size_t{C::n_ctx} * kKvPerPos<M>;
    const Weights         w           = d.weights;
    const unsigned        n_kv        = d.n_past + 1; // attention sees this position too
    // t and in point at device memory: the host only forms addresses in them.
    Trace<M>* const    t    = d.trace.get();
    const float* const rope = d.rope.get() + std::size_t{d.n_past} * M::head_dim;

    embed_kernel<M><<<blocks(M::n_embd), kThreads>>>(
        bf16(w, M::token_embd) + std::size_t{token.id} * M::n_embd, t->embd);
    const float* x = in->embd; // the residual stream
    for (std::size_t l = 0; l < M::n_layer; ++l) {
        const auto&                L       = M::layers[l];
        LayerTrace<M>* const       a       = &t->layers[l];
        const LayerTrace<M>* const i       = &in->layers[l];
        typename Kv::Bits* const   k_cache = d.k_cache.get() + l * kKvPerLayer;
        typename Kv::Bits* const   v_cache = d.v_cache.get() + l * kKvPerLayer;

        rms_norm<M>(w, L.attn_norm, x, a->attn_norm, 1);
        matvec<C>(w, L.q, i->attn_norm, a->Qcur);
        matvec<C>(w, L.k, i->attn_norm, a->Kcur);
        matvec<C>(w, L.v, i->attn_norm, a->Vcur);
        rms_norm<M>(w, L.q_norm, i->Qcur, a->Qcur_normed, M::n_head);
        rms_norm<M>(w, L.k_norm, i->Kcur, a->Kcur_normed, M::n_head_kv);
        rope_kernel<M><<<blocks(M::n_head * kHalf<M>), kThreads>>>(i->Qcur_normed, rope,
                                                                    a->Qcur_rope, M::n_head);
        rope_kernel<M><<<blocks(M::n_head_kv * kHalf<M>), kThreads>>>(i->Kcur_normed, rope,
                                                                       a->Kcur_rope, M::n_head_kv);
        store_kv_kernel<M, Kv><<<blocks(kKvPerPos<M>), kThreads>>>(
            i->Kcur_rope, i->Vcur, k_cache + d.n_past * kKvPerPos<M>,
            v_cache + d.n_past * kKvPerPos<M>);
        attention_kernel<C><<<M::n_head, kThreads>>>(i->Qcur_rope, k_cache, v_cache, n_kv,
                                                     d.scores.get(), a->kqv_out);
        matvec<C>(w, L.o, i->kqv_out, a->attn_out);
        add_kernel<<<blocks(M::n_embd), kThreads>>>(i->attn_out, x, a->ffn_inp, M::n_embd);

        rms_norm<M>(w, L.ffn_norm, i->ffn_inp, a->ffn_norm, 1);
        matvec<C>(w, L.gate, i->ffn_norm, a->ffn_gate);
        matvec<C>(w, L.up, i->ffn_norm, a->ffn_up);
        swiglu_kernel<<<blocks(M::n_ff), kThreads>>>(i->ffn_gate, i->ffn_up, a->ffn_swiglu,
                                                     M::n_ff);
        matvec<C>(w, L.down, i->ffn_swiglu, a->ffn_out);
        add_kernel<<<blocks(M::n_embd), kThreads>>>(i->ffn_out, i->ffn_inp, a->l_out, M::n_embd);
        x = i->l_out;
    }
    rms_norm<M>(w, M::output_norm, x, t->result_norm, 1);
    matvec<C>(w, M::output, in->result_norm, t->result_output);

    // A failed launch stays the last error until it is read, so one check covers them all.
    MK_TRY(cudaGetLastError());
    MK_TRY(cudaDeviceSynchronize());
    ++d.n_past;
    return std::nullopt;
}
} // namespace

template <DecoderConfig C>
std::variant<Decoder<C>, CudaError> Decoder<C>::make(Weights weights) {
    using M                     = Model;
    constexpr std::size_t n_ctx = C::n_ctx;
    Decoder               d{.weights = weights};
    MK_TRY(device_new(d.trace, 1));
    MK_TRY(device_new(d.k_cache, M::n_layer * n_ctx * kKvPerPos<M>));
    MK_TRY(device_new(d.v_cache, M::n_layer * n_ctx * kKvPerPos<M>));
    MK_TRY(device_new(d.rope, n_ctx * M::head_dim));
    MK_TRY(device_new(d.scores, M::n_head * n_ctx));

    // The table ggml's CPU RoPE builds for each position, with the same F32 steps: the angle
    // starts at the position and is multiplied by theta_scale from one pair to the next.
    std::vector<float> table(n_ctx * M::head_dim);
    const float theta_scale = std::pow(M::rope_base, -2.0f / static_cast<float>(M::head_dim));
    for (std::size_t pos = 0; pos < n_ctx; ++pos) {
        float theta = static_cast<float>(pos);
        for (std::size_t i = 0; i < kHalf<M>; ++i) {
            table[pos * M::head_dim + i]            = std::cos(theta);
            table[pos * M::head_dim + kHalf<M> + i] = std::sin(theta);
            theta *= theta_scale;
        }
    }
    MK_TRY(cudaMemcpy(d.rope.get(), table.data(), table.size() * sizeof(float),
                      cudaMemcpyHostToDevice));
    return d;
}

template <DecoderConfig C>
std::optional<CudaError> Decoder<C>::decode(Token<Model> token) {
    return step(*this, token, trace.get());
}

template <DecoderConfig C>
std::optional<CudaError> Decoder<C>::decode_from(Token<Model> token, const Trace<Model>& from) {
    if (!operands) {
        MK_TRY(device_new(operands, 1));
    }
    MK_TRY(cudaMemcpy(operands.get(), &from, sizeof(Trace<Model>), cudaMemcpyHostToDevice));
    return step(*this, token, operands.get());
}

template <DecoderConfig C>
std::optional<CudaError> Decoder<C>::read_trace(Trace<Model>& out) const {
    MK_TRY(cudaMemcpy(&out, trace.get(), sizeof(Trace<Model>), cudaMemcpyDeviceToHost));
    return std::nullopt;
}

template <DecoderConfig C>
std::optional<CudaError> Decoder<C>::read_logits(std::span<float, Model::n_vocab> out) const {
    MK_TRY(cudaMemcpy(out.data(), trace->result_output, out.size_bytes(), cudaMemcpyDeviceToHost));
    return std::nullopt;
}

template struct Decoder<BuildConfig>;
template struct Decoder<LlamaCpuConfig<BuildModel>>;
} // namespace mk
