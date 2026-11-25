#include "forward.hpp"

#include "cuda_try.hpp"
#include "numeric.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_fp16.h>

namespace mk {
namespace {
constexpr unsigned kThreads = 256;

constexpr float       kNormEps  = arch::attention::layer_norm_rms_epsilon;
constexpr float       kRopeBase = arch::rope::freq_base;
constexpr unsigned    kHalf     = head_dim / 2;         // RoPE rotates element i with i + kHalf
constexpr std::size_t kKvPerPos = n_head_kv * head_dim; // cache entries per layer and position

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
__device__ float f16_to_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}
__device__ std::uint16_t float_to_f16(float f) {
    return __half_as_ushort(__float2half(f));
}

// The embedding row of one token.
__global__ void embed_kernel(const std::uint16_t* row, float* out) {
    const unsigned i = thread_index();
    if (i < n_embd) {
        out[i] = bf16_to_float(row[i]);
    }
}

// RMSNorm of each vector of n in x: y = x / sqrt(mean(x^2) + eps) * weight. One block per
// vector, and every vector uses the same weight.
__global__ void rms_norm_kernel(const float* x, const float* weight, float* y, unsigned n) {
    __shared__ float scale;
    const float*     in = x + blockIdx.x * n;
    if (threadIdx.x == 0) {
        float sum = 0;
        for (unsigned i = 0; i < n; ++i) {
            sum += in[i] * in[i];
        }
        scale = 1.0f / sqrtf(sum / static_cast<float>(n) + kNormEps);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        y[blockIdx.x * n + i] = in[i] * scale * weight[i];
    }
}

// y = W x, with W rows x cols of BF16 in row-major order. One thread per row, summing in
// column order.
__global__ void matvec_kernel(const std::uint16_t* w, const float* x, float* y, unsigned rows,
                              unsigned cols, Arithmetic arithmetic) {
    const unsigned r = thread_index();
    if (r >= rows) {
        return;
    }
    const std::uint16_t* row = w + std::size_t{r} * cols;
    float                sum = 0;
    for (unsigned c = 0; c < cols; ++c) {
        // llama.cpp's CPU backend rounds the vector to BF16 before a BF16 dot product.
        const float xc = arithmetic == Arithmetic::llama_cpu ? round_to_bf16(x[c]) : x[c];
        sum += bf16_to_float(row[c]) * xc;
    }
    y[r] = sum;
}

// RoPE on each head of x, pairing element i with element i + head_dim/2 (ggml's NEOX type).
// cos_sin is this position's row of the table: the cosines, then the sines.
__global__ void rope_kernel(const float* x, const float* cos_sin, float* y, unsigned heads) {
    const unsigned t = thread_index();
    if (t >= heads * kHalf) {
        return;
    }
    const unsigned i   = t % kHalf;
    const float*   in  = x + (t / kHalf) * head_dim;
    float*         out = y + (t / kHalf) * head_dim;
    const float    c   = cos_sin[i];
    const float    s   = cos_sin[kHalf + i];
    out[i]             = in[i] * c - in[i + kHalf] * s;
    out[i + kHalf]     = in[i] * s + in[i + kHalf] * c;
}

// Writes this position's K and V into its slot of the cache, rounded to F16.
__global__ void store_kv_kernel(const float* k, const float* v, std::uint16_t* k_slot,
                                std::uint16_t* v_slot) {
    const unsigned i = thread_index();
    if (i < kKvPerPos) {
        k_slot[i] = float_to_f16(k[i]);
        v_slot[i] = float_to_f16(v[i]);
    }
}

// What ggml's CPU flash attention computes for one output element with an F16 cache: an
// online softmax over the positions, with the running weighted sum kept in F16.
__device__ float online_softmax_f16(const float* scores, const std::uint16_t* v, unsigned n_kv) {
    float         max = -INFINITY;
    float         sum = 0;
    std::uint16_t acc = float_to_f16(0.0f);
    for (unsigned j = 0; j < n_kv; ++j) {
        float rescale = 1.0f; // of what is summed so far, when a new maximum arrives
        float weight  = 1.0f; // exp(score - max) of this position
        if (scores[j] > max) {
            rescale = expf(max - scores[j]);
            max     = scores[j];
            acc     = float_to_f16(f16_to_float(acc) * rescale);
        } else {
            weight = expf(scores[j] - max);
        }
        acc = float_to_f16(f16_to_float(acc) + f16_to_float(v[j * kKvPerPos]) * weight);
        sum = sum * rescale + weight;
    }
    return f16_to_float(acc) * (1.0f / sum);
}

// Grouped-query attention of one step over the n_kv cached positions. One block per query
// head: the scores against every cached K, their softmax, then the weighted sum of the
// cached V. scores is scratch memory for n_head * n_kv values.
__global__ void attention_kernel(const float* q, const std::uint16_t* k_cache,
                                 const std::uint16_t* v_cache, unsigned n_kv, float* scores,
                                 float* out, Arithmetic arithmetic) {
    const unsigned head = blockIdx.x;
    // Consecutive query heads share a KV head, as in ggml.
    const unsigned       kv_head  = head / (n_head / n_head_kv);
    const float*         qh       = q + head * head_dim;
    const std::uint16_t* k        = k_cache + kv_head * head_dim; // this head at position 0
    const std::uint16_t* v        = v_cache + kv_head * head_dim;
    float*               s        = scores + std::size_t{head} * n_kv;
    const float          scale    = 1.0f / sqrtf(static_cast<float>(head_dim));
    const bool           as_llama = arithmetic == Arithmetic::llama_cpu;

    for (unsigned j = threadIdx.x; j < n_kv; j += blockDim.x) {
        float dot = 0;
        for (unsigned i = 0; i < head_dim; ++i) {
            // ggml rounds Q to the cache's F16 before the dot product.
            const float qi = as_llama ? f16_to_float(float_to_f16(qh[i])) : qh[i];
            dot += f16_to_float(k[j * kKvPerPos + i]) * qi;
        }
        s[j] = dot * scale;
    }
    __syncthreads();

    if (as_llama) {
        for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
            out[head * head_dim + i] = online_softmax_f16(s, v + i, n_kv);
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
    for (unsigned i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float sum = 0;
        for (unsigned j = 0; j < n_kv; ++j) {
            sum += s[j] * f16_to_float(v[j * kKvPerPos + i]);
        }
        out[head * head_dim + i] = sum;
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

void matvec(Weights w, const Tensor& m, const float* x, float* y, Arithmetic arithmetic) {
    const auto cols = static_cast<unsigned>(m.dims[0]);
    const auto rows = static_cast<unsigned>(m.dims[1]);
    matvec_kernel<<<blocks(rows), kThreads>>>(bf16(w, m), x, y, rows, cols, arithmetic);
}

void rms_norm(Weights w, const Tensor& weight, const float* x, float* y, unsigned vectors) {
    rms_norm_kernel<<<vectors, kThreads>>>(x, f32(w, weight), y,
                                           static_cast<unsigned>(weight.dims[0]));
}

template <class T>
cudaError_t device_new(std::unique_ptr<T, CudaFree>& out, std::size_t count) {
    void*             p = nullptr;
    const cudaError_t e = cudaMalloc(&p, count * sizeof(T));
    out.reset(static_cast<T*>(p));
    return e;
}
} // namespace

std::variant<Decoder, CudaError> make_decoder(Weights weights, std::uint32_t n_ctx,
                                              Arithmetic arithmetic) {
    Decoder d{.weights = weights, .arithmetic = arithmetic, .n_ctx = n_ctx};
    MK_TRY(device_new(d.trace, 1));
    MK_TRY(device_new(d.k_cache, n_layer * n_ctx * kKvPerPos));
    MK_TRY(device_new(d.v_cache, n_layer * n_ctx * kKvPerPos));
    MK_TRY(device_new(d.rope, std::size_t{n_ctx} * head_dim));
    MK_TRY(device_new(d.scores, n_head * std::size_t{n_ctx}));

    // The table ggml's CPU RoPE builds for each position, with the same F32 steps: the angle
    // starts at the position and is multiplied by theta_scale from one pair to the next.
    std::vector<float> table(std::size_t{n_ctx} * head_dim);
    const float        theta_scale = std::pow(kRopeBase, -2.0f / static_cast<float>(head_dim));
    for (std::size_t pos = 0; pos < n_ctx; ++pos) {
        float theta = static_cast<float>(pos);
        for (std::size_t i = 0; i < kHalf; ++i) {
            table[pos * head_dim + i]         = std::cos(theta);
            table[pos * head_dim + kHalf + i] = std::sin(theta);
            theta *= theta_scale;
        }
    }
    MK_TRY(cudaMemcpy(d.rope.get(), table.data(), table.size() * sizeof(float),
                      cudaMemcpyHostToDevice));
    return d;
}

namespace {
// One step. Every op writes its result into the trace, and takes its operands from in: the
// trace itself in a normal step, or another implementation's trace of the same position.
std::optional<CudaError> step(Decoder& d, Token token, const Trace* in) {
    assert(d.n_past < d.n_ctx);
    const Weights    w     = d.weights;
    const Arithmetic arith = d.arithmetic;
    const unsigned   n_kv  = d.n_past + 1; // attention sees this position too
    // t and in point at device memory: the host only forms addresses in them.
    Trace* const       t    = d.trace.get();
    const float* const rope = d.rope.get() + std::size_t{d.n_past} * head_dim;

    embed_kernel<<<blocks(n_embd), kThreads>>>(bf16(w, token_embd) + std::size_t{token.id} * n_embd,
                                               t->embd);
    const float* x = in->embd; // the residual stream
    for (std::size_t l = 0; l < n_layer; ++l) {
        const auto&             L       = model::layers[l];
        LayerTrace* const       a       = &t->layers[l];
        const LayerTrace* const i       = &in->layers[l];
        std::uint16_t* const    k_cache = d.k_cache.get() + l * d.n_ctx * kKvPerPos;
        std::uint16_t* const    v_cache = d.v_cache.get() + l * d.n_ctx * kKvPerPos;

        rms_norm(w, L.attn_norm, x, a->attn_norm, 1);
        matvec(w, L.q, i->attn_norm, a->Qcur, arith);
        matvec(w, L.k, i->attn_norm, a->Kcur, arith);
        matvec(w, L.v, i->attn_norm, a->Vcur, arith);
        rms_norm(w, L.q_norm, i->Qcur, a->Qcur_normed, n_head);
        rms_norm(w, L.k_norm, i->Kcur, a->Kcur_normed, n_head_kv);
        rope_kernel<<<blocks(n_head * kHalf), kThreads>>>(i->Qcur_normed, rope, a->Qcur_rope,
                                                          n_head);
        rope_kernel<<<blocks(n_head_kv * kHalf), kThreads>>>(i->Kcur_normed, rope, a->Kcur_rope,
                                                             n_head_kv);
        store_kv_kernel<<<blocks(kKvPerPos), kThreads>>>(i->Kcur_rope, i->Vcur,
                                                         k_cache + d.n_past * kKvPerPos,
                                                         v_cache + d.n_past * kKvPerPos);
        attention_kernel<<<n_head, kThreads>>>(i->Qcur_rope, k_cache, v_cache, n_kv,
                                               d.scores.get(), a->kqv_out, arith);
        matvec(w, L.o, i->kqv_out, a->attn_out, arith);
        add_kernel<<<blocks(n_embd), kThreads>>>(i->attn_out, x, a->ffn_inp, n_embd);

        rms_norm(w, L.ffn_norm, i->ffn_inp, a->ffn_norm, 1);
        matvec(w, L.gate, i->ffn_norm, a->ffn_gate, arith);
        matvec(w, L.up, i->ffn_norm, a->ffn_up, arith);
        swiglu_kernel<<<blocks(n_ff), kThreads>>>(i->ffn_gate, i->ffn_up, a->ffn_swiglu, n_ff);
        matvec(w, L.down, i->ffn_swiglu, a->ffn_out, arith);
        add_kernel<<<blocks(n_embd), kThreads>>>(i->ffn_out, i->ffn_inp, a->l_out, n_embd);
        x = i->l_out;
    }
    rms_norm(w, output_norm, x, t->result_norm, 1);
    matvec(w, output, in->result_norm, t->result_output, arith);

    // A failed launch stays the last error until it is read, so one check covers them all.
    MK_TRY(cudaGetLastError());
    MK_TRY(cudaDeviceSynchronize());
    ++d.n_past;
    return std::nullopt;
}
} // namespace

std::optional<CudaError> decode(Decoder& d, Token token) {
    return step(d, token, d.trace.get());
}

std::optional<CudaError> decode_from(Decoder& d, Token token, const Trace& operands) {
    if (!d.operands) {
        MK_TRY(device_new(d.operands, 1));
    }
    MK_TRY(cudaMemcpy(d.operands.get(), &operands, sizeof(Trace), cudaMemcpyHostToDevice));
    return step(d, token, d.operands.get());
}

std::optional<CudaError> read_trace(const Decoder& d, Trace& out) {
    MK_TRY(cudaMemcpy(&out, d.trace.get(), sizeof(Trace), cudaMemcpyDeviceToHost));
    return std::nullopt;
}

std::optional<CudaError> read_logits(const Decoder& d, std::span<float, n_vocab> out) {
    MK_TRY(cudaMemcpy(out.data(), d.trace->result_output, out.size_bytes(),
                      cudaMemcpyDeviceToHost));
    return std::nullopt;
}
} // namespace mk
