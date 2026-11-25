#include "megakernel.hpp"

#include "cuda_try.hpp"
#include "numeric.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mk {
namespace {
constexpr unsigned kThreads       = kMegaThreads;
constexpr unsigned kBlocks        = kMegaBlocks;
constexpr unsigned kWarpsPerBlock = kThreads / 32;
constexpr unsigned kWarps         = kBlocks * kWarpsPerBlock;
constexpr unsigned kBlocksPerSm   = kBlocks / BuildGpu::sms;
// One barrier after the embedding, five per layer, one after the logits.
constexpr unsigned kBarriers = 2 + 5 * static_cast<unsigned>(n_layer);
// About a second of spinning. A barrier that waits this long will never be met.
constexpr unsigned kSpinLimit = 1u << 27;

constexpr float    kNormEps   = arch::attention::layer_norm_rms_epsilon;
constexpr unsigned kEmbd      = n_embd;
constexpr unsigned kFf        = n_ff;
constexpr unsigned kHeadDim   = head_dim;
constexpr unsigned kHalf      = head_dim / 2;
constexpr unsigned kQ         = n_head * head_dim;    // the width of Q and of kqv_out
constexpr unsigned kKvPerPos  = n_head_kv * head_dim; // cache entries per layer and position
constexpr unsigned kGroup     = n_head / n_head_kv;   // query heads per KV head
constexpr unsigned kVocab     = n_vocab;

// row_dot reads a row 256 weights at a time, 16 bytes per thread.
static_assert(kEmbd % 256 == 0 && kFf % 256 == 0 && kQ % 256 == 0);
static_assert(kHeadDim <= kThreads && kThreads % 32 == 0);
// A warp's first row is warp * rows / kWarps, in 32 bits.
static_assert(std::uint64_t{kWarps} * kVocab < (std::uint64_t{1} << 32));

// Where each tensor starts, as constants in the kernel.
struct LayerOffsets {
    std::uint64_t attn_norm, q, k, v, q_norm, k_norm, o, ffn_norm, gate, up, down;
};
struct Offsets {
    LayerOffsets  layer[n_layer];
    std::uint64_t token_embd, output_norm, output;
};
constexpr Offsets make_offsets() {
    Offsets o{};
    for (std::size_t l = 0; l < n_layer; ++l) {
        const auto& t = model::layers[l];
        o.layer[l]    = {t.attn_norm.offset, t.q.offset,        t.k.offset,    t.v.offset,
                         t.q_norm.offset,    t.k_norm.offset,   t.o.offset,    t.ffn_norm.offset,
                         t.gate.offset,      t.up.offset,       t.down.offset};
    }
    o.token_embd  = token_embd.offset;
    o.output_norm = output_norm.offset;
    o.output      = output.offset;
    return o;
}
// At namespace scope: a kernel may copy this, but may not call make_offsets itself.
constexpr Offsets kOffsets = make_offsets();

__device__ const std::uint16_t* bf16_at(const std::byte* base, std::uint64_t offset) {
    return reinterpret_cast<const std::uint16_t*>(base + offset);
}
__device__ const float* f32_at(const std::byte* base, std::uint64_t offset) {
    return reinterpret_cast<const float*>(base + offset);
}

// Reads a value that another block wrote during this launch. A plain load may be served
// from this SM's L1 cache, which does not see other SMs' writes.
__device__ float fresh(const float* p) {
    const volatile float* v = p;
    return *v;
}

__device__ float warp_sum(float v) {
    for (unsigned lanes = 16; lanes > 0; lanes /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, lanes);
    }
    return v;
}

// The sum, or the maximum, of v over the block's threads. Every thread gets the result.
// scratch is shared memory for one value per warp.
__device__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = v;
    }
    __syncthreads();
    float total = 0;
    for (unsigned w = 0; w < kWarpsPerBlock; ++w) {
        total += scratch[w];
    }
    __syncthreads(); // scratch is free again
    return total;
}
__device__ float block_max(float v, float* scratch) {
    for (unsigned lanes = 16; lanes > 0; lanes /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, lanes));
    }
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = v;
    }
    __syncthreads();
    float max = scratch[0];
    for (unsigned w = 1; w < kWarpsPerBlock; ++w) {
        max = fmaxf(max, scratch[w]);
    }
    __syncthreads();
    return max;
}

// Waits until every block of the grid has finished the phase that ends here. target is
// what sync->arrived reaches then.
__device__ void barrier(MegaSync* sync, unsigned target) {
    __threadfence(); // this thread's writes are visible to the whole device
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&sync->arrived, 1u);
        const volatile unsigned* arrived = &sync->arrived;
        const volatile unsigned* failed  = &sync->failed;
        unsigned                 spins   = 0;
        // The difference as a signed number, so the count may wrap around.
        while (static_cast<int>(*arrived - target) < 0) {
            if (++spins > kSpinLimit || *failed != 0) {
                atomicExch(&sync->failed, 1u);
                break;
            }
        }
    }
    __syncthreads();
}

// Copies n values that other blocks wrote into the block's shared memory.
__device__ void load(const float* x, unsigned n, float* xs) {
    for (unsigned i = threadIdx.x; i < n; i += kThreads) {
        xs[i] = fresh(x + i);
    }
    __syncthreads();
}

// The same, through RMSNorm: xs = x / sqrt(mean(x^2) + eps) * weight.
__device__ void load_normed(const float* x, const float* weight, unsigned n, float* xs,
                            float* scratch) {
    float squares = 0;
    for (unsigned i = threadIdx.x; i < n; i += kThreads) {
        xs[i] = fresh(x + i);
        squares += xs[i] * xs[i];
    }
    const float scale =
        1.0f / sqrtf(block_sum(squares, scratch) / static_cast<float>(n) + kNormEps);
    for (unsigned i = threadIdx.x; i < n; i += kThreads) {
        xs[i] = xs[i] * scale * weight[i];
    }
    __syncthreads();
}

// The dot product of one BF16 row of Cols weights with xs, computed by the whole warp: each
// lane takes 8 consecutive weights (16 bytes) at a time, so the warp reads 512 contiguous
// bytes per step. Every lane gets the result.
template <unsigned Cols>
__device__ float row_dot(const std::uint16_t* row, const float* xs) {
    float sum = 0;
    for (unsigned c = (threadIdx.x % 32) * 8; c < Cols; c += 256) {
        // Aligned: tensors start on 32 bytes and a row is a multiple of 16 bytes.
        const uint4  w = *reinterpret_cast<const uint4*>(row + c);
        const float* x = xs + c;
        sum += bf16_to_float(static_cast<std::uint16_t>(w.x)) * x[0] +
               bf16_to_float(static_cast<std::uint16_t>(w.x >> 16)) * x[1] +
               bf16_to_float(static_cast<std::uint16_t>(w.y)) * x[2] +
               bf16_to_float(static_cast<std::uint16_t>(w.y >> 16)) * x[3] +
               bf16_to_float(static_cast<std::uint16_t>(w.z)) * x[4] +
               bf16_to_float(static_cast<std::uint16_t>(w.z >> 16)) * x[5] +
               bf16_to_float(static_cast<std::uint16_t>(w.w)) * x[6] +
               bf16_to_float(static_cast<std::uint16_t>(w.w >> 16)) * x[7];
    }
    return warp_sum(sum);
}

// The schedule of a matrix: the grid's warps split its rows into equal contiguous runs.
// Warp g takes rows [g * rows / kWarps, (g + 1) * rows / kWarps), so every row has exactly
// one warp.
struct Rows {
    unsigned begin, end;
};
__device__ Rows my_rows(unsigned rows) {
    const unsigned warp = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
    return {warp * rows / kWarps, (warp + 1) * rows / kWarps};
}

// store(row, W[row] . xs) for each of this warp's rows of a rows x Cols matrix.
template <unsigned Cols, class Store>
__device__ void for_rows(const std::uint16_t* w, unsigned rows, const float* xs, Store store) {
    const Rows mine = my_rows(rows);
    for (unsigned r = mine.begin; r < mine.end; ++r) {
        const float y = row_dot<Cols>(w + std::size_t{r} * Cols, xs);
        if (threadIdx.x % 32 == 0) {
            store(r, y);
        }
    }
}

// Attention for one query head, by one block: q_norm, k_norm, RoPE, the cache write, the
// scores, their softmax and the weighted sum of V. The block's own K and V of this position
// come from shared memory, in the cache's precision, and the earlier positions from the
// cache. scores is this head's scratch in global memory.
template <DecoderConfig C>
__device__ void attention(unsigned head, const LayerTrace* in, LayerTrace* out,
                          const float* q_norm, const float* k_norm,
                          typename C::Kv::Bits* k_cache, typename C::Kv::Bits* v_cache,
                          const float* rope, float* scores, unsigned pos, float* scratch) {
    using Kv = typename C::Kv;
    __shared__ float q[kHeadDim];
    __shared__ float k[kHeadDim];
    __shared__ float v[kHeadDim];
    const unsigned   t       = threadIdx.x;
    const unsigned   kv_head = head / kGroup; // consecutive query heads share a KV head
    const unsigned   slot    = kv_head * kHeadDim;

    float q_squares = 0;
    float k_squares = 0;
    if (t < kHeadDim) {
        q[t]      = fresh(in->Qcur + head * kHeadDim + t);
        k[t]      = fresh(in->Kcur + slot + t);
        v[t]      = fresh(in->Vcur + slot + t);
        q_squares = q[t] * q[t];
        k_squares = k[t] * k[t];
    }
    const float n       = static_cast<float>(kHeadDim);
    const float q_scale = 1.0f / sqrtf(block_sum(q_squares, scratch) / n + kNormEps);
    const float k_scale = 1.0f / sqrtf(block_sum(k_squares, scratch) / n + kNormEps);
    if (t < kHeadDim) {
        q[t] = q[t] * q_scale * q_norm[t];
        k[t] = k[t] * k_scale * k_norm[t];
    }
    __syncthreads();

    // RoPE: element t rotates with element t + kHalf.
    if (t < kHalf) {
        const float c  = rope[t];
        const float s  = rope[kHalf + t];
        const float q0 = q[t], q1 = q[t + kHalf];
        const float k0 = k[t], k1 = k[t + kHalf];
        q[t]           = q0 * c - q1 * s;
        q[t + kHalf]   = q0 * s + q1 * c;
        k[t]           = k0 * c - k1 * s;
        k[t + kHalf]   = k0 * s + k1 * c;
    }
    __syncthreads();

    // Later steps read this position from the cache, so this step uses the same rounded
    // values. Only the first query head of a group writes the cache: one writer per slot.
    if (t < kHeadDim) {
        const typename Kv::Bits k_bits = Kv::store(k[t]);
        const typename Kv::Bits v_bits = Kv::store(v[t]);
        if (head % kGroup == 0) {
            k_cache[std::size_t{pos} * kKvPerPos + slot + t] = k_bits;
            v_cache[std::size_t{pos} * kKvPerPos + slot + t] = v_bits;
        }
        k[t] = Kv::load(k_bits);
        v[t] = Kv::load(v_bits);
    }
    __syncthreads();

    const float scale    = 1.0f / sqrtf(n);
    float       my_max   = -INFINITY;
    for (unsigned j = t; j <= pos; j += kThreads) {
        float dot = 0;
        if (j < pos) {
            const typename Kv::Bits* kj = k_cache + std::size_t{j} * kKvPerPos + slot;
            for (unsigned i = 0; i < kHeadDim; ++i) {
                dot += Kv::load(kj[i]) * q[i];
            }
        } else {
            for (unsigned i = 0; i < kHeadDim; ++i) {
                dot += k[i] * q[i];
            }
        }
        scores[j] = dot * scale;
        my_max    = fmaxf(my_max, scores[j]);
    }
    const float max    = block_max(my_max, scratch);
    float       my_sum = 0;
    for (unsigned j = t; j <= pos; j += kThreads) {
        scores[j] = expf(scores[j] - max);
        my_sum += scores[j];
    }
    const float total = block_sum(my_sum, scratch); // its barriers also publish scores

    if (t < kHeadDim) {
        float sum = 0;
        for (unsigned j = 0; j < pos; ++j) {
            sum += scores[j] / total * Kv::load(v_cache[std::size_t{j} * kKvPerPos + slot + t]);
        }
        sum += scores[pos] / total * v[t];
        out->kqv_out[head * kHeadDim + t] = sum;
    }
}

// One decode step. out is the trace the phases write, in the trace they read: the same one
// in a normal step.
// The launch bounds make the compiler fit kBlocksPerSm blocks of this kernel on one SM.
template <DecoderConfig C>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm)
megakernel(Weights weights, Trace* out, const Trace* in,
                           typename C::Kv::Bits* k_cache, typename C::Kv::Bits* v_cache,
                           const float* rope, float* scores, MegaSync* sync, unsigned token,
                           unsigned pos, unsigned epoch) {
    // static: without it every thread rebuilds the table on its stack.
    static constexpr Offsets offsets = kOffsets;
    __shared__ float         xs[kFf]; // the vector of the phase's matrix products
    __shared__ float         scratch[kWarpsPerBlock];
    const std::byte* const   base    = weights.base;
    unsigned                 arrived = epoch; // what sync->arrived reaches at the next barrier

    // The embedding row, by one block.
    if (blockIdx.x == 0) {
        const std::uint16_t* row = bf16_at(base, offsets.token_embd) + std::size_t{token} * kEmbd;
        for (unsigned i = threadIdx.x; i < kEmbd; i += kThreads) {
            out->embd[i] = bf16_to_float(row[i]);
        }
    }
    barrier(sync, arrived += kBlocks);

    const float* x = in->embd; // the residual stream
    for (unsigned l = 0; l < n_layer; ++l) {
        const LayerOffsets&      o = offsets.layer[l];
        LayerTrace* const        a = &out->layers[l];
        const LayerTrace* const  i = &in->layers[l];
        const std::size_t cache    = std::size_t{l} * C::n_ctx * kKvPerPos;

        // Q, K and V of the normed residual stream.
        load_normed(x, f32_at(base, o.attn_norm), kEmbd, xs, scratch);
        for_rows<kEmbd>(bf16_at(base, o.q), kQ, xs, [a](unsigned r, float y) { a->Qcur[r] = y; });
        for_rows<kEmbd>(bf16_at(base, o.k), kKvPerPos, xs,
                        [a](unsigned r, float y) { a->Kcur[r] = y; });
        for_rows<kEmbd>(bf16_at(base, o.v), kKvPerPos, xs,
                        [a](unsigned r, float y) { a->Vcur[r] = y; });
        barrier(sync, arrived += kBlocks);

        // Attention: one block per query head.
        if (blockIdx.x < n_head) {
            attention<C>(blockIdx.x, i, a, f32_at(base, o.q_norm), f32_at(base, o.k_norm),
                         k_cache + cache, v_cache + cache, rope,
                         scores + std::size_t{blockIdx.x} * C::n_ctx, pos, scratch);
        }
        barrier(sync, arrived += kBlocks);

        // The output projection, added to the residual stream.
        load(i->kqv_out, kQ, xs);
        for_rows<kQ>(bf16_at(base, o.o), kEmbd, xs,
                     [a, x](unsigned r, float y) { a->ffn_inp[r] = y + fresh(x + r); });
        barrier(sync, arrived += kBlocks);

        // Gate and up of the normed stream, through SwiGLU. A warp does both rows.
        load_normed(i->ffn_inp, f32_at(base, o.ffn_norm), kEmbd, xs, scratch);
        {
            const std::uint16_t* gate = bf16_at(base, o.gate);
            const std::uint16_t* up   = bf16_at(base, o.up);
            const Rows           mine = my_rows(kFf);
            for (unsigned r = mine.begin; r < mine.end; ++r) {
                const float g = row_dot<kEmbd>(gate + std::size_t{r} * kEmbd, xs);
                const float u = row_dot<kEmbd>(up + std::size_t{r} * kEmbd, xs);
                if (threadIdx.x % 32 == 0) {
                    a->ffn_swiglu[r] = g / (1.0f + expf(-g)) * u;
                }
            }
        }
        barrier(sync, arrived += kBlocks);

        // Down, added to the residual stream.
        load(i->ffn_swiglu, kFf, xs);
        for_rows<kFf>(bf16_at(base, o.down), kEmbd, xs, [a, i](unsigned r, float y) {
            a->l_out[r] = y + fresh(i->ffn_inp + r);
        });
        barrier(sync, arrived += kBlocks);
        x = i->l_out;
    }

    // The logits of the normed stream, and each block's best one. A later row wins only if
    // it is strictly larger, so the first maximum wins, as in llama.cpp's greedy sampler.
    load_normed(x, f32_at(base, offsets.output_norm), kEmbd, xs, scratch);
    {
        __shared__ float     warp_logit[kWarpsPerBlock];
        __shared__ unsigned  warp_token[kWarpsPerBlock];
        const std::uint16_t* w    = bf16_at(base, offsets.output);
        const Rows           mine = my_rows(kVocab);
        float                best = -INFINITY;
        unsigned             at   = mine.begin;
        for (unsigned r = mine.begin; r < mine.end; ++r) {
            const float y = row_dot<kEmbd>(w + std::size_t{r} * kEmbd, xs);
            if (threadIdx.x % 32 == 0) {
                out->result_output[r] = y;
                if (y > best) {
                    best = y;
                    at   = r;
                }
            }
        }
        if (threadIdx.x % 32 == 0) {
            warp_logit[threadIdx.x / 32] = best;
            warp_token[threadIdx.x / 32] = at;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            // Warps hold rising rows, so the first strictly larger one is the first maximum.
            unsigned first = 0;
            for (unsigned warp = 1; warp < kWarpsPerBlock; ++warp) {
                if (warp_logit[warp] > warp_logit[first]) {
                    first = warp;
                }
            }
            sync->best_logit[blockIdx.x] = warp_logit[first];
            sync->best_token[blockIdx.x] = warp_token[first];
        }
    }
    barrier(sync, arrived += kBlocks);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const volatile float*    logit = sync->best_logit;
        const volatile unsigned* tok   = sync->best_token;
        unsigned                 first = 0;
        for (unsigned b = 1; b < kBlocks; ++b) {
            if (logit[b] > logit[first]) {
                first = b;
            }
        }
        sync->token = tok[first];
    }
}

template <class T>
cudaError_t device_new(std::unique_ptr<T, CudaFree>& out, std::size_t count) {
    void*             p = nullptr;
    const cudaError_t e = cudaMalloc(&p, count * sizeof(T));
    out.reset(static_cast<T*>(p));
    return e;
}

template <DecoderConfig C>
std::variant<Token, CudaError> launch(Megakernel<C>& m, Decoder<C>& d, Token token,
                                      const Trace* in) {
    assert(d.n_past < C::n_ctx);
    megakernel<C><<<kBlocks, kThreads>>>(d.weights, d.trace.get(), in, d.k_cache.get(),
                                         d.v_cache.get(),
                                         d.rope.get() + std::size_t{d.n_past} * head_dim,
                                         d.scores.get(), m.sync.get(), token.id, d.n_past,
                                         m.epoch);
    MK_TRY(cudaGetLastError());
    // failed and token, read together. The copy waits for the kernel.
    unsigned result[2];
    MK_TRY(cudaMemcpy(result, &m.sync->failed, sizeof result, cudaMemcpyDeviceToHost));
    if (result[0] != 0) {
        return CudaError{"megakernel", "a barrier gave up waiting for the rest of the grid"};
    }
    m.epoch += kBarriers * kBlocks;
    ++d.n_past;
    return *Token::checked(result[1]); // a row of the output matrix: inside the vocabulary
}
} // namespace

template <DecoderConfig C>
std::variant<Megakernel<C>, CudaError> Megakernel<C>::make() {
    // A barrier waits for every block, so every block must be running. That holds only if
    // each SM can keep its share of the grid resident.
    int resident = 0;
    MK_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, megakernel<C>,
                                                         static_cast<int>(kThreads), 0));
    if (resident < static_cast<int>(kBlocksPerSm)) {
        return CudaError{"megakernel", "the GPU cannot keep the whole grid resident"};
    }
    Megakernel m;
    MK_TRY(device_new(m.sync, 1));
    MK_TRY(cudaMemset(m.sync.get(), 0, sizeof(MegaSync)));
    return m;
}

template <DecoderConfig C>
std::variant<Token, CudaError> Megakernel<C>::decode(Decoder<C>& d, Token token) {
    return launch(*this, d, token, d.trace.get());
}

template <DecoderConfig C>
std::variant<Token, CudaError> Megakernel<C>::decode_from(Decoder<C>& d, Token token,
                                                          const Trace& from) {
    if (!operands) {
        MK_TRY(device_new(operands, 1));
    }
    MK_TRY(cudaMemcpy(operands.get(), &from, sizeof(Trace), cudaMemcpyHostToDevice));
    return launch(*this, d, token, operands.get());
}

template struct Megakernel<EngineConfig>;
} // namespace mk
