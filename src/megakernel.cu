#include "megakernel.hpp"

#include "cuda_try.hpp"
#include "numeric.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mk {
namespace {
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
constexpr unsigned kKvHeads   = n_head_kv;
constexpr unsigned kLaneDims  = kHeadDim / 32; // head dimensions one lane of a warp holds
constexpr unsigned kVocab     = n_vocab;
static_assert(kHeadDim % 32 == 0);

// Attention splits the grid by KV head: block b works for KV head b % kKvHeads, as its
// member b / kKvHeads.
template <MegakernelConfig C>
constexpr unsigned kMembers = Grid<C>::blocks / kKvHeads;

// What a grid must satisfy to run this model. Checked when the kernel is instantiated.
template <MegakernelConfig C>
constexpr bool grid_fits_model =
    // The row kernel reads whole steps of every matrix it is given.
    kEmbd % C::RowKernel::step == 0 && kFf % C::RowKernel::step == 0 &&
    kQ % C::RowKernel::step == 0 &&
    // Each KV head gets the same number of blocks.
    Grid<C>::blocks % kKvHeads == 0 &&
    // The attention merges give one thread to each (query head of the group, dimension).
    Grid<C>::threads == kGroup * kHeadDim &&
    // A warp's first row is warp * rows / warps, in 32 bits.
    std::uint64_t{Grid<C>::warps} * kVocab < (std::uint64_t{1} << 32);

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

// The sum of v over the block's threads. Every thread gets the result. scratch is shared
// memory for one value per warp.
template <MegakernelConfig C>
__device__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = v;
    }
    __syncthreads();
    float total = 0;
    for (unsigned w = 0; w < Grid<C>::warps_per_block; ++w) {
        total += scratch[w];
    }
    __syncthreads(); // scratch is free again
    return total;
}

// Waits until every block of the grid has finished the phase that ends here. target is
// what sync->arrived reaches then.
template <MegakernelConfig C>
__device__ void barrier(MegaSync<C>* sync, unsigned target) {
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
template <MegakernelConfig C>
__device__ void load(const float* x, unsigned n, float* xs) {
    for (unsigned i = threadIdx.x; i < n; i += Grid<C>::threads) {
        xs[i] = fresh(x + i);
    }
    __syncthreads();
}

// The same, through RMSNorm: xs = x / sqrt(mean(x^2) + eps) * weight.
template <MegakernelConfig C>
__device__ void load_normed(const float* x, const float* weight, unsigned n, float* xs,
                            float* scratch) {
    float squares = 0;
    for (unsigned i = threadIdx.x; i < n; i += Grid<C>::threads) {
        xs[i] = fresh(x + i);
        squares += xs[i] * xs[i];
    }
    const float scale =
        1.0f / sqrtf(block_sum<C>(squares, scratch) / static_cast<float>(n) + kNormEps);
    for (unsigned i = threadIdx.x; i < n; i += Grid<C>::threads) {
        xs[i] = xs[i] * scale * weight[i];
    }
    __syncthreads();
}

// The schedule of a matrix: the grid's warps split its rows into equal contiguous runs.
// Warp g takes rows [g * rows / warps, (g + 1) * rows / warps), so every row has exactly
// one warp.
struct Rows {
    unsigned begin, end;
};
template <MegakernelConfig C>
__device__ Rows my_rows(unsigned rows) {
    const unsigned warp = blockIdx.x * Grid<C>::warps_per_block + threadIdx.x / 32;
    return {warp * rows / Grid<C>::warps, (warp + 1) * rows / Grid<C>::warps};
}

// store(row, W[row] . xs) for each of this warp's rows of a rows x Cols matrix.
template <MegakernelConfig C, unsigned Cols, class Store>
__device__ void for_rows(const std::uint16_t* w, unsigned rows, const float* xs, Store store) {
    const Rows mine = my_rows<C>(rows);
    for (unsigned r = mine.begin; r < mine.end; ++r) {
        const float y = C::RowKernel::template dot<Cols>(w + std::size_t{r} * Cols, xs);
        if (threadIdx.x % 32 == 0) {
            store(r, y);
        }
    }
}

// How much of a softmax one block, or one warp, has summed for each query head of a group:
// the largest score it saw, and the sums of exp(score - max) and of exp(score - max) * V.
// Two of these merge by scaling each with exp(its max - the larger max).
struct AttnPartial {
    float max[kGroup];
    float sum[kGroup];
    float acc[kGroup][kHeadDim];
};

// The factor that brings a partial with largest score `max` to the scale of `to`. A
// partial that saw no position has max = -infinity and counts for nothing.
__device__ float rescale(float max, float to) {
    return max == -INFINITY ? 0.0f : expf(max - to);
}

// The first half of attention, by every block: this block's part of the softmax of one KV
// head's two query heads. The grid's warps split the cached positions among them, position
// j going to warp j % (members * warps per block) of the KV head's blocks. A warp reads the
// K and V of a position once, for both query heads, each lane holding kLaneDims
// dimensions. This step's own K and V come from shared memory, in the cache's precision.
template <MegakernelConfig C>
__device__ void attention_part(unsigned kv_head, unsigned member, const LayerTrace* in,
                               const float* q_norm, const float* k_norm,
                               typename C::Kv::Bits* k_cache, typename C::Kv::Bits* v_cache,
                               const float* rope, unsigned pos, AttnPartial* mine,
                               float* scratch) {
    using Kv = typename C::Kv;
    constexpr unsigned kWarpsPerBlock = Grid<C>::warps_per_block;
    __shared__ float q[kGroup][kHeadDim];
    __shared__ float k[kHeadDim];
    __shared__ float v[kHeadDim];
    __shared__ float warp_max[kWarpsPerBlock][kGroup];
    __shared__ float warp_sum_[kWarpsPerBlock][kGroup];
    __shared__ float warp_acc[kWarpsPerBlock][kGroup][kHeadDim];
    const unsigned   t    = threadIdx.x;
    const unsigned   slot = kv_head * kHeadDim;
    const float      n    = static_cast<float>(kHeadDim);

    // q_norm of both query heads and k_norm, then RoPE.
    float squares[kGroup + 1] = {};
    if (t < kHeadDim) {
        for (unsigned h = 0; h < kGroup; ++h) {
            q[h][t]    = fresh(in->Qcur + (kv_head * kGroup + h) * kHeadDim + t);
            squares[h] = q[h][t] * q[h][t];
        }
        k[t]            = fresh(in->Kcur + slot + t);
        v[t]            = fresh(in->Vcur + slot + t);
        squares[kGroup] = k[t] * k[t];
    }
    float scale[kGroup + 1];
    for (unsigned h = 0; h <= kGroup; ++h) {
        scale[h] = 1.0f / sqrtf(block_sum<C>(squares[h], scratch) / n + kNormEps);
    }
    if (t < kHeadDim) {
        for (unsigned h = 0; h < kGroup; ++h) {
            q[h][t] = q[h][t] * scale[h] * q_norm[t];
        }
        k[t] = k[t] * scale[kGroup] * k_norm[t];
    }
    __syncthreads();
    if (t < kHalf) { // element t rotates with element t + kHalf
        const float c = rope[t];
        const float r = rope[kHalf + t];
        for (unsigned h = 0; h < kGroup; ++h) {
            const float q0  = q[h][t], q1 = q[h][t + kHalf];
            q[h][t]         = q0 * c - q1 * r;
            q[h][t + kHalf] = q0 * r + q1 * c;
        }
        const float k0 = k[t], k1 = k[t + kHalf];
        k[t]           = k0 * c - k1 * r;
        k[t + kHalf]   = k0 * r + k1 * c;
    }
    __syncthreads();

    // Later steps read this position from the cache, so this step uses the same rounded
    // values. One block per KV head writes the cache: one writer per slot.
    if (t < kHeadDim) {
        const typename Kv::Bits k_bits = Kv::store(k[t]);
        const typename Kv::Bits v_bits = Kv::store(v[t]);
        if (member == 0) {
            k_cache[std::size_t{pos} * kKvPerPos + slot + t] = k_bits;
            v_cache[std::size_t{pos} * kKvPerPos + slot + t] = v_bits;
        }
        k[t] = Kv::load(k_bits);
        v[t] = Kv::load(v_bits);
    }
    __syncthreads();

    // An online softmax over this warp's positions, for both query heads.
    const unsigned warp   = t / 32;
    const unsigned dim    = (t % 32) * kLaneDims; // this lane's first dimension
    const float    scaled = 1.0f / sqrtf(n);
    float          max[kGroup];
    float          sum[kGroup] = {};
    float          acc[kGroup][kLaneDims] = {};
    for (unsigned h = 0; h < kGroup; ++h) {
        max[h] = -INFINITY;
    }
    for (unsigned j = member * kWarpsPerBlock + warp; j <= pos; j += kMembers<C> * kWarpsPerBlock) {
        float kj[kLaneDims];
        float vj[kLaneDims];
        for (unsigned d = 0; d < kLaneDims; ++d) {
            const std::size_t at = std::size_t{j} * kKvPerPos + slot + dim + d;
            kj[d]                = j < pos ? Kv::load(k_cache[at]) : k[dim + d];
            vj[d]                = j < pos ? Kv::load(v_cache[at]) : v[dim + d];
        }
        for (unsigned h = 0; h < kGroup; ++h) {
            float dot = 0;
            for (unsigned d = 0; d < kLaneDims; ++d) {
                dot += kj[d] * q[h][dim + d];
            }
            const float score = warp_sum(dot) * scaled;
            float       weight = 1.0f;
            if (score > max[h]) { // a new maximum: bring what is summed so far to its scale
                const float shrink = expf(max[h] - score);
                max[h]             = score;
                sum[h] *= shrink;
                for (unsigned d = 0; d < kLaneDims; ++d) {
                    acc[h][d] *= shrink;
                }
            } else {
                weight = expf(score - max[h]);
            }
            sum[h] += weight;
            for (unsigned d = 0; d < kLaneDims; ++d) {
                acc[h][d] += weight * vj[d];
            }
        }
    }
    for (unsigned h = 0; h < kGroup; ++h) {
        for (unsigned d = 0; d < kLaneDims; ++d) {
            warp_acc[warp][h][dim + d] = acc[h][d];
        }
        if (t % 32 == 0) {
            warp_max[warp][h]  = max[h];
            warp_sum_[warp][h] = sum[h];
        }
    }
    __syncthreads();

    // Merge the block's warps. Thread t has query head t / kHeadDim, dimension t % kHeadDim.
    const unsigned h = t / kHeadDim;
    const unsigned i = t % kHeadDim;
    float          block_max_ = warp_max[0][h];
    for (unsigned w = 1; w < kWarpsPerBlock; ++w) {
        block_max_ = fmaxf(block_max_, warp_max[w][h]);
    }
    float total = 0;
    float value = 0;
    for (unsigned w = 0; w < kWarpsPerBlock; ++w) {
        const float f = rescale(warp_max[w][h], block_max_);
        total += f * warp_sum_[w][h];
        value += f * warp_acc[w][h][i];
    }
    mine->acc[h][i] = value;
    if (i == 0) {
        mine->max[h] = block_max_;
        mine->sum[h] = total;
    }
}

// The second half, by one block per KV head: merges the blocks' partials into kqv_out.
// parts is the first member's partial; member m's is kKvHeads further for each m.
template <MegakernelConfig C>
__device__ void attention_merge(unsigned kv_head, unsigned pos, const AttnPartial* parts,
                                LayerTrace* out) {
    __shared__ float max[kGroup][kMembers<C>];
    __shared__ float sum[kGroup][kMembers<C>];
    const unsigned   t = threadIdx.x;
    // Members past this one saw no position yet.
    const unsigned members = min(kMembers<C>, pos / Grid<C>::warps_per_block + 1);
    if (t < kGroup * members) {
        const AttnPartial& part = parts[(t / kGroup) * kKvHeads];
        max[t % kGroup][t / kGroup] = fresh(&part.max[t % kGroup]);
        sum[t % kGroup][t / kGroup] = fresh(&part.sum[t % kGroup]);
    }
    __syncthreads();
    const unsigned h      = t / kHeadDim;
    const unsigned i      = t % kHeadDim;
    float          largest = max[h][0];
    for (unsigned m = 1; m < members; ++m) {
        largest = fmaxf(largest, max[h][m]);
    }
    float total = 0;
    float value = 0;
    for (unsigned m = 0; m < members; ++m) {
        const float f = rescale(max[h][m], largest);
        total += f * sum[h][m];
        value += f * fresh(&parts[m * kKvHeads].acc[h][i]);
    }
    out->kqv_out[(kv_head * kGroup + h) * kHeadDim + i] = value / total;
}

// What a phase of one layer works on.
template <MegakernelConfig C>
struct Step {
    const std::byte*      base;
    const LayerOffsets*   w;       // where the layer's tensors start
    const LayerTrace*     in;      // what the phases read
    LayerTrace*           out;     // what they write: the same trace in a normal step
    const float*          x;       // the residual stream entering the layer
    typename C::Kv::Bits* k_cache; // the layer's cache
    typename C::Kv::Bits* v_cache;
    const float*          rope;     // this position's row of the table
    unsigned              pos;
    AttnPartial*          partials; // one per block
    float*                xs;       // shared memory: the vector of the phase's matrix products
    float*                scratch;  // shared memory: one value per warp
};

// The buffers of a layer that its phases pass to each other, one bit each.
enum Buffer : unsigned {
    kResidual = 1 << 0, // the stream entering the layer: the layer before wrote it
    kQkv      = 1 << 1, // Qcur, Kcur, Vcur
    kPartials = 1 << 2, // each block's part of the attention softmax
    kKqv      = 1 << 3, // kqv_out
    kFfnInp   = 1 << 4,
    kSwiglu   = 1 << 5,
    kLayerOut = 1 << 6, // l_out: the stream entering the next layer
};

// A phase is a type: what every block does between two barriers. reads and writes are
// sets of Buffer, and run does the work for the calling thread.
template <class P, class C>
concept Phase = requires(const Step<C>& s) {
    { P::reads } -> std::convertible_to<unsigned>;
    { P::writes } -> std::convertible_to<unsigned>;
    { P::run(s) } -> std::same_as<void>;
};

// Q, K and V of the normed residual stream.
struct QkvPhase {
    static constexpr unsigned reads  = kResidual;
    static constexpr unsigned writes = kQkv;
    template <MegakernelConfig C>
    __device__ static void run(const Step<C>& s) {
        LayerTrace* const a = s.out;
        load_normed<C>(s.x, f32_at(s.base, s.w->attn_norm), kEmbd, s.xs, s.scratch);
        for_rows<C, kEmbd>(bf16_at(s.base, s.w->q), kQ, s.xs,
                           [a](unsigned r, float y) { a->Qcur[r] = y; });
        for_rows<C, kEmbd>(bf16_at(s.base, s.w->k), kKvPerPos, s.xs,
                           [a](unsigned r, float y) { a->Kcur[r] = y; });
        for_rows<C, kEmbd>(bf16_at(s.base, s.w->v), kKvPerPos, s.xs,
                           [a](unsigned r, float y) { a->Vcur[r] = y; });
    }
};

// Every block sums its part of the softmax of one KV head's query heads.
struct AttentionPartPhase {
    static constexpr unsigned reads  = kQkv;
    static constexpr unsigned writes = kPartials;
    template <MegakernelConfig C>
    __device__ static void run(const Step<C>& s) {
        attention_part<C>(blockIdx.x % kKvHeads, blockIdx.x / kKvHeads, s.in,
                          f32_at(s.base, s.w->q_norm), f32_at(s.base, s.w->k_norm), s.k_cache,
                          s.v_cache, s.rope, s.pos, s.partials + blockIdx.x, s.scratch);
    }
};

// One block per KV head merges the parts into kqv_out.
struct AttentionMergePhase {
    static constexpr unsigned reads  = kPartials;
    static constexpr unsigned writes = kKqv;
    template <MegakernelConfig C>
    __device__ static void run(const Step<C>& s) {
        if (blockIdx.x < kKvHeads) {
            attention_merge<C>(blockIdx.x, s.pos, s.partials + blockIdx.x, s.out);
        }
    }
};

// The output projection, added to the residual stream.
struct OutputPhase {
    static constexpr unsigned reads  = kKqv | kResidual;
    static constexpr unsigned writes = kFfnInp;
    template <MegakernelConfig C>
    __device__ static void run(const Step<C>& s) {
        LayerTrace* const  a = s.out;
        const float* const x = s.x;
        load<C>(s.in->kqv_out, kQ, s.xs);
        for_rows<C, kQ>(bf16_at(s.base, s.w->o), kEmbd, s.xs,
                        [a, x](unsigned r, float y) { a->ffn_inp[r] = y + fresh(x + r); });
    }
};

// Gate and up of the normed stream, through SwiGLU. A warp does both rows.
struct FfnPhase {
    static constexpr unsigned reads  = kFfnInp;
    static constexpr unsigned writes = kSwiglu;
    template <MegakernelConfig C>
    __device__ static void run(const Step<C>& s) {
        load_normed<C>(s.in->ffn_inp, f32_at(s.base, s.w->ffn_norm), kEmbd, s.xs, s.scratch);
        const std::uint16_t* gate = bf16_at(s.base, s.w->gate);
        const std::uint16_t* up   = bf16_at(s.base, s.w->up);
        const Rows           mine = my_rows<C>(kFf);
        for (unsigned r = mine.begin; r < mine.end; ++r) {
            const float g = C::RowKernel::template dot<kEmbd>(gate + std::size_t{r} * kEmbd, s.xs);
            const float u = C::RowKernel::template dot<kEmbd>(up + std::size_t{r} * kEmbd, s.xs);
            if (threadIdx.x % 32 == 0) {
                s.out->ffn_swiglu[r] = g / (1.0f + expf(-g)) * u;
            }
        }
    }
};

// Down, added to the residual stream.
struct DownPhase {
    static constexpr unsigned reads  = kSwiglu | kFfnInp;
    static constexpr unsigned writes = kLayerOut;
    template <MegakernelConfig C>
    __device__ static void run(const Step<C>& s) {
        LayerTrace* const       a = s.out;
        const LayerTrace* const i = s.in;
        load<C>(i->ffn_swiglu, kFf, s.xs);
        for_rows<C, kFf>(bf16_at(s.base, s.w->down), kEmbd, s.xs, [a, i](unsigned r, float y) {
            a->l_out[r] = y + fresh(i->ffn_inp + r);
        });
    }
};

template <class... Ps>
struct PhaseList {
    static constexpr unsigned count = sizeof...(Ps);
};

// The schedule of one layer: its phases, in the order every block runs them. To swap a
// phase, or to reorder, change this list.
using LayerPhases = PhaseList<QkvPhase, AttentionPartPhase, AttentionMergePhase, OutputPhase,
                              FfnPhase, DownPhase>;

// A barrier ends every phase, so a phase may read only what an earlier phase of the layer
// wrote, or the stream that entered the layer. This is that rule, plus one writer per
// buffer, plus a layer that does write its output.
template <class... Ps>
constexpr bool well_ordered(PhaseList<Ps...>) {
    unsigned written = kResidual;
    bool     ok      = true;
    ((ok = ok && (Ps::reads & ~written) == 0 && (Ps::writes & written) == 0,
      written |= Ps::writes),
     ...);
    return ok && (written & kLayerOut) != 0;
}
static_assert(well_ordered(LayerPhases{}),
              "a phase reads a buffer no earlier phase wrote, or two phases write one buffer");

// One barrier after the embedding, one after each phase of each layer, one after the logits.
constexpr unsigned kBarriers = 2 + LayerPhases::count * static_cast<unsigned>(n_layer);

// Runs the phases of one layer, with a barrier after each.
template <MegakernelConfig C, class... Ps>
    requires(Phase<Ps, C> && ...)
__device__ void run_layer(const Step<C>& s, MegaSync<C>* sync, unsigned& arrived,
                          PhaseList<Ps...>) {
    ((Ps::run(s), barrier<C>(sync, arrived += Grid<C>::blocks)), ...);
}

// One decode step. out is the trace the phases write, in the trace they read: the same one
// in a normal step. The launch bounds make the compiler fit the grid's blocks on one SM.
template <MegakernelConfig C>
__global__ void __launch_bounds__(C::threads, C::blocks_per_sm)
megakernel(Weights weights, Trace* out, const Trace* in, typename C::Kv::Bits* k_cache,
           typename C::Kv::Bits* v_cache, const float* rope, AttnPartial* partials,
           MegaSync<C>* sync, unsigned token, unsigned pos, unsigned epoch) {
    static_assert(grid_fits_model<C>, "this grid cannot run this model");
    constexpr unsigned kBlocks        = Grid<C>::blocks;
    constexpr unsigned kThreads       = Grid<C>::threads;
    constexpr unsigned kWarpsPerBlock = Grid<C>::warps_per_block;
    // static: without it every thread rebuilds the table on its stack.
    static constexpr Offsets offsets = kOffsets;
    __shared__ float         xs[kFf];
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
    barrier<C>(sync, arrived += kBlocks);

    const float* x = in->embd; // the residual stream
    for (unsigned l = 0; l < n_layer; ++l) {
        const std::size_t cache = std::size_t{l} * C::n_ctx * kKvPerPos;
        const Step<C>     s{base,     &offsets.layer[l], &in->layers[l], &out->layers[l],
                            x,        k_cache + cache,   v_cache + cache, rope,
                            pos,      partials,          xs,              scratch};
        run_layer(s, sync, arrived, LayerPhases{});
        x = in->layers[l].l_out;
    }

    // The logits of the normed stream, and each block's best one. A later row wins only if
    // it is strictly larger, so the first maximum wins, as in llama.cpp's greedy sampler.
    load_normed<C>(x, f32_at(base, offsets.output_norm), kEmbd, xs, scratch);
    {
        __shared__ float     warp_logit[kWarpsPerBlock];
        __shared__ unsigned  warp_token[kWarpsPerBlock];
        const std::uint16_t* w    = bf16_at(base, offsets.output);
        const Rows           mine = my_rows<C>(kVocab);
        float                best = -INFINITY;
        unsigned             at   = mine.begin;
        for (unsigned r = mine.begin; r < mine.end; ++r) {
            const float y = C::RowKernel::template dot<kEmbd>(w + std::size_t{r} * kEmbd, xs);
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
    barrier<C>(sync, arrived += kBlocks);

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

template <MegakernelConfig C>
std::variant<Token, CudaError> launch(Megakernel<C>& m, Decoder<C>& d, Token token,
                                      const Trace* in) {
    assert(d.n_past < C::n_ctx);
    megakernel<C><<<Grid<C>::blocks, Grid<C>::threads>>>(d.weights, d.trace.get(), in, d.k_cache.get(),
                                         d.v_cache.get(),
                                         d.rope.get() + std::size_t{d.n_past} * head_dim,
                                         static_cast<AttnPartial*>(m.partials.get()),
                                         m.sync.get(), token.id, d.n_past, m.epoch);
    MK_TRY(cudaGetLastError());
    // failed and token, read together. The copy waits for the kernel.
    unsigned result[2];
    MK_TRY(cudaMemcpy(result, &m.sync->failed, sizeof result, cudaMemcpyDeviceToHost));
    if (result[0] != 0) {
        return CudaError{"megakernel", "a barrier gave up waiting for the rest of the grid"};
    }
    m.epoch += kBarriers * Grid<C>::blocks;
    ++d.n_past;
    return *Token::checked(result[1]); // a row of the output matrix: inside the vocabulary
}
} // namespace

template <MegakernelConfig C>
std::variant<Megakernel<C>, CudaError> Megakernel<C>::make() {
    // A barrier waits for every block, so every block must be running. That holds only if
    // each SM can keep its share of the grid resident.
    int resident = 0;
    MK_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, megakernel<C>,
                                                         static_cast<int>(C::threads), 0));
    if (resident < static_cast<int>(C::blocks_per_sm)) {
        return CudaError{"megakernel", "the GPU cannot keep the whole grid resident"};
    }
    Megakernel m;
    MK_TRY(device_new(m.sync, 1));
    MK_TRY(cudaMemset(m.sync.get(), 0, sizeof(MegaSync<C>)));
    void* partials = nullptr;
    MK_TRY(cudaMalloc(&partials, Grid<C>::blocks * sizeof(AttnPartial)));
    m.partials.reset(partials);
    return m;
}

template <MegakernelConfig C>
std::variant<Token, CudaError> Megakernel<C>::decode(Decoder<C>& d, Token token) {
    return launch(*this, d, token, d.trace.get());
}

template <MegakernelConfig C>
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
