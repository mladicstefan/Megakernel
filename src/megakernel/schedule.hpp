#pragma once
#include "base/host_device.hpp"
#include "config/platform.hpp"

#include <cstdint>

// The megakernel's schedule and the proofs that it fits a model, a grid and a GPU. Each
// proof is a constexpr function over plain numbers or over phase types, so a host test can
// feed it failing input, and the megakernel asserts each one with its own message. No model,
// no GPU type and no CUDA header. nvcc compiles this header: C++20 only.
//
// A proof the kernel asserts is MK_HOST_DEVICE: nvcc refuses a call to a host function in
// device code, even inside a static_assert.
namespace mk {
// The schedule of a matrix: the grid's warps split its rows into contiguous runs of equal
// length, to within one row.
struct Rows {
    unsigned begin, end;
};
// The rows of a matrix that warp `warp` of `warps` takes.
MK_HOST_DEVICE constexpr Rows rows_of(unsigned warp, unsigned rows, unsigned warps) {
    return {warp * rows / warps, (warp + 1) * rows / warps};
}

// Whether split, called as split(warp, rows, warps), gives every row of a matrix exactly one
// warp: the first run starts at row 0, each run starts where the one before ended, and the
// last ends at the last row.
template <class Split>
MK_HOST_DEVICE constexpr bool partitions(Split split, unsigned rows, unsigned warps) {
    unsigned next = 0;
    for (unsigned warp = 0; warp < warps; ++warp) {
        const Rows run = split(warp, rows, warps);
        if (run.begin != next || run.end < run.begin) {
            return false;
        }
        next = run.end;
    }
    return next == rows;
}

// Whether rows_of's products for a matrix of `rows` fit in 32 bits: the largest is
// warps * rows.
MK_HOST_DEVICE constexpr bool split_fits_32_bits(unsigned rows, unsigned warps) {
    return std::uint64_t{warps} * rows < (std::uint64_t{1} << 32);
}

// Whether a row kernel that reads `step` weights at a time reads a row of `cols` in whole
// steps.
MK_HOST_DEVICE constexpr bool whole_steps(unsigned cols, unsigned step) {
    return cols % step == 0;
}

// Attention splits the grid by KV head. Whether each KV head gets at least one block.
MK_HOST_DEVICE constexpr bool each_kv_head_has_a_block(unsigned blocks, unsigned kv_heads) {
    return blocks >= kv_heads;
}

// Whether a block has one thread for each dimension of each query head of a group, which is
// how the attention merges split their work.
MK_HOST_DEVICE constexpr bool one_thread_per_group_dimension(unsigned threads, unsigned group,
                                                             unsigned head_dim) {
    return threads == group * head_dim;
}

// The megakernel's static shared memory in bytes, from the numbers that size its buffers,
// all of 4-byte values: xs and scratch; attention_part's q, k, v and its warps' max, sum and
// acc; attention_merge's max and sum of each member; the logits' best of each warp.
// Megakernel::make refuses a kernel whose shared memory is not this count.
MK_HOST_DEVICE constexpr std::uint64_t megakernel_shared_bytes(unsigned widest,
                                                               unsigned warps_per_block,
                                                               unsigned group, unsigned head_dim,
                                                               unsigned members) {
    const std::uint64_t warps  = warps_per_block;
    const std::uint64_t values = widest + warps + std::uint64_t{group + 2} * head_dim +
                                 warps * group * (2 + std::uint64_t{head_dim}) +
                                 2 * std::uint64_t{group} * members + 2 * warps;
    return sizeof(float) * values;
}

// Whether a block's static shared memory is within CUDA's limit for __shared__ variables, on
// every GPU.
MK_HOST_DEVICE constexpr bool fits_static_shared_memory(std::uint64_t block_bytes) {
    return block_bytes <= max_static_shared_bytes;
}

// Whether blocks_per_sm blocks of block_bytes static shared memory each are resident on one
// SM at once, with the driver keeping reserved_per_block of shared_per_sm for each.
MK_HOST_DEVICE constexpr bool fits_shared_memory(unsigned blocks_per_sm,
                                                 std::uint64_t block_bytes,
                                                 unsigned reserved_per_block,
                                                 unsigned shared_per_sm) {
    return blocks_per_sm * (block_bytes + reserved_per_block) <= shared_per_sm;
}

// Whether the weights and a KV cache of n_ctx positions fit in device_memory. It divides, so
// no product can wrap.
MK_HOST_DEVICE constexpr bool fits_device_memory(std::uint64_t weight_bytes,
                                                 std::uint64_t kv_bytes_per_position,
                                                 std::uint64_t n_ctx,
                                                 std::uint64_t device_memory) {
    return weight_bytes <= device_memory &&
           n_ctx <= (device_memory - weight_bytes) / kv_bytes_per_position;
}

// Whether the KV cache holds no more positions than the model's context length.
MK_HOST_DEVICE constexpr bool within_context_length(std::uint64_t n_ctx,
                                                    std::uint64_t context_length) {
    return n_ctx <= context_length;
}

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

// The schedule of one layer: its phases, in the order every block runs them. A phase is a
// type whose reads and writes are sets of Buffer.
template <class... Ps>
struct PhaseList {
    static constexpr unsigned count = sizeof...(Ps);
};

// A barrier ends every phase, so a phase may read only what an earlier phase of the layer
// wrote, or the stream that entered the layer.
template <class... Ps>
constexpr bool well_ordered(PhaseList<Ps...>) {
    unsigned written = kResidual;
    bool     ok      = true;
    ((ok = ok && (Ps::reads & ~written) == 0, written |= Ps::writes), ...);
    return ok;
}

// Whether no two phases write one buffer, and no phase writes the stream the layer before
// wrote.
template <class... Ps>
constexpr bool one_writer_per_buffer(PhaseList<Ps...>) {
    unsigned written = kResidual;
    bool     ok      = true;
    ((ok = ok && (Ps::writes & written) == 0, written |= Ps::writes), ...);
    return ok;
}

// Whether a phase writes l_out, the stream the next layer reads.
template <class... Ps>
constexpr bool writes_layer_output(PhaseList<Ps...>) {
    return (((Ps::writes & kLayerOut) != 0) || ...);
}

// The barriers of one decode step: one after the embedding, one after each phase of each
// layer, one after the logits.
template <class... Ps>
constexpr unsigned barriers(PhaseList<Ps...>, unsigned layers) {
    return 2 + PhaseList<Ps...>::count * layers;
}
} // namespace mk
