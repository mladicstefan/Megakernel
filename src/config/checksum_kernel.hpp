#pragma once
#include <concepts>

// How the weights' checksum kernel is launched. It is a type, like the row kernel, so a
// build can name another launch without a change to the kernel. nvcc compiles this header:
// C++20 only.
namespace mk {
// threads is the size of a block, a whole number of warps (weights.cu asserts it), and the
// grid is blocks_per_sm blocks on each SM of the GPU.
template <class K>
concept ChecksumKernel = requires {
    { K::threads } -> std::convertible_to<unsigned>;
    { K::blocks_per_sm } -> std::convertible_to<unsigned>;
};

// Measured on the RTX 4070 Ti SUPER: 1 block per SM reads at 422 GB/s, 4 at 623, 8 at 615
// (docs/PLAN.md, Phase B).
struct Checksum256x4 {
    static constexpr unsigned threads       = 256;
    static constexpr unsigned blocks_per_sm = 4;
};
static_assert(ChecksumKernel<Checksum256x4>);
} // namespace mk
