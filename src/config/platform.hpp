#pragma once
#include <cstdint>

// Facts about CUDA that hold on every Nvidia GPU, so they belong to no GPU type. Each is named
// once here and used wherever the number appears. nvcc compiles this header: C++20 only.
namespace mk {
// Lanes in a warp.
inline constexpr unsigned warp_size = 32;
// The mask of __shfl_xor_sync and its kin when every lane of the warp takes part.
inline constexpr std::uint32_t full_warp_mask = 0xffffffffu;
// What a block's __shared__ variables may take together, in bytes. Only dynamic shared memory,
// which a kernel opts into, may use more of an SM.
inline constexpr unsigned max_static_shared_bytes = 48 * 1024;
} // namespace mk
