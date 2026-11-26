#pragma once
#include <concepts>
#include <cstdint>

// The GPUs a build can be compiled for. A GPU is a type, and the build names one with
// CMake's MK_GPU, so its SM count, architecture and capacity are constants in every kernel
// and schedule. nvcc compiles this header: C++20 only.
namespace mk {
// Who a GPU is.
template <class G>
concept GpuIdentity = requires {
    { G::name } -> std::convertible_to<const char*>;
    { G::sms } -> std::convertible_to<unsigned>;       // streaming multiprocessors
    { G::cuda_arch } -> std::convertible_to<unsigned>; // 89 is sm_89
};

// What it holds, in bytes.
template <class G>
concept GpuCapacity = requires {
    { G::device_memory } -> std::convertible_to<std::uint64_t>;
    { G::shared_per_sm } -> std::convertible_to<unsigned>;
    // What the driver keeps of shared_per_sm for each resident block.
    { G::shared_reserved_per_block } -> std::convertible_to<unsigned>;
};

template <class G>
concept GpuSpec = GpuIdentity<G> && GpuCapacity<G>;

// Read from cudaGetDeviceProperties on the development machine.
struct Rtx4070TiSuper {
    static constexpr const char*   name                      = "RTX 4070 Ti SUPER";
    static constexpr unsigned      sms                       = 66;
    static constexpr unsigned      cuda_arch                 = 89;
    static constexpr std::uint64_t device_memory             = 16764764160; // totalGlobalMem
    static constexpr unsigned      shared_per_sm             = 102400;
    static constexpr unsigned      shared_reserved_per_block = 1024;
    // The megakernel's blocks per SM. Measured over 141 positions: 2.27 ms per step with 2
    // blocks per SM, 2.29 with 1, 2.30 with 3, 2.43 with 4. More than 4 do not fit on an SM.
    static constexpr unsigned blocks_per_sm = 2;
};
static_assert(GpuSpec<Rtx4070TiSuper>);

// No such card has been run: everything but the SM count and the architecture is unmeasured.
struct Rtx5070 {
    static constexpr const char*   name      = "RTX 5070";
    static constexpr unsigned      sms       = 48;
    static constexpr unsigned      cuda_arch = 120;
    // Unmeasured: the 12 GB of the specification. A real card reports somewhat less.
    static constexpr std::uint64_t device_memory = std::uint64_t{12} << 30;
    // Unmeasured: the CUDA programming guide gives compute capability 12.x 100 KB per SM and
    // 99 KB per block, which leaves 1 KB per block.
    static constexpr unsigned shared_per_sm             = 102400;
    static constexpr unsigned shared_reserved_per_block = 1024;
    // Unmeasured: the RTX 4070 Ti SUPER's.
    static constexpr unsigned blocks_per_sm = 2;
};
static_assert(GpuSpec<Rtx5070>);

#ifndef MK_GPU
#error "MK_GPU is not defined: CMake's MK_GPU names the GPU this build is for"
#endif
// The GPU this build is for: the type CMake's MK_GPU names.
using BuildGpu = MK_GPU;
static_assert(GpuSpec<BuildGpu>);

#ifdef __CUDA_ARCH__
// CMake compiles for BuildGpu's architecture unless CMAKE_CUDA_ARCHITECTURES overrides it.
static_assert(__CUDA_ARCH__ == BuildGpu::cuda_arch * 10,
              "nvcc compiles device code for another CUDA architecture than MK_GPU's cuda_arch: "
              "leave CMAKE_CUDA_ARCHITECTURES unset, or set it to that architecture alone");
#endif
} // namespace mk
