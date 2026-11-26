#pragma once
#include <concepts>
#include <cstdint>
#include <string_view>

// The GPUs a build can be compiled for, one hand-written type each, and the rule that finds
// which of them the driver's name for a card is. Apart from gpu.hpp so that the configure-time
// probe (apps/gpu_probe) can include it without a BuildGpu. nvcc compiles this header: C++20
// only.
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
    // shortcut: measured with Qwen3-0.6B's 256 threads a block, which the model derives;
    // measure again for a model whose n_head / n_head_kv * head_dim differs.
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

// Every type above, for the probe: a GPU type is added here and nowhere else.
#define MK_GPU_CATALOGUE(X) X(Rtx4070TiSuper) X(Rtx5070)

// Whether the driver's name for a card is the catalogue's name for it. The driver prefixes
// the brand ("NVIDIA GeForce RTX 4070 Ti SUPER"), so the catalogue's name must end the
// driver's, after a space. A prefix is not enough: the RTX 4070 Ti is not the SUPER.
constexpr bool same_card(std::string_view driver, std::string_view catalogue) {
    if (catalogue.empty() || !driver.ends_with(catalogue)) {
        return false;
    }
    return driver.size() == catalogue.size() || driver[driver.size() - catalogue.size() - 1] == ' ';
}
} // namespace mk
