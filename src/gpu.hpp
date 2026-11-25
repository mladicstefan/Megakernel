#pragma once
#include <concepts>

// The GPU a build is compiled for. A GPU is a type, so the SM count is a constant in every
// kernel and schedule. nvcc compiles this header: C++20 only.
namespace mk {
template <class G>
concept GpuSpec = requires {
    { G::sms } -> std::convertible_to<unsigned>; // streaming multiprocessors
    { G::name } -> std::convertible_to<const char*>;
};

struct Rtx4070TiSuper {
    static constexpr unsigned    sms  = 66;
    static constexpr const char* name = "RTX 4070 Ti SUPER";
};

// The GPU this build targets.
using BuildGpu = Rtx4070TiSuper;
static_assert(GpuSpec<BuildGpu>);
} // namespace mk
