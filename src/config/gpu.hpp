#pragma once
#include "config/gpu_catalogue.hpp"

// The GPU this build is for. CMake's MK_GPU picks it: a type in gpu_catalogue.hpp, the
// catalogue type of the GPU in this machine, or one the configure-time probe generated, so
// its SM count, architecture and capacity are constants in every kernel and schedule.
// build_gpu.hpp is the probe's output, in the build directory. nvcc compiles this header:
// C++20 only.
#include "build_gpu.hpp"

namespace mk {
#ifndef MK_GPU
#error "MK_GPU is not defined: CMake's MK_GPU names the GPU this build is for"
#endif
// The GPU this build is for: the type CMake's MK_GPU names.
using BuildGpu = MK_GPU;
static_assert(GpuSpec<BuildGpu>);

#ifdef __CUDA_ARCH__
// CMake compiles for BuildGpu's architecture alone.
static_assert(__CUDA_ARCH__ == BuildGpu::cuda_arch * 10,
              "nvcc compiles device code for another CUDA architecture than MK_GPU's cuda_arch: "
              "CMake sets that architecture itself, so look for an -arch or -gencode in "
              "CMAKE_CUDA_FLAGS");
#endif
} // namespace mk
