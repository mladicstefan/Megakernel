#pragma once
#include "config/config.hpp"

// The configuration this build is for: the model type that CMake's table names for the
// GGUF's general.architecture, on the GPU that MK_GPU names. The apps and the decoders'
// explicit instantiations name it. Only a build with a decoder has one. nvcc compiles this
// header: C++20 only.
#if !defined(MK_MODEL) || !defined(MK_MODEL_HEADER)
#error "MK_MODEL is not defined: CMake names a model type only for an architecture with a decoder"
#endif
#include MK_MODEL_HEADER

namespace mk {
using BuildModel = MK_MODEL;
static_assert(ModelSpec<BuildModel>);
using BuildConfig = EngineConfig<BuildModel, BuildGpu>;
static_assert(MegakernelConfig<BuildConfig>);
static_assert(DecoderConfig<LlamaCpuConfig<BuildModel>>);
} // namespace mk
