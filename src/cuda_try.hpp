#pragma once
#include "weights.hpp"

#include <cuda_runtime.h>

// For .cu files only. Leaves the calling function with a mk::CudaError when a CUDA call
// fails. The function's return type must be constructible from one.
#define MK_TRY(call)                                                                              \
    do {                                                                                          \
        if (const cudaError_t mk_try_e = (call); mk_try_e != cudaSuccess) {                       \
            return ::mk::CudaError{#call, cudaGetErrorString(mk_try_e)};                          \
        }                                                                                         \
    } while (false)
