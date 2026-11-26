#pragma once

// Marks a function that both host code and CUDA device code call.
#ifdef __CUDACC__
#define MK_HOST_DEVICE __host__ __device__
#else
#define MK_HOST_DEVICE
#endif
