#include "weights/weights.hpp"

#include "base/cuda_try.hpp"

#include <chrono>
#include <cstdint>
#include <utility>

namespace mk {
namespace {
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// 4 blocks of 256 threads per SM. Measured on the 4070 Ti SUPER: 1 block reads at 422 GB/s,
// 4 at 623, 8 at 615 (docs/PLAN.md, Phase B).
constexpr unsigned kBlocks = BuildGpu::sms * 4;
constexpr unsigned kThreads     = 256; // a multiple of 32: add_checksum reduces whole warps

// Each tensor's offset and size in words. The kernel indexes this with a run-time tensor
// number, which it cannot do with model::tensors itself (a std::array of host types).
struct Extents {
    std::uint64_t offset[tensor_count];
    std::uint64_t words[tensor_count];
};

constexpr Extents make_extents() {
    Extents e{};
    for (std::size_t i = 0; i < tensor_count; ++i) {
        e.offset[i] = model::tensors[i].offset;
        e.words[i]  = model::tensors[i].bytes / 8;
    }
    return e;
}

// At namespace scope: a kernel may copy this, but may not call make_extents itself.
constexpr Extents kExtents = make_extents();

// Adds this thread's part of the checksum of n_words words at data into *out. Thread i of T
// takes words i, i + T, i + 2T and so on. Every warp adds its total with one atomic.
__device__ void add_checksum(const std::byte* data, std::uint64_t n_words, std::uint64_t* out) {
    // Aligned: weights.hpp asserts it for the offsets and upload checks the base.
    const auto*         words     = reinterpret_cast<const std::uint64_t*>(data);
    const std::uint64_t n_threads = std::uint64_t{gridDim.x} * blockDim.x;
    std::uint64_t       sum       = 0;
    for (std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < n_words;
         i += n_threads) {
        sum += checksum_term(words[i], i);
    }
    for (unsigned lanes = 16; lanes > 0; lanes /= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, lanes);
    }
    // A warp with no words of this tensor has sum 0 and skips the atomic.
    if (threadIdx.x % 32 == 0 && sum != 0) {
        static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));
        atomicAdd(reinterpret_cast<unsigned long long*>(out), static_cast<unsigned long long>(sum));
    }
}

// Every tensor, at base + its compile-time offset.
__global__ void checksum_kernel(Weights w, std::uint64_t* sums) {
    // static: without it every thread rebuilds the table on its stack (model.hpp explains).
    static constexpr Extents e = kExtents;
    for (std::size_t t = 0; t < tensor_count; ++t) {
        add_checksum(w.base + e.offset[t], e.words[t], sums + t);
    }
}

// One tensor through the per-layer table, read the way model.hpp documents: a constexpr
// local copy of layers[I].
template <std::size_t I>
__global__ void layer_q_kernel(Weights w, std::uint64_t* sum) {
    constexpr auto L = model::layers[I];
    add_checksum(w.base + L.q.offset, L.q.bytes / 8, sum);
}

// A template only so that a model without a layer table never instantiates the kernel.
template <std::size_t Count>
cudaError_t launch_layer_q(Weights w, std::uint64_t* sum) {
    if constexpr (Count > 0) {
        layer_q_kernel<Count - 1><<<1, kThreads>>>(w, sum);
        return cudaGetLastError();
    } else {
        return cudaSuccess;
    }
}
} // namespace

void CudaFree::operator()(void* p) const noexcept {
    cudaFree(p); // a deleter has nowhere to report a failure
}

std::variant<Upload, CudaError> upload(HostWeights data) {
    void* p = nullptr;
    MK_TRY(cudaMalloc(&p, data.size()));
    DeviceWeights weights;
    weights.mem.reset(static_cast<std::byte*>(p));
    if (reinterpret_cast<std::uintptr_t>(p) % model::alignment != 0) {
        return CudaError{"cudaMalloc", "the allocation is not aligned to general.alignment"};
    }

    // A plain copy from the pageable mapping. Staging through pinned buffers was within 3% of
    // it, and this GPU cannot register a read-only mapping (docs/PLAN.md, Phase B).
    const auto start = Clock::now();
    MK_TRY(cudaMemcpy(p, data.data(), data.size(), cudaMemcpyHostToDevice));
    // A copy from pageable memory may return before the last transfer has landed.
    MK_TRY(cudaDeviceSynchronize());
    return Upload{std::move(weights), seconds_since(start)};
}

std::variant<unsigned, CudaError> device_sms() {
    int device = 0;
    int sms    = 0;
    MK_TRY(cudaGetDevice(&device));
    MK_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return static_cast<unsigned>(sms);
}

std::variant<Checksums, CudaError> device_checksums(Weights w) {
    // One sum per tensor, then the layer probe's.
    constexpr std::size_t kSumBytes = (tensor_count + 1) * sizeof(std::uint64_t);
    void*                 p         = nullptr;
    MK_TRY(cudaMalloc(&p, kSumBytes));
    const std::unique_ptr<std::uint64_t, CudaFree> sums(static_cast<std::uint64_t*>(p));
    MK_TRY(cudaMemset(sums.get(), 0, kSumBytes));
    MK_TRY(cudaDeviceSynchronize()); // the memset is not part of the kernel's time

    Checksums  out{};
    const auto start = Clock::now();
    checksum_kernel<<<kBlocks, kThreads>>>(w, sums.get());
    MK_TRY(cudaGetLastError());
    MK_TRY(cudaDeviceSynchronize());
    out.seconds = seconds_since(start);

    MK_TRY(launch_layer_q<layer_count>(w, sums.get() + tensor_count));
    MK_TRY(cudaMemcpy(out.tensors.data(), sums.get(), tensor_count * sizeof(std::uint64_t),
                      cudaMemcpyDeviceToHost));
    MK_TRY(cudaMemcpy(&out.layer_q, sums.get() + tensor_count, sizeof(std::uint64_t),
                      cudaMemcpyDeviceToHost));
    return out;
}
} // namespace mk
