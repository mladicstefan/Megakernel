#pragma once
#include "checksum.hpp"
#include "gpu.hpp"
#include "model.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <variant>

// The model's tensor data on the GPU: one device allocation holding the file's bytes
// [data_start, file_size), so tensor t is at base + t.offset. nvcc compiles this header:
// C++20 only, so a result is a std::variant where host code would use std::expected.

// A model with no blk.<n>.* tensors has no model::layers. This gives the name a meaning
// there, an empty table, so layer_count works for every model. A table that model.hpp
// declares hides this one.
namespace mk::no_layers {
struct Layer {
    Tensor q;
};
inline constexpr std::array<Layer, 0> layers{};
} // namespace mk::no_layers
namespace model {
using namespace mk::no_layers;
}

namespace mk {
inline constexpr std::size_t tensor_count = model::tensors.size();
inline constexpr std::size_t layer_count  = std::size(model::layers);
inline constexpr std::size_t weights_size = model::file_size - model::data_start;

static_assert(tensor_count > 0, "the model has no tensors to load");
// The device base is aligned to general.alignment (upload checks), so with this every
// tensor's device address is too.
static_assert(std::ranges::all_of(model::tensors,
                                  [](const Tensor& t) { return t.offset % model::alignment == 0; }),
              "a tensor's offset is not a multiple of general.alignment");
// shortcut: the checksum reads aligned 8-byte words only, add a tail word when a model has a
// tensor that is not a whole number of them.
static_assert(model::alignment % 8 == 0 &&
                  std::ranges::all_of(model::tensors,
                                      [](const Tensor& t) { return t.bytes % 8 == 0; }),
              "a tensor is not a whole number of aligned 8-byte words");

// The tensor device_checksums also reads through model::layers: the last layer's q.
inline constexpr Tensor layer_probe = layer_count > 0 ? model::layers[layer_count - 1].q : Tensor{};

// The file's bytes [data_start, file_size) on the host. The fixed extent is the check: only
// a file of the size the engine was compiled for can make one.
using HostWeights = std::span<const std::byte, weights_size>;

// Where those bytes are on the device. Kernels take this by value.
struct Weights {
    const std::byte* base;
};

struct CudaFree {
    void operator()(void* p) const noexcept;
};

// Owns the device allocation.
struct DeviceWeights {
    std::unique_ptr<std::byte, CudaFree> mem;

    Weights view() const {
        return {mem.get()};
    }
};

// A failed CUDA call: the call as written, and CUDA's text for its error code.
struct CudaError {
    const char* call;
    const char* text;
};

struct Upload {
    DeviceWeights weights;
    double        seconds; // the host-to-device copy alone
};

// Allocates on the device and copies data there.
std::variant<Upload, CudaError> upload(HostWeights data);

// The SM count of the device in use. A build is compiled for BuildGpu::sms of them, and
// nothing may launch a kernel on a device with another count.
std::variant<unsigned, CudaError> device_sms();

struct Checksums {
    std::array<std::uint64_t, tensor_count> tensors; // in model::tensors order
    std::uint64_t                           layer_q; // of layer_probe; 0 without a layer table
    double                                  seconds; // the all-tensors kernel alone
};

// Checksums of the device copy, computed on the device.
std::variant<Checksums, CudaError> device_checksums(Weights w);

// The same checksums of the host copy.
inline std::array<std::uint64_t, tensor_count> host_checksums(HostWeights data) {
    std::array<std::uint64_t, tensor_count> sums;
    std::ranges::transform(model::tensors, sums.begin(), [&](const Tensor& t) {
        return checksum(data.subspan(t.offset, t.bytes));
    });
    return sums;
}
} // namespace mk
