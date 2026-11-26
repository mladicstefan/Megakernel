#pragma once
#include "config/gpu.hpp"
#include "weights/weights.hpp"

#include <expected>
#include <format>
#include <string>
#include <variant>

// Host only (C++23): nvcc never includes this. megakernel.hpp declares the type.
namespace mk {
// The CUDA device in use, which has G's SM count: the count a grid for G is sized for. On
// another device the megakernel's barriers wait for blocks that never run. check() is the
// only place that makes one, so holding one means the check passed.
template <GpuSpec G>
struct VerifiedDevice {
    // The error is the message a user sees.
    static std::expected<VerifiedDevice, std::string> check() {
        const auto sms = device_sms();
        if (const auto* e = std::get_if<CudaError>(&sms)) {
            return std::unexpected(std::format("{}: {}", e->call, e->text));
        }
        if (std::get<unsigned>(sms) != G::sms) {
            return std::unexpected(std::format("built for the {} ({} SMs), this GPU has {} SMs",
                                               G::name, G::sms, std::get<unsigned>(sms)));
        }
        return VerifiedDevice{};
    }

private:
    VerifiedDevice() = default;
};
} // namespace mk
