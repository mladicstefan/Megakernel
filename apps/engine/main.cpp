#include "decode/positions.hpp"
#include "model.hpp"
#include "weights/verified_device.hpp"
#include "weights/verified_file.hpp"
#include "weights/weights.hpp"
#ifdef MK_ENGINE_DECODES
#include "decode/greedy.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <charconv>
#include <optional>
#include <print>
#include <span>
#include <string_view>
#include <variant>

namespace {
int cuda_failed(const mk::CudaError& e) {
    std::println(stderr, "engine: {}: {}", e.call, e.text);
    return 1;
}

constexpr double kBytesPerGB = 1e9;
constexpr double kMsPerS     = 1e3;

constexpr std::string_view kUsage =
    "usage: engine [model.gguf] [--tokens <tokens.txt> [--generate <n>]]";

struct Options {
    const char* model    = MODEL_GGUF; // the file the engine was configured with
    const char* tokens   = nullptr;
    std::size_t generate = mk::default_generate;
};

std::optional<Options> parse_args(std::span<char*> args) {
    Options o;
    bool    model_given = false;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (a == "--tokens" || a == "--generate") {
            if (i + 1 == args.size()) {
                return std::nullopt;
            }
            const std::string_view v = args[++i];
            if (a == "--tokens") {
                o.tokens = v.data();
            } else if (const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(),
                                                              o.generate);
                       ec != std::errc{} || end != v.data() + v.size()) {
                return std::nullopt; // ec is set for no digits and for a value past SIZE_MAX
            }
        } else if (!a.starts_with("--") && !model_given) {
            o.model     = a.data();
            model_given = true;
        } else {
            return std::nullopt;
        }
    }
    return o;
}

using Device = mk::VerifiedDevice<mk::BuildGpu>;

// Decodes the prompt in the token file greedily through the megakernel.
int decode(const Device& device, mk::Weights weights, const Options& o) {
#ifdef MK_ENGINE_DECODES
    using C           = mk::BuildConfig;
    const auto prompt = mk::read_tokens<C::Model>(o.tokens);
    if (!prompt) {
        std::println(stderr, "engine: {}: {}", o.tokens, prompt.error());
        return 1;
    }
    const std::size_t steps = mk::positions(prompt->size(), o.generate);
    if (steps > C::n_ctx) {
        std::println(stderr,
                     "engine: {} positions are more than the {} this build's KV cache holds",
                     steps, C::n_ctx);
        return 1;
    }
    const auto g = mk::generate_greedy<C>(device, weights, *prompt, o.generate);
    if (!g) {
        std::println(stderr, "engine: {}", g.error());
        return 1;
    }
    mk::print_generated(g->tokens);
    mk::print_speed(*g);
    return 0;
#else
    static_cast<void>(device);
    static_cast<void>(weights);
    std::println(stderr, "engine: {}: this build cannot decode: it has no forward pass for a "
                         "'{}' model",
                 o.tokens, model::meta::general::architecture);
    return 1;
#endif
}
} // namespace

// Without --tokens the engine proves the upload: it checksums every tensor on the device
// and compares with the file. With --tokens it decodes the prompt and generates.
int main(int argc, char** argv) {
    const auto options = parse_args({argv, static_cast<std::size_t>(argc)});
    if (!options) {
        std::println(stderr, "{}", kUsage);
        return 2;
    }
    const char* path = options->model;

    std::println("{}: {} layers, {} tensors", model::meta::general::architecture,
                 model::meta::arch::block_count, model::tensors.size());

    const auto file = mk::VerifiedFile::open(path);
    if (!file) {
        std::println(stderr, "engine: {}: {}", path, file.error());
        return 1;
    }
    std::println("file: {} verified ({} bytes, header hash {:#018x})", path, model::file_size,
                 model::hash);

    // The kernels are compiled for one GPU's SM count. Refuse any other device.
    const auto gpu = Device::check();
    if (!gpu) {
        std::println(stderr, "engine: {}", gpu.error());
        return 1;
    }

    const auto uploaded = mk::upload(file->weights());
    if (const auto* e = std::get_if<mk::CudaError>(&uploaded)) {
        return cuda_failed(*e);
    }
    const mk::Upload& up = std::get<mk::Upload>(uploaded);
    std::println("upload: {} bytes in {:.3f} s, {:.2f} GB/s", mk::weights_size, up.seconds,
                 static_cast<double>(mk::weights_size) / up.seconds / kBytesPerGB);
    if (options->tokens != nullptr) {
        return decode(*gpu, up.weights.view(), *options);
    }

    const auto          host   = mk::host_checksums(file->weights());
    const std::uint64_t host_q = mk::checksum(
        file->weights().subspan(mk::layer_probe.offset, mk::layer_probe.bytes));

    // Every run must match. Run -1 is not timed: the first launch also loads the kernel.
    constexpr int             kRuns   = 5;
    std::array<double, kRuns> seconds{};
    std::size_t               matched = mk::tensor_count;
    bool                      q_ok    = true;
    for (int run = -1; run < kRuns; ++run) {
        const auto result = mk::device_checksums(up.weights.view());
        if (const auto* e = std::get_if<mk::CudaError>(&result)) {
            return cuda_failed(*e);
        }
        const mk::Checksums& device = std::get<mk::Checksums>(result);
        std::size_t          same   = 0;
        for (std::size_t i = 0; i < mk::tensor_count; ++i) {
            same += device.tensors[i] == host[i];
        }
        matched = std::min(matched, same);
        q_ok    = q_ok && (mk::layer_count == 0 || device.layer_q == host_q);
        if (run >= 0) {
            seconds[static_cast<std::size_t>(run)] = device.seconds;
        }
    }

    if (mk::layer_count > 0) {
        std::println("checksums: {}/{} tensors match, layers[{}].q {} through model::layers",
                     matched, mk::tensor_count, mk::layer_count - 1,
                     q_ok ? "matches" : "differs");
    } else {
        std::println("checksums: {}/{} tensors match", matched, mk::tensor_count);
    }

    std::ranges::sort(seconds);
    const double median = seconds[kRuns / 2];
    std::println("checksum kernel: {:.3f} ms, {:.0f} GB/s (median of {} runs, {:.3f} to {:.3f} "
                 "ms; a first measurement, not tuned)",
                 median * kMsPerS, static_cast<double>(mk::weights_size) / median / kBytesPerGB,
                 kRuns, seconds.front() * kMsPerS, seconds.back() * kMsPerS);

    if (matched != mk::tensor_count || !q_ok) {
        std::println(stderr, "engine: {}: the device copy differs from the file", path);
        return 1;
    }
    return 0;
}
