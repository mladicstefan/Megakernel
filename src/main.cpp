#include "model.hpp"
#include "reader.hpp"
#include "weights.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <print>
#include <string>
#include <utility>
#include <variant>

namespace {
// A mapped GGUF that is the file this engine was compiled for: the same size and the same
// header hash as model.hpp. The tensor data is not hashed. open() is the only place that
// makes one, so holding one means the checks passed.
struct VerifiedFile {
    gguf::MappedFile file;

    mk::HostWeights weights() const {
        return file.bytes().subspan(model::data_start).first<mk::weights_size>();
    }

    // The error is the message, without the path. A reader error reads as gguf2hpp prints it.
    static std::expected<VerifiedFile, std::string> open(const char* path) {
        const auto reader_error = [](gguf::Error e) {
            return std::unexpected(std::format("{} at byte {}", gguf::describe(e.kind), e.pos));
        };
        auto file = gguf::MappedFile::open(path);
        if (!file) {
            return reader_error(file.error());
        }
        // Size before parse: a cut-off copy is a file of the wrong size, whatever parse
        // would say about it.
        if (file->bytes().size() != model::file_size) {
            return std::unexpected(
                std::format("file size {} differs from the {} this engine was built for",
                            file->bytes().size(),
                            model::file_size));
        }
        const auto parsed = gguf::parse(file->bytes());
        if (!parsed) {
            return reader_error(parsed.error());
        }
        if (parsed->hash != model::hash) {
            return std::unexpected(std::format(
                "header hash {:#018x} differs from the {:#018x} this engine was built for",
                parsed->hash,
                model::hash));
        }
        return VerifiedFile{std::move(*file)};
    }
};

int cuda_failed(const mk::CudaError& e) {
    std::println(stderr, "engine: {}: {}", e.call, e.text);
    return 1;
}

constexpr double kBytesPerGB = 1e9;
constexpr double kMsPerS     = 1e3;
} // namespace

// engine [model.gguf]: without an argument, the file the engine was configured with.
int main(int argc, char** argv) {
    if (argc > 2) {
        std::println(stderr, "usage: engine [model.gguf]");
        return 2;
    }
    const char* path = argc == 2 ? argv[1] : MODEL_GGUF;

    std::println("{}: {} layers, {} tensors", model::meta::general::architecture,
                 model::meta::arch::block_count, model::tensors.size());

    const auto file = VerifiedFile::open(path);
    if (!file) {
        std::println(stderr, "engine: {}: {}", path, file.error());
        return 1;
    }
    std::println("file: {} verified ({} bytes, header hash {:#018x})", path, model::file_size,
                 model::hash);

    const auto uploaded = mk::upload(file->weights());
    if (const auto* e = std::get_if<mk::CudaError>(&uploaded)) {
        return cuda_failed(*e);
    }
    const mk::Upload& up = std::get<mk::Upload>(uploaded);
    std::println("upload: {} bytes in {:.3f} s, {:.2f} GB/s", mk::weights_size, up.seconds,
                 static_cast<double>(mk::weights_size) / up.seconds / kBytesPerGB);

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
