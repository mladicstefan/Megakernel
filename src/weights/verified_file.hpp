#pragma once
#include "gguf/reader.hpp"
#include "model.hpp"
#include "weights/weights.hpp"

#include <expected>
#include <format>
#include <string>
#include <utility>

// Host only (C++23): nvcc never includes this.
namespace mk {
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
} // namespace mk
