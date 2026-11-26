#include "gguf/emitter.hpp"
#include "gguf/reader.hpp"

#include <bit>
#include <filesystem>
#include <fstream>
#include <print>
#include <string_view>
#include <system_error>

static_assert(std::endian::native == std::endian::little);

namespace {
void report(const char* path, gguf::Error e) {
    std::println(stderr, "gguf2hpp: {}: {} at byte {}", path, gguf::describe(e.kind), e.pos);
}
} // namespace

// gguf2hpp <in.gguf> <out.hpp>
int main(int argc, char** argv) {
    if (argc != 3) {
        std::println(stderr, "usage: gguf2hpp <in.gguf> <out.hpp>");
        return 2;
    }
    const char*                 in  = argv[1];
    const std::filesystem::path out = argv[2];

    // Keep the mapping alive while model is used: model points into it.
    auto file = gguf::MappedFile::open(in);
    if (!file) {
        report(in, file.error());
        return 1;
    }
    auto model = gguf::parse(file->bytes());
    if (!model) {
        report(in, model.error());
        return 1;
    }
    // The file name only: the output must not change with the directory the model sits in.
    const auto text = gguf::emit(*model, std::filesystem::path(in).filename().string());
    if (!text) {
        std::println(stderr, "gguf2hpp: {}: {}", in, text.error());
        return 1;
    }

    // Write a temp file, then rename over out: a failed run never leaves a half-written
    // header that make would take as up to date.
    std::filesystem::path tmp = out;
    tmp += ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    f << *text;
    f.close();
    if (!f) {
        std::println(stderr, "gguf2hpp: cannot write {}", tmp.string());
        return 1;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, out, ec);
    if (ec) {
        std::println(stderr,
                     "gguf2hpp: cannot rename {} to {}: {}",
                     tmp.string(),
                     out.string(),
                     ec.message());
        return 1;
    }
    return 0;
}
