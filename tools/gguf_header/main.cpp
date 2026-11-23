#include "emitter.hpp"
#include "reader.hpp"

#include <bit>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <string_view>
#include <system_error>

static_assert(std::endian::native == std::endian::little);

namespace {
std::string_view name(gguf::ErrorKind kind) {
    using enum gguf::ErrorKind;
    switch (kind) {
    case OpenFailed:         return "cannot open (missing, not a regular file, or mmap failed)";
    case Truncated:          return "truncated";
    case BadMagic:           return "not a GGUF file (bad magic)";
    case UnsupportedVersion: return "unsupported GGUF version";
    case UnknownValueType:   return "unknown metadata value type";
    case BadBool:            return "bool other than 0 or 1";
    case NestedArray:        return "nested array";
    case UnknownTensorType:  return "unknown tensor type";
    case TooManyDims:        return "more than 4 dims";
    case BadShape:           return "first dim not a whole number of blocks";
    case BadAlignment:       return "bad general.alignment";
    case BadTensorExtent:    return "tensor data runs past the end of the file";
    case SizeOverflow:       return "tensor size overflows u64";
    }
    return "unknown error";
}

void report(const char* path, gguf::Error e) {
    std::println(stderr, "gguf2hpp: {}: {} at byte {}", path, name(e.kind), e.pos);
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
    const std::string text = gguf::emit(*model);

    // Write a temp file, then rename over out: a failed run never leaves a half-written
    // header that make would take as up to date.
    std::filesystem::path tmp = out;
    tmp += ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    f << text;
    f.close();
    if (!f) {
        std::println(stderr, "gguf2hpp: cannot write {}", tmp.string());
        return 1;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, out, ec);
    if (ec) {
        std::println(stderr, "gguf2hpp: cannot rename {} to {}: {}", tmp.string(), out.string(),
                     ec.message());
        return 1;
    }
    return 0;
}
