#include "gguf/reader.hpp"

#include <print>
#include <string_view>
#include <variant>

namespace {
void report(const char* path, gguf::Error e) {
    std::println(stderr, "{}: {} at byte {}", path, gguf::describe(e.kind), e.pos);
}
} // namespace

// gguf_architecture <in.gguf>: prints the file's general.architecture, or nothing when it has
// none. CMake runs it at configure time to pick the model type, so the build reads the
// untrusted file through the same reader as gguf2hpp.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::println(stderr, "usage: gguf_architecture <in.gguf>");
        return 2;
    }
    const char* in   = argv[1];
    const auto  file = gguf::MappedFile::open(in);
    if (!file) {
        report(in, file.error());
        return 1;
    }
    const auto model = gguf::parse(file->bytes());
    if (!model) {
        report(in, model.error());
        return 1;
    }
    for (const gguf::KeyValue& kv : model->metadata) {
        if (kv.key == "general.architecture") {
            const auto* name = std::get_if<std::string_view>(&kv.val);
            if (name == nullptr) {
                std::println(stderr, "{}: general.architecture is not a string", in);
                return 1;
            }
            // CMake drops a NUL, so it would read another name than the file's.
            if (name->contains('\0')) {
                std::println(stderr, "{}: general.architecture holds a NUL byte", in);
                return 1;
            }
            std::print("{}", *name);
        }
    }
    return 0;
}
