#include "reader.hpp"

#include <sys/mman.h>
#include <sys/stat.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>

#include <fcntl.h>
#include <unistd.h>
// GGUF stores values little-endian; reader reads them in place without byte-swapping.
static_assert(std::endian::native == std::endian::little);

namespace gguf {
std::expected<std::string_view, Error> Cursor::read_string() {
    const std::size_t start = pos;

    auto len = read<std::uint64_t>();
    if (!len) {
        return std::unexpected(len.error());
    }

    if (*len > data.size() - pos) {
        pos = start;
        return std::unexpected(Error{ErrorKind::Truncated, start});
    }
    const std::string_view s{reinterpret_cast<const char*>(data.data() + pos), *len};
    pos += *len;
    return s;
}

std::expected<MappedFile, Error> MappedFile::open(const char* path) {
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return std::unexpected(Error{ErrorKind::OpenFailed, 0});
    }

    // A directory opens fine with O_RDONLY; only a regular file can be mapped.
    struct stat st{};
    bool        ok   = fstat(fd, &st) == 0 && S_ISREG(st.st_mode);
    std::size_t size = 0;
    void*       addr = nullptr;
    if (ok) {
        size = static_cast<std::size_t>(st.st_size);
    }
    // mmap with length 0 fails, so an empty file stays unmapped and parses as Truncated.
    if (ok && size > 0) {
        addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ok   = addr != MAP_FAILED;
    }
    ::close(fd); // the mapping outlives the descriptor
    if (!ok) {
        return std::unexpected(Error{ErrorKind::OpenFailed, 0});
    }

    MappedFile file;
    file.map.reset(static_cast<std::byte*>(addr));
    file.map.get_deleter().size = size;
    return file;
}

void Unmap::operator()(std::byte* p) const noexcept {
    munmap(p, size);
}
} // namespace gguf
