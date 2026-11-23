#include "reader.hpp"

#include <sys/mman.h>
#include <sys/stat.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
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

namespace {
// Casting validates nothing, so check the id first. Value type ids run 0 to Float64 with no gaps.
std::expected<ValueType, Error> read_value_type(Cursor& c) {
    const std::size_t at = c.pos;
    auto              id = c.read<std::uint32_t>();
    if (!id) {
        return std::unexpected(id.error());
    }
    if (*id > static_cast<std::uint32_t>(ValueType::Float64)) {
        return std::unexpected(Error{ErrorKind::UnknownValueType, at});
    }
    return static_cast<ValueType>(*id);
}

// Reads a T from the file and stores it widened to the MetaValue alternative Wide.
template <class T, class Wide>
std::expected<MetaValue, Error> read_widened(Cursor& c) {
    auto v = c.read<T>();
    if (!v) {
        return std::unexpected(v.error());
    }
    return MetaValue{static_cast<Wide>(*v)};
}

std::expected<MetaValue, Error> read_value(Cursor& c, ValueType type);

// Records where the elements are, then walks them to find where the array ends.
std::expected<MetaValue, Error> read_array(Cursor& c) {
    const std::size_t type_at = c.pos;
    auto              elem    = read_value_type(c);
    if (!elem) {
        return std::unexpected(elem.error());
    }
    if (*elem == ValueType::Array) {
        return std::unexpected(Error{ErrorKind::NestedArray, type_at});
    }
    auto count = c.read<std::uint64_t>();
    if (!count) {
        return std::unexpected(count.error());
    }

    const ArrayRef ref{*elem, *count, c.pos};
    // Every element takes at least one byte, so a bogus count hits Truncated, not a long loop.
    for (std::uint64_t i = 0; i < *count; ++i) {
        auto v = read_value(c, *elem);
        if (!v) {
            return std::unexpected(v.error());
        }
    }
    return MetaValue{ref};
}

std::expected<MetaValue, Error> read_value(Cursor& c, ValueType type) {
    switch (type) {
    case ValueType::Uint8:
        return read_widened<std::uint8_t, std::uint64_t>(c);
    case ValueType::Uint16:
        return read_widened<std::uint16_t, std::uint64_t>(c);
    case ValueType::Uint32:
        return read_widened<std::uint32_t, std::uint64_t>(c);
    case ValueType::Uint64:
        return read_widened<std::uint64_t, std::uint64_t>(c);
    case ValueType::Int8:
        return read_widened<std::int8_t, std::int64_t>(c);
    case ValueType::Int16:
        return read_widened<std::int16_t, std::int64_t>(c);
    case ValueType::Int32:
        return read_widened<std::int32_t, std::int64_t>(c);
    case ValueType::Int64:
        return read_widened<std::int64_t, std::int64_t>(c);
    case ValueType::Float32:
        return read_widened<float, double>(c);
    case ValueType::Float64:
        return read_widened<double, double>(c);
    case ValueType::Bool: {
        // Read a byte, not a bool: a bool holding anything but 0 or 1 is undefined behaviour.
        const std::size_t at = c.pos;
        auto              b  = c.read<std::uint8_t>();
        if (!b) {
            return std::unexpected(b.error());
        }
        if (*b > 1) {
            return std::unexpected(Error{ErrorKind::BadBool, at});
        }
        return MetaValue{*b == 1};
    }
    case ValueType::String: {
        auto s = c.read_string();
        if (!s) {
            return std::unexpected(s.error());
        }
        return MetaValue{*s};
    }
    case ValueType::Array:
        return read_array(c);
    }
    return std::unexpected(Error{ErrorKind::UnknownValueType, c.pos}); // unreachable once validated
}
} // namespace

std::expected<Model, Error> parse(std::span<const std::byte> file) {
    Cursor c{file};
    Model  model{};
    model.file_size = file.size();

    auto header = c.read<Header>();
    if (!header) {
        return std::unexpected(header.error());
    }
    if (header->magic != kMagic) {
        return std::unexpected(Error{ErrorKind::BadMagic, offsetof(Header, magic)});
    }
    if (header->version != kVersion) {
        return std::unexpected(Error{ErrorKind::UnsupportedVersion, offsetof(Header, version)});
    }
    model.header = *header;

    // No reserve(kv_count): the count is untrusted, and each pair consumes bytes, so the loop
    // stops at Truncated.
    for (std::uint64_t i = 0; i < model.header.kv_count; ++i) {
        auto key = c.read_string();
        if (!key) {
            return std::unexpected(key.error());
        }
        auto type = read_value_type(c);
        if (!type) {
            return std::unexpected(type.error());
        }
        auto val = read_value(c, *type);
        if (!val) {
            return std::unexpected(val.error());
        }
        model.metadata.push_back(KeyValue{*key, *type, *val});
    }

    return model;
}
} // namespace gguf
