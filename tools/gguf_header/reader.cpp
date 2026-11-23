#pragma once
#include "reader.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>
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
} // namespace gguf
