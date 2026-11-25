#include "reader.hpp"

#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <xxhash.h>
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

std::string_view describe(ErrorKind kind) {
    using enum ErrorKind;
    switch (kind) {
    case OpenFailed:
        return "cannot open (missing, not a regular file, or mmap failed)";
    case Truncated:
        return "truncated";
    case BadMagic:
        return "not a GGUF file (bad magic)";
    case UnsupportedVersion:
        return "unsupported GGUF version";
    case UnknownValueType:
        return "unknown metadata value type";
    case BadBool:
        return "bool other than 0 or 1";
    case NestedArray:
        return "nested array";
    case UnknownTensorType:
        return "unknown tensor type";
    case TooManyDims:
        return "more than 4 dims";
    case BadShape:
        return "first dim not a whole number of blocks";
    case BadAlignment:
        return "bad general.alignment";
    case BadTensorExtent:
        return "tensor data runs past the end of the file";
    case SizeOverflow:
        return "tensor size overflows u64";
    case DuplicateKey:
        return "duplicate metadata key";
    case DuplicateTensor:
        return "duplicate tensor name";
    }
    return "unknown error";
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

// From gguf-py's GGML_QUANT_SIZES. Ids that are not listed, including the gaps, give nullopt.
std::optional<BlockLayout> block_layout(TensorType t) {
    switch (t) {
    case TensorType::F32:
        return BlockLayout{1, 4};
    case TensorType::F16:
        return BlockLayout{1, 2};
    case TensorType::Q4_0:
        return BlockLayout{32, 18};
    case TensorType::Q4_1:
        return BlockLayout{32, 20};
    case TensorType::Q5_0:
        return BlockLayout{32, 22};
    case TensorType::Q5_1:
        return BlockLayout{32, 24};
    case TensorType::Q8_0:
        return BlockLayout{32, 34};
    case TensorType::Q8_1:
        return BlockLayout{32, 40};
    case TensorType::Q2_K:
        return BlockLayout{256, 84};
    case TensorType::Q3_K:
        return BlockLayout{256, 110};
    case TensorType::Q4_K:
        return BlockLayout{256, 144};
    case TensorType::Q5_K:
        return BlockLayout{256, 176};
    case TensorType::Q6_K:
        return BlockLayout{256, 210};
    case TensorType::Q8_K:
        return BlockLayout{256, 292};
    case TensorType::IQ2_XXS:
        return BlockLayout{256, 66};
    case TensorType::IQ2_XS:
        return BlockLayout{256, 74};
    case TensorType::IQ3_XXS:
        return BlockLayout{256, 98};
    case TensorType::IQ1_S:
        return BlockLayout{256, 50};
    case TensorType::IQ4_NL:
        return BlockLayout{32, 18};
    case TensorType::IQ3_S:
        return BlockLayout{256, 110};
    case TensorType::IQ2_S:
        return BlockLayout{256, 82};
    case TensorType::IQ4_XS:
        return BlockLayout{256, 136};
    case TensorType::I8:
        return BlockLayout{1, 1};
    case TensorType::I16:
        return BlockLayout{1, 2};
    case TensorType::I32:
        return BlockLayout{1, 4};
    case TensorType::I64:
        return BlockLayout{1, 8};
    case TensorType::F64:
        return BlockLayout{1, 8};
    case TensorType::IQ1_M:
        return BlockLayout{256, 56};
    case TensorType::BF16:
        return BlockLayout{1, 2};
    case TensorType::TQ1_0:
        return BlockLayout{256, 54};
    case TensorType::TQ2_0:
        return BlockLayout{256, 66};
    case TensorType::MXFP4:
        return BlockLayout{32, 17};
    case TensorType::NVFP4:
        return BlockLayout{64, 36};
    case TensorType::Q1_0:
        return BlockLayout{128, 18};
    }
    return std::nullopt;
}

std::expected<TensorInfo, Error> read_tensor_info(Cursor& c) {
    const std::size_t start = c.pos;
    TensorInfo        t{};
    t.dims.fill(1); // unused dims are 1, as in ggml

    auto name = c.read_string();
    if (!name) {
        return std::unexpected(name.error());
    }
    t.name = *name;

    const std::size_t n_dims_at = c.pos;

    auto n_dims = c.read<std::uint32_t>();
    if (!n_dims) {
        return std::unexpected(n_dims.error());
    }
    if (*n_dims > kMaxDims) {
        return std::unexpected(Error{ErrorKind::TooManyDims, n_dims_at});
    }
    t.n_dims = *n_dims;
    for (std::uint32_t d = 0; d < t.n_dims; ++d) {
        auto dim = c.read<std::uint64_t>();
        if (!dim) {
            return std::unexpected(dim.error());
        }
        t.dims[d] = *dim;
    }

    // The enum has a fixed underlying type, so any u32 casts safely; block_layout is the check.
    const std::size_t type_at = c.pos;
    auto              type    = c.read<std::uint32_t>();
    if (!type) {
        return std::unexpected(type.error());
    }
    t.dtype           = static_cast<TensorType>(*type);
    const auto layout = block_layout(t.dtype);
    if (!layout) {
        return std::unexpected(Error{ErrorKind::UnknownTensorType, type_at});
    }

    auto offset = c.read<std::uint64_t>();
    if (!offset) {
        return std::unexpected(offset.error());
    }
    t.offset = *offset;

    // Blocks run along the first dim, so each row must hold whole blocks.
    if (t.dims[0] % layout->elements != 0) {
        return std::unexpected(Error{ErrorKind::BadShape, start});
    }
    t.element_count = 1;
    for (const std::uint64_t d : t.dims) {
        if (__builtin_mul_overflow(t.element_count, d, &t.element_count)) {
            return std::unexpected(Error{ErrorKind::SizeOverflow, start});
        }
    }
    const std::uint64_t blocks = t.element_count / layout->elements;
    if (__builtin_mul_overflow(blocks, std::uint64_t{layout->bytes}, &t.byte_size)) {
        return std::unexpected(Error{ErrorKind::SizeOverflow, start});
    }
    return t;
}

// Sorts a copy and compares neighbours. Returns one name of an equal pair, still a view into
// the file, or nullopt when all names differ.
std::optional<std::string_view> find_duplicate(std::vector<std::string_view> names) {
    std::ranges::sort(names);
    const auto it = std::ranges::adjacent_find(names);
    if (it == names.end()) {
        return std::nullopt;
    }
    return *it;
}

// Where the string s (a view into file made by read_string) starts, counting its u64 length.
std::size_t string_pos(std::span<const std::byte> file, std::string_view s) {
    return static_cast<std::size_t>(reinterpret_cast<const std::byte*>(s.data()) - file.data())
           - sizeof(std::uint64_t);
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
    model.header    = *header;
    model.alignment = kDefaultAlignment;

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
        const std::size_t value_at = c.pos;
        auto              val      = read_value(c, *type);
        if (!val) {
            return std::unexpected(val.error());
        }
        if (*key == "general.alignment") {
            // The spec says u32; any unsigned width is accepted. has_single_bit(0) is false.
            const auto* a = std::get_if<std::uint64_t>(&*val);
            if (!a || !std::has_single_bit(*a)) {
                return std::unexpected(Error{ErrorKind::BadAlignment, value_at});
            }
            model.alignment = *a;
        }
        model.metadata.push_back(KeyValue{*key, *type, *val});
    }
    if (const auto dup = find_duplicate(
            std::ranges::to<std::vector>(model.metadata | std::views::transform(&KeyValue::key)))) {
        return std::unexpected(Error{ErrorKind::DuplicateKey, string_pos(file, *dup)});
    }

    // Same as kv_count: no reserve, each info consumes bytes.
    for (std::uint64_t i = 0; i < model.header.tensor_count; ++i) {
        auto info = read_tensor_info(c);
        if (!info) {
            return std::unexpected(info.error());
        }
        model.tensors.push_back(*info);
    }
    if (const auto dup = find_duplicate(std::ranges::to<std::vector>(
            model.tensors | std::views::transform(&TensorInfo::name)))) {
        return std::unexpected(Error{ErrorKind::DuplicateTensor, string_pos(file, *dup)});
    }

    // No overflow: pos < 2^63 (file sizes are off_t) and alignment <= 2^63.
    model.data_start = (c.pos + model.alignment - 1) & ~(model.alignment - 1);

    // Subtraction only, as in the Cursor. data_start may pass the end only if there are no
    // tensors. The error points at data_start: Error has no room to say which tensor.
    for (const auto& t : model.tensors) {
        if (model.data_start
            > model.file_size
            || t.offset
            > model.file_size
            - model.data_start
            || t.byte_size
            > model.file_size
            - model.data_start
            - t.offset) {
            return std::unexpected(Error{ErrorKind::BadTensorExtent, model.data_start});
        }
    }

    // The seed mixes in file_size, so one call covers [0, data_start) and the size.
    // With no tensors data_start can pass the end, so stop at the end of the file.
    const std::size_t hashed = std::min(model.data_start, model.file_size);
    model.hash               = XXH3_64bits_withSeed(file.data(), hashed, model.file_size);

    return model;
}
} // namespace gguf
