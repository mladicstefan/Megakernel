#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace gguf {
inline constexpr std::uint32_t kMagic            = 0x46554747;
inline constexpr std::uint32_t kVersion          = 3;
inline constexpr std::uint64_t kDefaultAlignment = 32;
inline constexpr std::uint32_t kMaxDims          = 4;

enum class ValueType : std::uint32_t {
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Uint32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7, // 1 byte: 0 or 1
    String  = 8, // u64 length + bytes, no terminator
    Array   = 9, // u32 element type + u64 count + elements
    Uint64  = 10,
    Int64   = 11,
    Float64 = 12,
};

enum class TensorType : std::uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
    MXFP4   = 39,
    NVFP4   = 40,
    Q1_0    = 41,
};

// A dtype stores 'elements' values in every 'bytes' bytes.
// byte_size of a tenosr = element_count / elements * bytes
struct BlockLayout {
    std::uint32_t elements;
    std::uint32_t bytes;
};

enum class ErrorKind {
    OpenFailed,
    Truncated,       // a read would pass the end of the file
    BadMagic,
    UnsupportedVersion,
    UnknownValueType,
    BadBool,         // a bool byte other than 0 or 1
    NestedArray,     // array of arrays: rejected, no model file uses them
    UnknownTensorType,
    TooManyDims,
    BadShape,        // first dim is not a whole number of the dtype's blocks
    BadAlignment,    // general.alignment is not an unsigned power of two
    BadTensorExtent, // offset + byte_size runs past the end of the file
    SizeOverflow,    // element count or byte size does not fit in u64
};

struct Error {
    ErrorKind   kind;
    std::size_t pos;
};

struct Header {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint64_t tensor_count;
    std::uint64_t kv_count;
};

// parse() reads the header in one read<Header>(), so it must match the file byte for byte.
static_assert(sizeof(Header) == 24);

struct ArrayRef {
    ValueType     element_type;
    std::uint64_t count;
    std::size_t   pos;
};

using MetaValue = std::variant<std::uint64_t, std::int64_t, double, bool, std::string_view,
                               ArrayRef>;

struct KeyValue {
    std::string_view key;
    ValueType        type;
    MetaValue        val;
};

struct TensorInfo {
    std::string_view                    name;
    std::uint32_t                       n_dims;
    std::array<std::uint64_t, kMaxDims> dims;
    TensorType                          dtype;
    std::uint64_t                       offset;
    std::uint64_t                       element_count;
    std::uint64_t                       byte_size;
};

// Every string_view and ArrayRef points into the mapped file.
// A Model is valid only while that mapping is alive.
struct Model {
    Header                  header;
    std::vector<KeyValue>   metadata;
    std::vector<TensorInfo> tensors;
    std::uint64_t           alignment;  // general.alignment, or kDefaultAlignment
    std::uint64_t           data_start; // end of tensor infos, rounded up to alignment
    std::uint64_t           file_size;
    std::uint64_t           hash;       // over bytes [0, data_start) and file_size
};

// Walks a byte span front to back. Invariant: pos <= data.size().
struct Cursor {
    std::span<const std::byte> data;
    std::size_t                pos = 0;

    // On failure pos is unchanged.
    template <class T>
    requires std::is_trivially_copyable_v<T>
    std::expected<T, Error> read() {
        // Not pos + sizeof(T) > size: the invariant keeps this from underflowing.
        if (data.size() - pos < sizeof(T)) {
            return std::unexpected(Error{ErrorKind::Truncated, pos});
        }

        T value;
        std::memcpy(&value, data.data() + pos, sizeof(T)); // pos may be unaligned
        pos += sizeof(T);

        return value;
    }

    // u64 length, then that many bytes. The view points into data.
    // On failure pos is unchanged and the error names where the string starts.
    std::expected<std::string_view, Error> read_string();
};

// unique_ptr deleter for an mmap; carries the length munmap needs.
struct Unmap {
    std::size_t size = 0;
    void        operator()(std::byte* p) const noexcept;
};

// Owns one read-only mmap of a file. unique_ptr makes it move-only and unmaps it once.
struct MappedFile {
    std::unique_ptr<std::byte, Unmap> map; // null when empty; an empty file is never mapped

    static std::expected<MappedFile, Error> open(const char* path);

    std::span<const std::byte> bytes() const {
        // A moved-from deleter keeps its size, so check the pointer, not the size.
        if (!map) {
            return {};
        }
        return {map.get(), map.get_deleter().size};
    }
};

// The returned Model points into file, so file must outlive it.
std::expected<Model, Error> parse(std::span<const std::byte> file);
} // namespace gguf
