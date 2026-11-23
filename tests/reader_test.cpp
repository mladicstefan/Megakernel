#include "reader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <sys/mman.h>

using namespace gguf;

namespace {
template <class... Ts>
constexpr std::array<std::byte, sizeof...(Ts)> bytes(Ts... v) {
    return {static_cast<std::byte>(v)...};
}

// mincore fails with ENOMEM when any page in the range is unmapped.
bool is_mapped(void* addr, std::size_t size) {
    unsigned char page = 0; // one entry is enough: every mapping here fits in one page
    return mincore(addr, size, &page) == 0;
}
} // namespace

TEST(Cursor, ReadsLittleEndianAtUnalignedPos) {
    const auto b = bytes(0xAA, 0x78, 0x56, 0x34, 0x12);
    Cursor     c{b, 1};
    auto       r = c.read<std::uint32_t>();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 0x12345678u);
    EXPECT_EQ(c.pos, 5u);
}

TEST(Cursor, ReadUpToEndThenTruncated) {
    const auto b = bytes(1, 0, 0, 0);
    Cursor     c{b};
    ASSERT_TRUE(c.read<std::uint32_t>().has_value());
    auto r = c.read<std::uint8_t>();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::Truncated);
    EXPECT_EQ(r.error().pos, 4u);
    EXPECT_EQ(c.pos, 4u);
}

TEST(Cursor, ShortReadLeavesPos) {
    const auto b = bytes(1, 2, 3);
    Cursor     c{b};
    auto       r = c.read<std::uint32_t>();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::Truncated);
    EXPECT_EQ(r.error().pos, 0u);
    EXPECT_EQ(c.pos, 0u);
}

TEST(Cursor, ReadsString) {
    const auto b = bytes(5, 0, 0, 0, 0, 0, 0, 0, 'h', 'e', 'l', 'l', 'o');
    Cursor     c{b};
    auto       r = c.read_string();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, "hello");
    // A view into the buffer, not a copy.
    EXPECT_EQ(static_cast<const void*>(r->data()), static_cast<const void*>(b.data() + 8));
    EXPECT_EQ(c.pos, 13u);
}

TEST(Cursor, ReadsEmptyString) {
    const auto b = bytes(0, 0, 0, 0, 0, 0, 0, 0);
    Cursor     c{b};
    auto       r = c.read_string();
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->empty());
    EXPECT_EQ(c.pos, 8u);
}

TEST(Cursor, StringBodyCutOff) {
    const auto b = bytes(0xAA, 5, 0, 0, 0, 0, 0, 0, 0, 'h', 'e', 'l');
    Cursor     c{b, 1};
    auto       r = c.read_string();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::Truncated);
    EXPECT_EQ(r.error().pos, 1u); // where the string starts, not its body
    EXPECT_EQ(c.pos, 1u);
}

TEST(Cursor, StringLengthCutOff) {
    const auto b = bytes(5, 0, 0);
    Cursor     c{b};
    auto       r = c.read_string();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::Truncated);
    EXPECT_EQ(r.error().pos, 0u);
    EXPECT_EQ(c.pos, 0u);
}

// pos + len would wrap around to a small number and pass a naive check.
TEST(Cursor, HugeStringLengthIsTruncatedNotOverflow) {
    const auto b = bytes(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 'x');
    Cursor     c{b};
    auto       r = c.read_string();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, ErrorKind::Truncated);
    EXPECT_EQ(c.pos, 0u);
}

TEST(MappedFile, OpensToyFile) {
    auto f = MappedFile::open(TOY_GGUF);
    ASSERT_TRUE(f.has_value());
    EXPECT_EQ(f->bytes().size(), 384u);
    EXPECT_EQ(std::memcmp(f->bytes().data(), "GGUF", 4), 0);
    // The Cursor reads the same bytes back as the magic number.
    Cursor c{f->bytes()};
    auto   magic = c.read<std::uint32_t>();
    ASSERT_TRUE(magic.has_value());
    EXPECT_EQ(*magic, kMagic);
}

TEST(MappedFile, MissingFileFails) {
    auto f = MappedFile::open("/nonexistent/toy.gguf");
    ASSERT_FALSE(f.has_value());
    EXPECT_EQ(f.error().kind, ErrorKind::OpenFailed);
}

TEST(MappedFile, DirectoryFails) {
    auto f = MappedFile::open("/");
    ASSERT_FALSE(f.has_value());
    EXPECT_EQ(f.error().kind, ErrorKind::OpenFailed);
}

TEST(MappedFile, DeviceFails) {
    auto f = MappedFile::open("/dev/null");
    ASSERT_FALSE(f.has_value());
    EXPECT_EQ(f.error().kind, ErrorKind::OpenFailed);
}

TEST(MappedFile, DestructorUnmaps) {
    void* addr = nullptr;
    {
        auto f = MappedFile::open(TOY_GGUF);
        ASSERT_TRUE(f.has_value());
        addr = f->map.get();
        EXPECT_TRUE(is_mapped(addr, 384));
    }
    EXPECT_FALSE(is_mapped(addr, 384));
}

TEST(MappedFile, MoveConstructTransfersOwnership) {
    auto src = MappedFile::open(TOY_GGUF);
    ASSERT_TRUE(src.has_value());
    const auto* data = src->bytes().data();

    MappedFile dst = std::move(*src);
    EXPECT_TRUE(src->bytes().empty());
    EXPECT_EQ(dst.bytes().data(), data);
    EXPECT_EQ(dst.bytes().size(), 384u);
}

TEST(MappedFile, MoveAssignUnmapsOldMapping) {
    auto src = MappedFile::open(TOY_GGUF);
    auto dst = MappedFile::open(TOY_GGUF);
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());
    void*       old  = dst->map.get();
    const auto* data = src->bytes().data();

    *dst = std::move(*src);
    EXPECT_FALSE(is_mapped(old, 384)); // otherwise dst's first mapping leaked
    EXPECT_TRUE(src->bytes().empty());
    EXPECT_EQ(dst->bytes().data(), data);
}

namespace {
// Built from the struct, so these only test the checks; ParsesToyHeader tests the layout.
std::array<std::byte, sizeof(Header)> header_bytes(std::uint32_t magic, std::uint32_t version) {
    return std::bit_cast<std::array<std::byte, sizeof(Header)>>(Header{magic, version, 0, 0});
}
} // namespace

TEST(Parse, ParsesToyHeader) {
    auto f = MappedFile::open(TOY_GGUF);
    ASSERT_TRUE(f.has_value());
    auto m = parse(f->bytes());
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(m->header.magic, kMagic);
    EXPECT_EQ(m->header.version, 3u);
    EXPECT_EQ(m->header.tensor_count, 2u);
    EXPECT_EQ(m->header.kv_count, 6u);
    EXPECT_EQ(m->file_size, 384u);
}

TEST(Parse, RejectsBadMagic) {
    const auto b = header_bytes(0x58554747, kVersion); // "GGUX"
    auto       m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::BadMagic);
    EXPECT_EQ(m.error().pos, 0u);
}

TEST(Parse, RejectsOtherVersions) {
    for (const std::uint32_t version : {1u, 2u, 4u}) {
        const auto b = header_bytes(kMagic, version);
        auto       m = parse(b);
        ASSERT_FALSE(m.has_value());
        EXPECT_EQ(m.error().kind, ErrorKind::UnsupportedVersion);
        EXPECT_EQ(m.error().pos, 4u);
    }
}

TEST(Parse, TruncatedHeader) {
    const auto b = header_bytes(kMagic, kVersion);
    for (const std::size_t n : {std::size_t{0}, std::size_t{4}, sizeof(Header) - 1}) {
        auto m = parse(std::span(b).first(n));
        ASSERT_FALSE(m.has_value());
        EXPECT_EQ(m.error().kind, ErrorKind::Truncated);
        EXPECT_EQ(m.error().pos, 0u);
    }
}

TEST(Parse, ReadsToyMetadata) {
    auto f = MappedFile::open(TOY_GGUF);
    ASSERT_TRUE(f.has_value());
    auto m = parse(f->bytes());
    ASSERT_TRUE(m.has_value()) << "error kind " << static_cast<int>(m.error().kind) << " at "
                               << m.error().pos;
    const auto& kv = m->metadata;
    ASSERT_EQ(kv.size(), 6u);

    EXPECT_EQ(kv[0].key, "general.architecture");
    EXPECT_EQ(kv[0].type, ValueType::String);
    EXPECT_EQ(std::get<std::string_view>(kv[0].val), "toy");

    EXPECT_EQ(kv[1].key, "toy.block_count");
    EXPECT_EQ(kv[1].type, ValueType::Uint32);
    EXPECT_EQ(std::get<std::uint64_t>(kv[1].val), 2u);

    EXPECT_EQ(kv[2].key, "toy.scale");
    EXPECT_EQ(kv[2].type, ValueType::Float32);
    EXPECT_EQ(std::get<double>(kv[2].val), 0.5);

    EXPECT_EQ(kv[3].key, "toy.note");
    EXPECT_EQ(std::get<std::string_view>(kv[3].val), "hello");

    EXPECT_EQ(kv[4].key, "toy.flag");
    EXPECT_EQ(kv[4].type, ValueType::Bool);
    EXPECT_TRUE(std::get<bool>(kv[4].val));

    EXPECT_EQ(kv[5].key, "toy.list");
    EXPECT_EQ(kv[5].type, ValueType::Array);
    const auto ref = std::get<ArrayRef>(kv[5].val);
    EXPECT_EQ(ref.element_type, ValueType::Int32);
    EXPECT_EQ(ref.count, 3u);
    // ref.pos points at the first element.
    Cursor c{f->bytes(), ref.pos};
    for (const std::int32_t want : {10, 20, 30}) {
        auto got = c.read<std::int32_t>();
        ASSERT_TRUE(got.has_value());
        EXPECT_EQ(*got, want);
    }
}

namespace {
// Appends v's bytes as the file stores them (little-endian, checked in reader.cpp).
template <class T>
void put(std::vector<std::byte>& out, T v) {
    const auto b = std::bit_cast<std::array<std::byte, sizeof(T)>>(v);
    out.insert(out.end(), b.begin(), b.end());
}

void put_string(std::vector<std::byte>& out, std::string_view s) {
    put<std::uint64_t>(out, s.size());
    for (const char ch : s) {
        out.push_back(static_cast<std::byte>(ch));
    }
}

// A header announcing kv_count pairs, then key "k" and a type id. The caller adds the value.
std::vector<std::byte> kv_file(std::uint64_t kv_count, std::uint32_t type) {
    std::vector<std::byte> out;
    put(out, Header{kMagic, kVersion, 0, kv_count});
    put_string(out, "k");
    put(out, type);
    return out;
}

constexpr std::size_t kFirstValueAt = sizeof(Header) + 8 + 1 + 4; // header, key, type id

std::uint32_t id(ValueType t) {
    return static_cast<std::uint32_t>(t);
}
} // namespace

TEST(Parse, RejectsUnknownValueType) {
    const auto b = kv_file(1, 13);
    auto       m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::UnknownValueType);
    EXPECT_EQ(m.error().pos, kFirstValueAt - 4);
}

TEST(Parse, RejectsBoolOtherThanZeroOrOne) {
    auto b = kv_file(1, id(ValueType::Bool));
    put<std::uint8_t>(b, 2);
    auto m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::BadBool);
    EXPECT_EQ(m.error().pos, kFirstValueAt);
}

TEST(Parse, RejectsUnknownElementTypeEvenWhenEmpty) {
    auto b = kv_file(1, id(ValueType::Array));
    put<std::uint32_t>(b, 99);
    put<std::uint64_t>(b, 0);
    auto m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::UnknownValueType);
    EXPECT_EQ(m.error().pos, kFirstValueAt);
}

TEST(Parse, RejectsNestedArray) {
    auto b = kv_file(1, id(ValueType::Array));
    put(b, id(ValueType::Array));
    put<std::uint64_t>(b, 0);
    auto m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::NestedArray);
    EXPECT_EQ(m.error().pos, kFirstValueAt);
}

// The walk must step over each string by its own length to land on the next pair.
TEST(Parse, WalksArrayOfStrings) {
    auto b = kv_file(2, id(ValueType::Array));
    put(b, id(ValueType::String));
    put<std::uint64_t>(b, 2);
    put_string(b, "ab");
    put_string(b, "c");
    put_string(b, "next");
    put(b, id(ValueType::Uint8));
    put<std::uint8_t>(b, 7);

    auto m = parse(b);
    ASSERT_TRUE(m.has_value());
    ASSERT_EQ(m->metadata.size(), 2u);
    const auto ref = std::get<ArrayRef>(m->metadata[0].val);
    EXPECT_EQ(ref.element_type, ValueType::String);
    EXPECT_EQ(ref.count, 2u);
    EXPECT_EQ(ref.pos, kFirstValueAt + 4 + 8);
    EXPECT_EQ(m->metadata[1].key, "next");
    EXPECT_EQ(std::get<std::uint64_t>(m->metadata[1].val), 7u);
}

TEST(Parse, HugeArrayCountIsTruncated) {
    auto b = kv_file(1, id(ValueType::Array));
    put(b, id(ValueType::Int32));
    put<std::uint64_t>(b, UINT64_MAX);
    put<std::int32_t>(b, 1);
    auto m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::Truncated);
}

TEST(Parse, HugeKvCountIsTruncated) {
    const auto b = std::bit_cast<std::array<std::byte, sizeof(Header)>>(
        Header{kMagic, kVersion, 0, UINT64_MAX});
    auto m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::Truncated);
    EXPECT_EQ(m.error().pos, sizeof(Header));
}

namespace {
// The tensor that starts last ends at the end of the file, give or take alignment padding.
void expect_last_tensor_ends_file(const Model& m) {
    ASSERT_FALSE(m.tensors.empty());
    const auto          last = std::ranges::max_element(m.tensors, {}, &TensorInfo::offset);
    const std::uint64_t end  = m.data_start + last->offset + last->byte_size;
    EXPECT_LE(end, m.file_size);
    EXPECT_LT(m.file_size - end, m.alignment);
}

// No metadata and one tensor info named "t", then zero padding to 32 and data_bytes of data.
std::vector<std::byte> tensor_file(std::initializer_list<std::uint64_t> dims, std::uint32_t type,
                                   std::uint64_t offset, std::size_t data_bytes) {
    std::vector<std::byte> out;
    put(out, Header{kMagic, kVersion, 1, 0});
    put_string(out, "t");
    put(out, static_cast<std::uint32_t>(dims.size()));
    for (const std::uint64_t d : dims) {
        put(out, d);
    }
    put(out, type);
    put(out, offset);
    out.resize((out.size() + 31) / 32 * 32 + data_bytes);
    return out;
}

constexpr std::size_t kNDimsAt = sizeof(Header) + 8 + 1; // header, then the name "t"

std::uint32_t id(TensorType t) {
    return static_cast<std::uint32_t>(t);
}

// No tensors and one pair, general.alignment, with the given type id. The caller adds the value.
std::vector<std::byte> alignment_file(std::uint32_t type) {
    std::vector<std::byte> out;
    put(out, Header{kMagic, kVersion, 0, 1});
    put_string(out, "general.alignment");
    put(out, type);
    return out;
}

constexpr std::size_t kAlignmentAt = sizeof(Header) + 8 + 17 + 4; // header, key, type id
} // namespace

TEST(Parse, ReadsToyTensors) {
    auto f = MappedFile::open(TOY_GGUF);
    ASSERT_TRUE(f.has_value());
    auto m = parse(f->bytes());
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(m->alignment, 32u);
    EXPECT_EQ(m->data_start, 0x140u);
    ASSERT_EQ(m->tensors.size(), 2u);

    const auto& a = m->tensors[0];
    EXPECT_EQ(a.name, "a");
    EXPECT_EQ(a.n_dims, 2u);
    EXPECT_EQ(a.dims[0], 3u); // numpy shape (2, 3), stored in ggml order
    EXPECT_EQ(a.dims[1], 2u);
    EXPECT_EQ(a.dtype, TensorType::F32);
    EXPECT_EQ(a.offset, 0u);
    EXPECT_EQ(a.element_count, 6u);
    EXPECT_EQ(a.byte_size, 24u);

    const auto& b = m->tensors[1];
    EXPECT_EQ(b.name, "b");
    EXPECT_EQ(b.n_dims, 1u);
    EXPECT_EQ(b.dims[0], 4u);
    EXPECT_EQ(b.dims[1], 1u); // unused dims are 1
    EXPECT_EQ(b.dtype, TensorType::F16);
    EXPECT_EQ(b.offset, 32u);
    EXPECT_EQ(b.element_count, 4u);
    EXPECT_EQ(b.byte_size, 8u);

    expect_last_tensor_ends_file(*m);
}

// Expected values come from gguf-py's GGUFReader on the same file.
TEST(Parse, ReadsQwen) {
    if (!std::filesystem::exists(QWEN_GGUF)) {
        GTEST_SKIP() << QWEN_GGUF << " not found";
    }
    auto f = MappedFile::open(QWEN_GGUF);
    ASSERT_TRUE(f.has_value());
    auto m = parse(f->bytes());
    ASSERT_TRUE(m.has_value()) << "error kind " << static_cast<int>(m.error().kind) << " at "
                               << m.error().pos;

    EXPECT_EQ(m->metadata.size(), 27u);
    EXPECT_EQ(std::get<std::string_view>(m->metadata[0].val), "qwen3");
    EXPECT_EQ(m->tensors.size(), 311u);
    EXPECT_EQ(m->alignment, 32u); // general.alignment is absent
    EXPECT_EQ(m->data_start, 5951648u);

    const auto embd = std::ranges::find(m->tensors, "token_embd.weight", &TensorInfo::name);
    ASSERT_NE(embd, m->tensors.end());
    EXPECT_EQ(embd->n_dims, 2u);
    EXPECT_EQ(embd->dims[0], 1024u);
    EXPECT_EQ(embd->dims[1], 151936u);
    EXPECT_EQ(embd->dtype, TensorType::BF16);
    EXPECT_EQ(embd->offset, 311164928u);
    EXPECT_EQ(embd->byte_size, 311164928u);

    for (const auto& t : m->tensors) {
        EXPECT_TRUE(t.dtype == TensorType::BF16 || t.dtype == TensorType::F32) << t.name;
    }
    expect_last_tensor_ends_file(*m);
}

TEST(Parse, TensorFileHelperParses) {
    const auto b = tensor_file({4}, id(TensorType::F32), 0, 16);
    auto       m = parse(b);
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(m->tensors[0].byte_size, 16u);
    expect_last_tensor_ends_file(*m);
}

TEST(Parse, RejectsTooManyDims) {
    const auto b = tensor_file({1, 1, 1, 1, 1}, id(TensorType::F32), 0, 4);
    auto       m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::TooManyDims);
    EXPECT_EQ(m.error().pos, kNDimsAt);
}

TEST(Parse, RejectsTensorTypeInEnumGap) {
    const auto b = tensor_file({4}, 4, 0, 16); // 4 and 5 were removed from ggml
    auto       m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::UnknownTensorType);
    EXPECT_EQ(m.error().pos, kNDimsAt + 4 + 8);
}

TEST(Parse, RejectsPartialBlock) {
    const auto b = tensor_file({31}, id(TensorType::Q4_0), 0, 18); // Q4_0 blocks hold 32
    auto       m = parse(b);
    ASSERT_FALSE(m.has_value());
    EXPECT_EQ(m.error().kind, ErrorKind::BadShape);
    EXPECT_EQ(m.error().pos, sizeof(Header));
}

TEST(Parse, QuantizedByteSize) {
    const auto b = tensor_file({64, 2}, id(TensorType::Q4_0), 0, 4 * 18);
    auto       m = parse(b);
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(m->tensors[0].element_count, 128u);
    EXPECT_EQ(m->tensors[0].byte_size, 4u * 18u); // 4 blocks of 32 elements, 18 bytes each
}

TEST(Parse, RejectsSizeOverflow) {
    // The element count overflows, then the byte size: 2^62 F64s fit, their 2^65 bytes do not.
    for (const auto& b : {tensor_file({1ull << 32, 1ull << 32}, id(TensorType::F32), 0, 0),
                          tensor_file({1ull << 62}, id(TensorType::F64), 0, 0)}) {
        auto m = parse(b);
        ASSERT_FALSE(m.has_value());
        EXPECT_EQ(m.error().kind, ErrorKind::SizeOverflow);
        EXPECT_EQ(m.error().pos, sizeof(Header));
    }
}

TEST(Parse, RejectsTensorPastEnd) {
    for (const auto& b : {tensor_file({4}, id(TensorType::F32), 0, 15),
                          tensor_file({4}, id(TensorType::F32), 8, 16),
                          tensor_file({4}, id(TensorType::F32), UINT64_MAX, 16)}) {
        auto m = parse(b);
        ASSERT_FALSE(m.has_value());
        EXPECT_EQ(m.error().kind, ErrorKind::BadTensorExtent);
    }
}

TEST(Parse, UsesGeneralAlignment) {
    auto b = alignment_file(id(ValueType::Uint32));
    put<std::uint32_t>(b, 64);
    auto m = parse(b);
    ASSERT_TRUE(m.has_value());
    EXPECT_EQ(m->alignment, 64u);
    EXPECT_EQ(m->data_start, 64u); // pairs end at 57
}

TEST(Parse, RejectsBadAlignment) {
    auto zero = alignment_file(id(ValueType::Uint32));
    put<std::uint32_t>(zero, 0);
    auto not_pow2 = alignment_file(id(ValueType::Uint32));
    put<std::uint32_t>(not_pow2, 48);
    auto negative = alignment_file(id(ValueType::Int32));
    put<std::int32_t>(negative, 32);
    auto text = alignment_file(id(ValueType::String));
    put_string(text, "32");

    for (const auto& b : {zero, not_pow2, negative, text}) {
        auto m = parse(b);
        ASSERT_FALSE(m.has_value());
        EXPECT_EQ(m.error().kind, ErrorKind::BadAlignment);
        EXPECT_EQ(m.error().pos, kAlignmentAt);
    }
}

namespace {
std::vector<std::byte> toy_copy() {
    auto f = MappedFile::open(TOY_GGUF);
    EXPECT_TRUE(f.has_value());
    return {f->bytes().begin(), f->bytes().end()};
}

std::uint64_t hash_of(std::span<const std::byte> b) {
    auto m = parse(b);
    EXPECT_TRUE(m.has_value());
    return m ? m->hash : 0;
}
} // namespace

// Recorded from this code, not an outside tool: it catches an accidental change to what is hashed.
TEST(Hash, ToyHashIsStable) {
    EXPECT_EQ(hash_of(toy_copy()), 0x2a6d1ca21af41e12u);
}

TEST(Hash, ChangesWithHeaderBytes) {
    auto                b      = toy_copy();
    const std::uint64_t before = hash_of(b);
    b[sizeof(Header) + 8] ^= std::byte{0x20}; // "general.architecture" -> "General.architecture"
    EXPECT_NE(hash_of(b), before);
}

TEST(Hash, ChangesWithFileSize) {
    auto                b      = toy_copy();
    const std::uint64_t before = hash_of(b);
    b.resize(b.size() + 32); // same header, the data just grows
    EXPECT_NE(hash_of(b), before);
}

// By design the weights are not hashed: only the header and the size.
TEST(Hash, IgnoresTensorData) {
    auto                b      = toy_copy();
    const std::uint64_t before = hash_of(b);
    b[0x140] ^= std::byte{0xFF}; // first byte of tensor "a"
    EXPECT_EQ(hash_of(b), before);
}
