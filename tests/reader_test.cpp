#include "reader.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
