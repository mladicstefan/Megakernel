#include "reader.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

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
