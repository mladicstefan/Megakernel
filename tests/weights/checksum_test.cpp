#include "weights/checksum.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include <gtest/gtest.h>

namespace {
// The bytes of words, as the checksum reads them from a file.
template <std::size_t N>
std::array<std::byte, N * 8> bytes_of(const std::array<std::uint64_t, N>& words) {
    std::array<std::byte, N * 8> bytes;
    std::memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
}

template <std::size_t N>
std::uint64_t checksum_of(const std::array<std::uint64_t, N>& words) {
    return mk::checksum(bytes_of(words));
}
} // namespace

// The term for word 0 at position 0 is splitmix64's first output for seed 0, a published
// value. Host and device share the term, so it is checked at compile time too.
static_assert(mk::checksum_term(0, 0) == 0xe220a8397b1dcdafu);

// A change to the definition must show up here. The second value was computed by a separate
// Python transcription of the term, not by this code.
TEST(Checksum, Stable) {
    EXPECT_EQ(checksum_of(std::array<std::uint64_t, 0>{}), 0u);
    EXPECT_EQ(checksum_of(std::array<std::uint64_t, 1>{0}), 0xe220a8397b1dcdafu);
    EXPECT_EQ(checksum_of(std::array<std::uint64_t, 3>{1, 2, 3}), 0xedbe5ca3654f5804u);
}

TEST(Checksum, SwappedWordsChangeIt) {
    std::array<std::uint64_t, 4> words{10, 20, 30, 40};
    const std::uint64_t          before = checksum_of(words);
    std::swap(words[1], words[2]);
    EXPECT_NE(checksum_of(words), before);
}

// A plain sum of words would give the same value for both.
TEST(Checksum, SwappedBytesChangeIt) {
    auto                bytes  = bytes_of(std::array<std::uint64_t, 2>{0x0102030405060708, 0});
    const std::uint64_t before = mk::checksum(bytes);
    std::swap(bytes[0], bytes[8]);
    EXPECT_NE(mk::checksum(bytes), before);
}

// The engine's failure case: the same bytes read 32 bytes off.
TEST(Checksum, ShiftedDataChangesIt) {
    std::array<std::uint64_t, 12> words{};
    for (std::size_t i = 0; i < words.size(); ++i) {
        words[i] = i * 0x0101010101010101u;
    }
    const auto                       bytes = bytes_of(words);
    const std::span<const std::byte> all   = bytes;
    EXPECT_NE(mk::checksum(all.subspan(32, 64)), mk::checksum(all.subspan(0, 64)));
}

TEST(Checksum, ZeroWordsCountByPosition) {
    EXPECT_NE(checksum_of(std::array<std::uint64_t, 1>{0}), 0u);
    EXPECT_NE(checksum_of(std::array<std::uint64_t, 2>{0, 0}),
              checksum_of(std::array<std::uint64_t, 1>{0}));
}

// The sum of the terms: what lets device threads add them in any order.
TEST(Checksum, IsTheSumOfItsTerms) {
    const std::array<std::uint64_t, 3> words{7, 8, 9};
    EXPECT_EQ(checksum_of(words), mk::checksum_term(9, 2) + mk::checksum_term(7, 0) +
                                      mk::checksum_term(8, 1));
}
