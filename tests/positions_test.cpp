#include "positions.hpp"

#include <cstddef>
#include <limits>

#include <gtest/gtest.h>

namespace {
constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
} // namespace

TEST(Positions, TheLastGeneratedTokenTakesNone) {
    EXPECT_EQ(mk::positions(14, 32), 45u);
    EXPECT_EQ(mk::positions(14, 1), 14u);
    EXPECT_EQ(mk::positions(14, 0), 14u);
    EXPECT_EQ(mk::positions(14, 4083), 4096u);
    EXPECT_EQ(mk::positions(14, 4084), 4097u);
}

TEST(Positions, ACountPastSizeMaxSaturatesInsteadOfWrapping) {
    // 14 + (SIZE_MAX - 3) wraps to 10 in size_t: a count that fits any KV cache.
    EXPECT_EQ(mk::positions(14, kMax - 2), kMax);
    EXPECT_EQ(mk::positions(14, kMax), kMax);
    EXPECT_EQ(mk::positions(kMax, 2), kMax);
    EXPECT_EQ(mk::positions(1, kMax), kMax); // exactly SIZE_MAX: no overflow
    EXPECT_EQ(mk::positions(2, kMax), kMax);
}
