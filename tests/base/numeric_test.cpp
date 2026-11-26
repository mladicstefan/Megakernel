#include "base/numeric.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace {
std::uint32_t bits(float f) {
    return std::bit_cast<std::uint32_t>(f);
}
float from_bits(std::uint32_t u) {
    return std::bit_cast<float>(u);
}
} // namespace

TEST(Bf16, WidensExactly) {
    EXPECT_EQ(mk::bf16_to_float(0x3f80), 1.0f);
    EXPECT_EQ(mk::bf16_to_float(0xc000), -2.0f);
    EXPECT_EQ(mk::bf16_to_float(0x0000), 0.0f);
    EXPECT_EQ(bits(mk::bf16_to_float(0x1234)), 0x12340000u);
}

TEST(Bf16, AValueItCanHoldRoundsToItself) {
    // Each has at most 8 significant bits.
    for (const float f : {0.0f, 1.0f, -2.0f, 0.15625f, 0x1.8p+35f, 0x1.fep-20f}) {
        EXPECT_EQ(mk::round_to_bf16(f), f);
    }
    EXPECT_EQ(bits(mk::round_to_bf16(-0.0f)), 0x80000000u);
}

// The low 16 bits decide: below half goes down, above half goes up, and exactly half goes
// to the even neighbour.
TEST(Bf16, RoundsToNearestTiesToEven) {
    EXPECT_EQ(bits(mk::round_to_bf16(from_bits(0x3f807fff))), 0x3f800000u);
    EXPECT_EQ(bits(mk::round_to_bf16(from_bits(0x3f808001))), 0x3f810000u);
    EXPECT_EQ(bits(mk::round_to_bf16(from_bits(0x3f808000))), 0x3f800000u); // tie, 0x3f80 is even
    EXPECT_EQ(bits(mk::round_to_bf16(from_bits(0x3f818000))), 0x3f820000u); // tie, 0x3f81 is odd
}

TEST(Bf16, RoundingCarriesIntoTheExponent) {
    EXPECT_EQ(mk::round_to_bf16(from_bits(0x3fffffff)), 2.0f); // just below 2
    EXPECT_TRUE(std::isinf(mk::round_to_bf16(std::numeric_limits<float>::max())));
}

TEST(Bf16, NanStaysNan) {
    EXPECT_TRUE(std::isnan(mk::round_to_bf16(std::numeric_limits<float>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(mk::round_to_bf16(from_bits(0x7f800001)))); // a signalling NaN
}

TEST(Bf16, BitsAreTheTopHalfOfTheRoundedFloat) {
    EXPECT_EQ(mk::float_to_bf16(1.0f), 0x3f80);
    EXPECT_EQ(mk::float_to_bf16(from_bits(0x3f808001)), 0x3f81);
    EXPECT_EQ(mk::bf16_to_float(mk::float_to_bf16(-2.0f)), -2.0f);
}
