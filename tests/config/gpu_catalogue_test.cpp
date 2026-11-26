#include "config/gpu_catalogue.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace mk {
namespace {
TEST(SameCard, TheDriversBrandPrefixIsNotPartOfTheName) {
    EXPECT_TRUE(same_card("NVIDIA GeForce RTX 4070 Ti SUPER", Rtx4070TiSuper::name));
    EXPECT_TRUE(same_card("RTX 4070 Ti SUPER", Rtx4070TiSuper::name));
    EXPECT_TRUE(same_card("NVIDIA GeForce RTX 5070", Rtx5070::name));
}

TEST(SameCard, ANeighbouringCardIsAnotherCard) {
    EXPECT_FALSE(same_card("NVIDIA GeForce RTX 4070 Ti", Rtx4070TiSuper::name));
    EXPECT_FALSE(same_card("NVIDIA GeForce RTX 4070", Rtx4070TiSuper::name));
    EXPECT_FALSE(same_card("NVIDIA GeForce RTX 4070 Ti SUPER Laptop GPU", Rtx4070TiSuper::name));
    EXPECT_FALSE(same_card("NVIDIA GeForce RTX 5070 Ti", Rtx5070::name));
    EXPECT_FALSE(same_card("NVIDIA GeForce RTX 3070", Rtx5070::name));
    // The catalogue's name has to start a word of the driver's.
    EXPECT_FALSE(same_card("NVIDIA GeForce XRTX 5070", Rtx5070::name));
    EXPECT_FALSE(same_card("NVIDIA GeForce RTX 5070", ""));
    EXPECT_FALSE(same_card("", Rtx5070::name));
}

// Whatever the driver calls a catalogued card, it is that card and no other one.
#define MK_NAME(T) T::name,
TEST(SameCard, EveryCataloguedCardMatchesOnlyItself) {
    for (const std::string_view named : {MK_GPU_CATALOGUE(MK_NAME)}) {
        const std::string driver = "NVIDIA GeForce " + std::string(named);
        int               hits   = 0;
        for (const std::string_view other : {MK_GPU_CATALOGUE(MK_NAME)}) {
            hits += same_card(driver, other) ? 1 : 0;
        }
        EXPECT_EQ(hits, 1) << driver;
    }
}
#undef MK_NAME
} // namespace
} // namespace mk
