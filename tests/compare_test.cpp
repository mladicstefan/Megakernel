#include "compare.hpp"

#include <array>
#include <cmath>

#include <gtest/gtest.h>

TEST(Diff, EqualTensors) {
    const std::array<float, 3> a{1, -2, 3};
    const mk::Diff             d = mk::diff(a, a);
    EXPECT_EQ(d.max_abs, 0);
    EXPECT_EQ(d.rel_l2, 0);
    EXPECT_DOUBLE_EQ(d.cosine, 1);
}

TEST(Diff, KnownValues) {
    const std::array<float, 2> ours{3, 0};
    const std::array<float, 2> reference{0, 4};
    const mk::Diff             d = mk::diff(ours, reference);
    EXPECT_DOUBLE_EQ(d.max_abs, 4);
    EXPECT_DOUBLE_EQ(d.rel_l2, 5.0 / 4.0); // |(3, -4)| / |(0, 4)|
    EXPECT_DOUBLE_EQ(d.cosine, 0);         // at a right angle
}

TEST(Diff, CosineIgnoresScaleAndSeesSign) {
    const std::array<float, 2> a{1, 2};
    const std::array<float, 2> twice{2, 4};
    const std::array<float, 2> negated{-1, -2};
    EXPECT_DOUBLE_EQ(mk::diff(twice, a).cosine, 1);
    EXPECT_DOUBLE_EQ(mk::diff(twice, a).rel_l2, 1);
    EXPECT_DOUBLE_EQ(mk::diff(negated, a).cosine, -1);
}

TEST(Diff, AnAllZeroReferenceReportsTheAbsoluteError) {
    const std::array<float, 2> ours{3, 4};
    const std::array<float, 2> zero{0, 0};
    const mk::Diff             d = mk::diff(ours, zero);
    EXPECT_DOUBLE_EQ(d.rel_l2, 5);
    EXPECT_DOUBLE_EQ(d.cosine, 1);
}

TEST(Argmax, TheFirstOfEqualMaxima) {
    EXPECT_EQ(mk::argmax(std::array<float, 4>{1, 7, 7, 2}), 1u);
    EXPECT_EQ(mk::argmax(std::array<float, 1>{-5}), 0u);
}

TEST(Kl, ZeroForEqualDistributions) {
    const std::array<float, 3> a{1, 2, 3};
    const std::array<float, 3> shifted{101, 102, 103}; // softmax ignores a common shift
    EXPECT_NEAR(mk::kl_divergence(a, a), 0, 1e-15);
    EXPECT_NEAR(mk::kl_divergence(a, shifted), 0, 1e-12);
}

// P = (1/2, 1/2) against Q = (1/4, 3/4): KL = ln(2) - ln(3)/2.
TEST(Kl, KnownValue) {
    const std::array<float, 2> p{0, 0};
    const std::array<float, 2> q{0, std::log(3.0f)};
    EXPECT_NEAR(mk::kl_divergence(p, q), std::log(2.0) - std::log(3.0) / 2, 1e-7);
    EXPECT_GT(mk::kl_divergence(q, p), 0);
}

// Logits this large overflow a plain exp.
TEST(Kl, LargeLogitsStayFinite) {
    const std::array<float, 2> a{1000, 999};
    const std::array<float, 2> b{1000, 998};
    EXPECT_TRUE(std::isfinite(mk::kl_divergence(a, b)));
    EXPECT_GT(mk::kl_divergence(a, b), 0);
}
