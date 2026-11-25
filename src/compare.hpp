#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>

// How far two tensors, or two sets of logits, are apart. Host only.
namespace mk {
struct Diff {
    double max_abs; // the largest |ours - reference|
    double rel_l2;  // |ours - reference| / |reference|, in the L2 norm
    double cosine;  // of the angle between the two; 1 when either is all zero
};

inline Diff diff(std::span<const float> ours, std::span<const float> reference) {
    assert(ours.size() == reference.size());
    double max_abs = 0, err2 = 0, ours2 = 0, ref2 = 0, dot = 0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
        const double a = ours[i];
        const double b = reference[i];
        max_abs        = std::max(max_abs, std::abs(a - b));
        err2 += (a - b) * (a - b);
        ours2 += a * a;
        ref2 += b * b;
        dot += a * b;
    }
    const bool zero = ours2 == 0 || ref2 == 0;
    return {.max_abs = max_abs,
            // Against an all-zero reference the absolute error is the only honest number.
            .rel_l2 = ref2 == 0 ? std::sqrt(err2) : std::sqrt(err2 / ref2),
            .cosine = zero ? 1.0 : dot / std::sqrt(ours2 * ref2)};
}

// The first largest element, as llama.cpp's greedy sampler picks it.
inline std::size_t argmax(std::span<const float> v) {
    return static_cast<std::size_t>(std::ranges::max_element(v) - v.begin());
}

// KL(reference || ours) in nats, between the softmax distributions of two sets of logits.
inline double kl_divergence(std::span<const float> reference, std::span<const float> ours) {
    assert(ours.size() == reference.size() && !ours.empty());
    // log(sum(exp(x))), shifted by the maximum so that exp cannot overflow.
    const auto log_sum_exp = [](std::span<const float> x) {
        const double max = *std::ranges::max_element(x);
        double       sum = 0;
        for (const float v : x) {
            sum += std::exp(double{v} - max);
        }
        return max + std::log(sum);
    };
    const double ref_norm  = log_sum_exp(reference);
    const double ours_norm = log_sum_exp(ours);
    double       kl        = 0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
        const double log_p = double{reference[i]} - ref_norm;
        kl += std::exp(log_p) * (log_p - (double{ours[i]} - ours_norm));
    }
    return kl;
}
} // namespace mk
