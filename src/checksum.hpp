#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

// The checksum that proves a tensor's bytes on the device equal its bytes in the file. The
// host computes it over the mapping and a kernel over device memory, both from
// checksum_term. nvcc compiles this header: C++20 only.
#ifdef __CUDACC__
#define MK_HOST_DEVICE __host__ __device__
#else
#define MK_HOST_DEVICE
#endif

namespace mk {
// What the 8-byte word at position index adds to a checksum. The position goes through the
// mixer (splitmix64's finalizer) with the word, so moved or swapped words change the sum.
// A checksum is the plain sum of its terms, so threads may add them in any order.
MK_HOST_DEVICE constexpr std::uint64_t checksum_term(std::uint64_t word, std::uint64_t index) {
    // index + 1: a zero word at position 0 must not add zero.
    std::uint64_t x = word + (index + 1) * 0x9e3779b97f4a7c15u;
    x               = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9u;
    x               = (x ^ (x >> 27)) * 0x94d049bb133111ebu;
    return x ^ (x >> 31);
}

// The host side. data is a whole number of 8-byte words; weights.hpp asserts that of every
// tensor.
inline std::uint64_t checksum(std::span<const std::byte> data) {
    assert(data.size() % 8 == 0);
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < data.size() / 8; ++i) {
        std::uint64_t word;
        std::memcpy(&word, data.data() + 8 * i, 8); // data may be unaligned
        sum += checksum_term(word, i);
    }
    return sum;
}
} // namespace mk
