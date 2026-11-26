#pragma once
#include <cstddef>
#include <limits>

// How many positions a generation needs. Host only, and no model: a host test reaches it.
namespace mk {
// The tokens the apps generate when --generate is not given.
inline constexpr std::size_t default_generate = 32;

// The positions a prompt and `generate` more tokens take. The last generated token is
// never decoded, so it needs no position. `generate` comes from the command line, so a sum
// past SIZE_MAX saturates there instead of wrapping to a count that fits the KV cache.
inline std::size_t positions(std::size_t prompt, std::size_t generate) {
    const std::size_t decoded = generate > 0 ? generate - 1 : 0;
    const std::size_t max     = std::numeric_limits<std::size_t>::max();
    return decoded > max - prompt ? max : prompt + decoded;
}
} // namespace mk
