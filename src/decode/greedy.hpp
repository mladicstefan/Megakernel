#pragma once
#include "decode/positions.hpp"
#include "megakernel/megakernel.hpp"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <fstream>
#include <iterator>
#include <print>
#include <span>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

// Token IDs in, token IDs out: reading a prompt and decoding it greedily through the
// megakernel. Host only (C++23): nvcc never includes this.
namespace mk {
// The IDs in a file like tokens.txt, "[785, 32875, ...]": every run of digits is one ID.
// The file is untrusted, so each ID is checked against the vocabulary.
inline std::expected<std::vector<Token>, std::string> read_tokens(const std::string& path) {
    std::ifstream is(path);
    if (!is) {
        return std::unexpected("cannot open");
    }
    const std::string  text{std::istreambuf_iterator<char>(is), {}};
    std::vector<Token> out;
    const char*        p   = text.data();
    const char* const  end = p + text.size();
    while (p != end) {
        if (*p < '0' || *p > '9') {
            ++p;
            continue;
        }
        std::uint64_t id   = 0;
        const auto    read = std::from_chars(p, end, id);
        const auto    tok  = read.ec == std::errc{} ? Token::checked(id) : std::nullopt;
        if (!tok) {
            return std::unexpected(std::format("the ID at byte {} is outside the vocabulary of {}",
                                               p - text.data(), n_vocab));
        }
        out.push_back(*tok);
        p = read.ptr;
    }
    if (out.empty()) {
        return std::unexpected("no token IDs");
    }
    return out;
}

struct Generation {
    std::vector<Token> tokens;  // the generated ones, without the prompt
    std::size_t        steps;   // positions decoded
    double             seconds; // in the megakernel, token read-back included
};

// Decodes prompt, then generates `generate` tokens, each the most likely one. The caller
// has checked that the positions fit C::n_ctx.
template <MegakernelConfig C>
std::expected<Generation, std::string>
generate_greedy(const VerifiedDevice<typename C::Gpu>& device, Weights weights,
                std::span<const Token> prompt, std::size_t generate) {
    auto made_decoder = Decoder<C>::make(weights);
    auto made_kernel  = Megakernel<C>::make(device);
    const auto error  = [](const CudaError& e) {
        return std::unexpected(std::format("{}: {}", e.call, e.text));
    };
    if (const auto* e = std::get_if<CudaError>(&made_decoder)) {
        return error(*e);
    }
    if (const auto* e = std::get_if<CudaError>(&made_kernel)) {
        return error(*e);
    }
    auto& decoder = std::get<Decoder<C>>(made_decoder);
    auto& kernel  = std::get<Megakernel<C>>(made_kernel);

    Generation g{.tokens = {}, .steps = positions(prompt.size(), generate), .seconds = 0};
    Token      token = prompt.front();
    for (std::size_t pos = 0; pos < g.steps; ++pos) {
        if (pos < prompt.size()) {
            token = prompt[pos];
        }
        const auto start = std::chrono::steady_clock::now();
        const auto next  = kernel.decode(decoder, token);
        g.seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (const auto* e = std::get_if<CudaError>(&next)) {
            return error(*e);
        }
        token = std::get<Token>(next);
        if (pos + 1 >= prompt.size() && g.tokens.size() < generate) {
            g.tokens.push_back(token);
        }
    }
    return g;
}

inline void print_generated(std::span<const Token> generated) {
    std::print("generated: [");
    for (std::size_t i = 0; i < generated.size(); ++i) {
        std::print("{}{}", i ? ", " : "", generated[i].id);
    }
    std::println("]");
}

inline void print_speed(const Generation& g) {
    const auto steps = static_cast<double>(g.steps);
    std::println("decode: {} steps in {:.3f} s, {:.3f} ms per step, {:.0f} tokens/s", g.steps,
                 g.seconds, g.seconds * 1e3 / steps, steps / g.seconds);
}
} // namespace mk
