#include "compare.hpp"
#include "forward.hpp"
#include "verified_file.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// forward: decodes token IDs greedily with the reference forward pass. With --compare it
// checks every intermediate against a dump that tools/reference made with llama.cpp, in one
// of two ways:
// - as it runs, in exact arithmetic. The differences are llama.cpp's own rounding, grown
//   over 28 layers, so the limits are loose.
// - with --per-op, each op on llama.cpp's operands and with its rounding. An op then differs
//   from llama.cpp only in the order of its sums, so the limits are tight.
namespace {
constexpr std::string_view kUsage =
    "usage: forward [--model <gguf>] [--generate <n>] [--compare <dump-dir> [--per-op]] "
    "<tokens.txt>";

// The limits a comparison must stay inside: a few times the worst value measured on the
// planets prompt (docs/PLAN.md, Phase C).
struct Tolerance {
    double rel_l2;        // per tensor
    double one_minus_cos; // per tensor
    double kl;            // per position, nats
};
// Measured worst: rel L2 1.05e-4 (kqv_out; every other tensor is below 3e-6), 1-cosine
// 5.4e-9, KL 4.2e-11.
constexpr Tolerance kPerOpTolerance{.rel_l2 = 5e-4, .one_minus_cos = 1e-7, .kl = 1e-9};
// Measured worst: rel L2 1.5e-2, 1-cosine 1.1e-4, KL 1.5e-4.
constexpr Tolerance kExactTolerance{.rel_l2 = 5e-2, .one_minus_cos = 1e-3, .kl = 1e-3};

struct Options {
    std::string                          model = MODEL_GGUF;
    std::string                          tokens;
    std::optional<std::filesystem::path> compare;
    std::size_t                          generate = 32;
    bool                                 per_op   = false;
};

std::optional<Options> parse_args(std::span<char*> args) {
    Options o;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (a == "--per-op") {
            o.per_op = true;
        } else if (a == "--model" || a == "--compare" || a == "--generate") {
            if (i + 1 == args.size()) {
                return std::nullopt;
            }
            const std::string_view v = args[++i];
            if (a == "--model") {
                o.model = v;
            } else if (a == "--compare") {
                o.compare = v;
            } else if (std::from_chars(v.data(), v.data() + v.size(), o.generate).ptr !=
                       v.data() + v.size()) {
                return std::nullopt;
            }
        } else if (!a.starts_with("--") && o.tokens.empty()) {
            o.tokens = a;
        } else {
            return std::nullopt;
        }
    }
    if (o.tokens.empty() || (o.per_op && !o.compare)) {
        return std::nullopt;
    }
    return o;
}

// The IDs in a file like tokens.txt, "[785, 32875, ...]": every run of digits is one ID.
// The file is untrusted, so each ID is checked against the vocabulary.
std::expected<std::vector<mk::Token>, std::string> read_tokens(const std::string& path) {
    std::ifstream is(path);
    if (!is) {
        return std::unexpected("cannot open");
    }
    const std::string      text{std::istreambuf_iterator<char>(is), {}};
    std::vector<mk::Token> out;
    const char*            p   = text.data();
    const char* const      end = p + text.size();
    while (p != end) {
        if (*p < '0' || *p > '9') {
            ++p;
            continue;
        }
        std::uint64_t id   = 0;
        const auto    read = std::from_chars(p, end, id);
        const auto    tok  = read.ec == std::errc{} ? mk::Token::checked(id) : std::nullopt;
        if (!tok) {
            return std::unexpected(std::format("the ID at byte {} is outside the vocabulary of {}",
                                               p - text.data(), mk::n_vocab));
        }
        out.push_back(*tok);
        p = read.ptr;
    }
    if (out.empty()) {
        return std::unexpected("no token IDs");
    }
    return out;
}

// One F32 tensor of a dump, which must be exactly the size of out.
std::expected<void, std::string> read_tensor(const std::filesystem::path& file,
                                             std::span<float> out) {
    std::error_code ec;
    const auto      size = std::filesystem::file_size(file, ec);
    if (ec) {
        return std::unexpected(std::format("{}: cannot read", file.string()));
    }
    if (size != out.size_bytes()) {
        return std::unexpected(std::format("{}: {} bytes, this tensor has {}", file.string(), size,
                                           out.size_bytes()));
    }
    std::ifstream is(file, std::ios::binary);
    is.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
    if (!is) {
        return std::unexpected(std::format("{}: cannot read", file.string()));
    }
    return {};
}

// The dump's trace of one position: the logits, and with layers every other tensor too.
std::expected<void, std::string> read_reference(const std::filesystem::path& dir, std::size_t pos,
                                                bool layers, mk::Trace& out) {
    const auto                       pos_dir = dir / std::format("pos{:03}", pos);
    std::expected<void, std::string> result;
    if (layers) {
        mk::for_each_tensor(out, [&](int layer, std::string_view name, std::span<float> values) {
            if (result) {
                result = read_tensor(pos_dir / (layer < 0 ? std::format("{}.bin", name)
                                                          : std::format("blk{:02}.{}.bin", layer,
                                                                        name)),
                                     values);
            }
        });
    }
    if (result) {
        result = read_tensor(pos_dir / "result_output.bin", out.result_output);
    }
    return result;
}

// The worst differences seen for one tensor name, over every layer and position.
struct Row {
    std::string_view name;
    std::size_t      tensors = 0;
    double           max_abs = 0;
    double           rel_l2  = 0;
    double           cosine  = 1;
    std::string      worst; // where rel_l2 was largest
};

struct Table {
    std::vector<Row> rows;

    void add(std::size_t pos, int layer, std::string_view name, std::span<const float> ours,
             std::span<const float> reference) {
        auto row = std::ranges::find(rows, name, &Row::name);
        if (row == rows.end()) {
            row       = rows.emplace(rows.end());
            row->name = name;
        }
        const mk::Diff d = mk::diff(ours, reference);
        ++row->tensors;
        row->max_abs = std::max(row->max_abs, d.max_abs);
        row->cosine  = std::min(row->cosine, d.cosine);
        if (d.rel_l2 >= row->rel_l2) {
            row->rel_l2 = d.rel_l2;
            row->worst  = layer < 0 ? std::format("pos{:03}", pos)
                                    : std::format("pos{:03} blk{:02}", pos, layer);
        }
    }

    void add_layers(std::size_t pos, const mk::Trace& ours, const mk::Trace& reference) {
        std::vector<std::span<const float>> theirs;
        mk::for_each_tensor(reference, [&](int, std::string_view, std::span<const float> values) {
            theirs.push_back(values);
        });
        std::size_t next = 0;
        mk::for_each_tensor(ours,
                            [&](int layer, std::string_view name, std::span<const float> values) {
                                add(pos, layer, name, values, theirs[next++]);
                            });
    }
};

int fail(std::string_view what) {
    std::println(stderr, "forward: {}", what);
    return 1;
}
int fail(const mk::CudaError& e) {
    return fail(std::format("{}: {}", e.call, e.text));
}
} // namespace

int main(int argc, char** argv) {
    const auto options = parse_args({argv, static_cast<std::size_t>(argc)});
    if (!options) {
        std::println(stderr, "{}", kUsage);
        return 2;
    }
    const Options& o = *options;

    const auto prompt = read_tokens(o.tokens);
    if (!prompt) {
        return fail(std::format("{}: {}", o.tokens, prompt.error()));
    }
    // A dump has the layers of the prompt's positions only, so --per-op stops there. The
    // last generated token is never decoded, so it needs no position.
    const std::size_t generate = o.per_op ? 1 : o.generate;
    const std::size_t steps    = prompt->size() + (generate > 0 ? generate - 1 : 0);
    if (steps > mk::arch::context_length) {
        return fail(std::format("{} positions are more than the model's context of {}", steps,
                                mk::arch::context_length));
    }

    const auto file = mk::VerifiedFile::open(o.model.c_str());
    if (!file) {
        return fail(std::format("{}: {}", o.model, file.error()));
    }
    const auto uploaded = mk::upload(file->weights());
    if (const auto* e = std::get_if<mk::CudaError>(&uploaded)) {
        return fail(*e);
    }
    auto made = mk::make_decoder(std::get<mk::Upload>(uploaded).weights.view(),
                                 static_cast<std::uint32_t>(steps),
                                 o.per_op ? mk::Arithmetic::llama_cpu : mk::Arithmetic::exact);
    if (const auto* e = std::get_if<mk::CudaError>(&made)) {
        return fail(*e);
    }
    mk::Decoder& decoder = std::get<mk::Decoder>(made);

    std::println("forward: {} prompt tokens, {} to generate, {}", prompt->size(), generate,
                 o.per_op ? "each op on the reference's operands, with llama.cpp's CPU rounding"
                          : "exact arithmetic");

    // 4 MB each: not on the stack.
    const auto             ours      = std::make_unique<mk::Trace>();
    const auto             reference = std::make_unique<mk::Trace>();
    std::vector<mk::Token> generated;
    Table                  table;
    // Logits against the reference's, while both chains have decoded the same tokens.
    std::size_t                compared = 0, top1_agree = 0;
    double                     max_kl = 0, min_margin = 1e30;
    std::optional<std::size_t> diverged;
    double                     decode_seconds = 0;

    mk::Token token = prompt->front();
    for (std::size_t pos = 0; pos < steps; ++pos) {
        if (pos < prompt->size()) {
            token = (*prompt)[pos];
        }
        const bool comparing = o.compare && !diverged;
        const bool layers    = comparing && pos < prompt->size();
        if (comparing) {
            if (const auto read = read_reference(*o.compare, pos, layers, *reference); !read) {
                return fail(read.error());
            }
        }

        const auto start = std::chrono::steady_clock::now();
        if (const auto e = o.per_op ? mk::decode_from(decoder, token, *reference)
                                    : mk::decode(decoder, token)) {
            return fail(*e);
        }
        decode_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (const auto e = layers ? mk::read_trace(decoder, *ours)
                                  : mk::read_logits(decoder, ours->result_output)) {
            return fail(*e);
        }

        const std::span<const float> logits = ours->result_output;
        token = *mk::Token::checked(mk::argmax(logits)); // an index into the logits: in range
        if (pos + 1 >= prompt->size() && generated.size() < generate) {
            generated.push_back(token);
        }
        if (!comparing) {
            continue;
        }

        if (layers) {
            table.add_layers(pos, *ours, *reference);
        }
        const std::span<const float> ref_logits = reference->result_output;
        table.add(pos, -1, "result_output", logits, ref_logits);
        ++compared;
        max_kl = std::max(max_kl, mk::kl_divergence(ref_logits, logits));
        // How far the reference's best token is ahead of its second best.
        std::vector<float> top(ref_logits.begin(), ref_logits.end());
        std::ranges::nth_element(top, top.begin() + 1, std::greater{});
        min_margin = std::min(min_margin, static_cast<double>(top[0] - top[1]));
        if (mk::argmax(ref_logits) == token.id) {
            ++top1_agree;
        } else {
            diverged = pos; // from here on the two chains decode different tokens
        }
    }

    std::print("generated: [");
    for (std::size_t i = 0; i < generated.size(); ++i) {
        std::print("{}{}", i ? ", " : "", generated[i].id);
    }
    std::println("]");
    std::println("decode: {} steps in {:.3f} s, {:.1f} ms per step", steps, decode_seconds,
                 decode_seconds * 1e3 / static_cast<double>(steps));
    if (!o.compare) {
        return 0;
    }

    const Tolerance tol = o.per_op ? kPerOpTolerance : kExactTolerance;
    bool            ok  = !diverged && max_kl <= tol.kl;
    std::println("compare: {}", o.compare->string());
    std::println("  {:<14} {:>7} {:>10} {:>10} {:>10}  {}", "tensor", "tensors", "max abs", "rel L2",
                 "1-cosine", "worst rel L2 at");
    for (const Row& r : table.rows) {
        std::println("  {:<14} {:>7} {:>10.2e} {:>10.2e} {:>10.2e}  {}", r.name, r.tensors,
                     r.max_abs, r.rel_l2, 1 - r.cosine, r.worst);
        ok = ok && r.rel_l2 <= tol.rel_l2 && 1 - r.cosine <= tol.one_minus_cos;
    }
    std::println("logits: top-1 agrees at {} of {} positions, largest KL {:.2e} nats, smallest "
                 "reference margin {:.3f}",
                 top1_agree, compared, max_kl, min_margin);
    if (diverged) {
        std::println("greedy: the chains part at position {}", *diverged);
    }
    std::println("result: {} (limits: rel L2 {:.0e}, 1-cosine {:.0e}, KL {:.0e}, top-1 equal)",
                 ok ? "within tolerance" : "OUTSIDE TOLERANCE", tol.rel_l2, tol.one_minus_cos, tol.kl);
    return ok ? 0 : 1;
}
