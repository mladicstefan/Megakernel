#include "decode/greedy.hpp"
#include "decode/positions.hpp"
#include "megakernel/megakernel.hpp"
#include "reference/compare.hpp"
#include "reference/forward.hpp"
#include "weights/verified_device.hpp"
#include "weights/verified_file.hpp"

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
using Model = mk::BuildModel;
using Trace = mk::Trace<Model>;
using Token = mk::Token<Model>;

// With --megakernel the megakernel decodes instead. --check then runs the reference pass next
// to it and compares every phase of the megakernel on the reference's operands.
constexpr std::string_view kUsage =
    "usage: forward [--model <gguf>] [--generate <n>] [--compare <dump-dir> [--per-op]] "
    "<tokens.txt>\n"
    "       forward [--model <gguf>] [--generate <n>] --megakernel [--check] <tokens.txt>";

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
// Measured worst, with the engine's BF16 cache: rel L2 3.0e-2, 1-cosine 3.7e-4, KL 3.0e-4.
constexpr Tolerance kExactTolerance{.rel_l2 = 5e-2, .one_minus_cos = 1e-3, .kl = 1e-3};
// The megakernel against the reference pass, phase by phase: only the order of sums
// differs. Measured worst over 600 positions: rel L2 5.7e-6, 1-cosine 2.4e-12, KL 1.6e-10.
constexpr Tolerance kMegakernelTolerance{.rel_l2 = 5e-5, .one_minus_cos = 1e-10, .kl = 2e-9};
// kqv_out has its own limits. K and V are rounded to BF16 on their way into the cache, so a
// sum in another order can move a cache entry by one BF16 step. Measured worst: rel L2
// 1.3e-3, 1-cosine 8.0e-7. With an F32 cache it is 3.7e-6, like the other tensors.
constexpr Tolerance kMegakernelAttentionTolerance{.rel_l2 = 5e-3, .one_minus_cos = 5e-6, .kl = 0};

struct Options {
    std::string                          model = MODEL_GGUF;
    std::string                          tokens;
    std::optional<std::filesystem::path> compare;
    std::size_t                          generate   = mk::default_generate;
    bool                                 per_op     = false;
    bool                                 megakernel = false;
    bool                                 check      = false;
};

std::optional<Options> parse_args(std::span<char*> args) {
    Options o;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (a == "--per-op") {
            o.per_op = true;
        } else if (a == "--megakernel") {
            o.megakernel = true;
        } else if (a == "--check") {
            o.check = true;
        } else if (a == "--model" || a == "--compare" || a == "--generate") {
            if (i + 1 == args.size()) {
                return std::nullopt;
            }
            const std::string_view v = args[++i];
            if (a == "--model") {
                o.model = v;
            } else if (a == "--compare") {
                o.compare = v;
            } else if (const auto [end, ec] = std::from_chars(v.data(), v.data() + v.size(),
                                                              o.generate);
                       ec != std::errc{} || end != v.data() + v.size()) {
                return std::nullopt; // ec is set for no digits and for a value past SIZE_MAX
            }
        } else if (!a.starts_with("--") && o.tokens.empty()) {
            o.tokens = a;
        } else {
            return std::nullopt;
        }
    }
    if (o.tokens.empty() || (o.per_op && !o.compare) || (o.check && !o.megakernel) ||
        (o.megakernel && o.compare)) {
        return std::nullopt;
    }
    return o;
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
                                                bool layers, Trace& out) {
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

    void add_layers(std::size_t pos, const Trace& ours, const Trace& reference) {
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

std::string text(const mk::CudaError& e) {
    return std::format("{}: {}", e.call, e.text);
}

// The model's weights on the device, after the file passed the engine's checks. The
// mapping is closed again: the device copy is all a decoder reads.
std::expected<mk::DeviceWeights, std::string> load_weights(const std::string& model) {
    const auto file = mk::VerifiedFile::open(model.c_str());
    if (!file) {
        return std::unexpected(std::format("{}: {}", model, file.error()));
    }
    auto uploaded = mk::upload(file->weights());
    if (const auto* e = std::get_if<mk::CudaError>(&uploaded)) {
        return std::unexpected(text(*e));
    }
    return std::move(std::get<mk::Upload>(uploaded).weights);
}

// The megakernel decodes. With --check, the reference pass decodes the same tokens first
// and the megakernel runs each phase on the reference's operands.
int run_megakernel(const Options& o, const std::vector<Token>* prompt) {
    using C = mk::BuildConfig;
    const std::size_t steps = mk::positions(prompt->size(), o.generate);
    if (steps > C::n_ctx) {
        return fail(std::format("{} positions are more than the {} this build's KV cache holds",
                                steps, C::n_ctx));
    }
    const auto device = mk::VerifiedDevice<C::Gpu>::check();
    if (!device) {
        return fail(device.error());
    }
    const auto weights = load_weights(o.model);
    if (!weights) {
        return fail(weights.error());
    }
    std::println("forward: {} prompt tokens, {} to generate, megakernel{}", prompt->size(),
                 o.generate, o.check ? ", each phase on the reference pass's operands" : "");
    if (!o.check) {
        const auto g = mk::generate_greedy<C>(*device, weights->view(), *prompt, o.generate);
        if (!g) {
            return fail(g.error());
        }
        mk::print_generated(g->tokens);
        mk::print_speed(*g);
        return 0;
    }

    auto made_decoder   = mk::Decoder<C>::make(weights->view());
    auto made_reference = mk::Decoder<C>::make(weights->view());
    auto made_kernel    = mk::Megakernel<C>::make(*device);
    for (const auto* e : {std::get_if<mk::CudaError>(&made_decoder),
                          std::get_if<mk::CudaError>(&made_reference)}) {
        if (e != nullptr) {
            return fail(*e);
        }
    }
    if (const auto* e = std::get_if<mk::CudaError>(&made_kernel)) {
        return fail(*e);
    }
    auto& decoder   = std::get<mk::Decoder<C>>(made_decoder);
    auto& reference = std::get<mk::Decoder<C>>(made_reference);
    auto& kernel    = std::get<mk::Megakernel<C>>(made_kernel);

    // The tensors the megakernel writes. It fuses the others away.
    constexpr std::string_view kMaterialized[] = {"embd",    "Qcur",    "Kcur",       "Vcur",
                                                  "kqv_out", "ffn_inp", "ffn_swiglu", "l_out"};
    const auto                 ours = std::make_unique<Trace>();
    const auto                 ref  = std::make_unique<Trace>();
    std::vector<Token>         generated;
    Table                      table;
    std::size_t                top1_agree = 0;
    double                     max_kl = 0, seconds = 0;

    Token token = prompt->front();
    for (std::size_t pos = 0; pos < steps; ++pos) {
        if (pos < prompt->size()) {
            token = (*prompt)[pos];
        }
        if (const auto e = reference.decode(token)) {
            return fail(*e);
        }
        if (const auto e = reference.read_trace(*ref)) {
            return fail(*e);
        }
        const auto start = std::chrono::steady_clock::now();
        const auto next  = kernel.decode_from(decoder, token, *ref);
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (const auto* e = std::get_if<mk::CudaError>(&next)) {
            return fail(*e);
        }
        token = std::get<Token>(next);

        if (const auto e = decoder.read_trace(*ours)) {
            return fail(*e);
        }
        std::vector<std::span<const float>> theirs;
        mk::for_each_tensor(std::as_const(*ref),
                            [&](int, std::string_view, std::span<const float> values) {
                                theirs.push_back(values);
                            });
        std::size_t at = 0;
        mk::for_each_tensor(std::as_const(*ours), [&](int layer, std::string_view name,
                                                      std::span<const float> values) {
            if (std::ranges::contains(kMaterialized, name)) {
                table.add(pos, layer, name, values, theirs[at]);
            }
            ++at;
        });
        const std::span<const float> ref_logits = ref->result_output;
        table.add(pos, -1, "result_output", ours->result_output, ref_logits);
        max_kl = std::max(max_kl, mk::kl_divergence(ref_logits, ours->result_output));
        // The megakernel's own argmax, against the reference's logits.
        const Token expected = *Token::checked(mk::argmax(ref_logits));
        top1_agree += token.id == expected.id;
        token = expected; // both keep decoding the reference's chain
        if (pos + 1 >= prompt->size() && generated.size() < o.generate) {
            generated.push_back(token);
        }
    }

    mk::print_generated(generated);
    std::println("decode: {} steps in {:.3f} s, {:.3f} ms per step, {:.0f} tokens/s", steps,
                 seconds, seconds * 1e3 / static_cast<double>(steps),
                 static_cast<double>(steps) / seconds);
    bool ok = top1_agree == steps && max_kl <= kMegakernelTolerance.kl;
    std::println("check: against the reference pass");
    std::println("  {:<14} {:>7} {:>10} {:>10} {:>10}  {}", "tensor", "tensors", "max abs", "rel L2",
                 "1-cosine", "worst rel L2 at");
    for (const Row& r : table.rows) {
        std::println("  {:<14} {:>7} {:>10.2e} {:>10.2e} {:>10.2e}  {}", r.name, r.tensors,
                     r.max_abs, r.rel_l2, 1 - r.cosine, r.worst);
        const Tolerance& tol = r.name == "kqv_out" ? kMegakernelAttentionTolerance
                                                   : kMegakernelTolerance;
        ok = ok && r.rel_l2 <= tol.rel_l2 && 1 - r.cosine <= tol.one_minus_cos;
    }
    std::println("logits: the megakernel's token is the reference's at {} of {} positions, "
                 "largest KL {:.2e} nats",
                 top1_agree, steps, max_kl);
    std::println("result: {} (limits: rel L2 {:.0e}, 1-cosine {:.0e}, KL {:.0e}, token equal; "
                 "kqv_out rel L2 {:.0e}, 1-cosine {:.0e})",
                 ok ? "within tolerance" : "OUTSIDE TOLERANCE", kMegakernelTolerance.rel_l2,
                 kMegakernelTolerance.one_minus_cos, kMegakernelTolerance.kl,
                 kMegakernelAttentionTolerance.rel_l2,
                 kMegakernelAttentionTolerance.one_minus_cos);
    return ok ? 0 : 1;
}

// Decodes with the decoder that is compiled for configuration C.
template <mk::DecoderConfig C>
int run(const Options& o, const std::vector<Token>* prompt) {
    // A dump has the layers of the prompt's positions only, so --per-op stops there.
    const std::size_t generate = o.per_op ? 1 : o.generate;
    const std::size_t steps    = mk::positions(prompt->size(), generate);
    if (steps > C::n_ctx) {
        return fail(std::format("{} positions are more than the {} this build's KV cache holds",
                                steps, C::n_ctx));
    }

    const auto weights = load_weights(o.model);
    if (!weights) {
        return fail(weights.error());
    }
    auto made = mk::Decoder<C>::make(weights->view());
    if (const auto* e = std::get_if<mk::CudaError>(&made)) {
        return fail(*e);
    }
    mk::Decoder<C>& decoder = std::get<mk::Decoder<C>>(made);

    std::println("forward: {} prompt tokens, {} to generate, {}", prompt->size(), generate,
                 o.per_op ? "each op on the reference's operands, with llama.cpp's CPU rounding"
                          : "exact arithmetic");

    // 4 MB each: not on the stack.
    const auto             ours      = std::make_unique<Trace>();
    const auto             reference = std::make_unique<Trace>();
    std::vector<Token>     generated;
    Table                  table;
    // Logits against the reference's, while both chains have decoded the same tokens.
    std::size_t                compared = 0, top1_agree = 0;
    double                     max_kl = 0, min_margin = 1e30;
    std::optional<std::size_t> diverged;
    double                     decode_seconds = 0;

    Token token = prompt->front();
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
        if (const auto e = o.per_op ? decoder.decode_from(token, *reference)
                                    : decoder.decode(token)) {
            return fail(*e);
        }
        decode_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (const auto e = layers ? decoder.read_trace(*ours)
                                  : decoder.read_logits(ours->result_output)) {
            return fail(*e);
        }

        const std::span<const float> logits = ours->result_output;
        token = *Token::checked(mk::argmax(logits)); // an index into the logits: in range
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

    mk::print_generated(generated);
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
} // namespace

int main(int argc, char** argv) {
    const auto options = parse_args({argv, static_cast<std::size_t>(argc)});
    if (!options) {
        std::println(stderr, "{}", kUsage);
        return 2;
    }
    const auto prompt = mk::read_tokens<Model>(options->tokens);
    if (!prompt) {
        return fail(std::format("{}: {}", options->tokens, prompt.error()));
    }
    if (options->megakernel) {
        return run_megakernel(*options, &*prompt);
    }
    return options->per_op ? run<mk::LlamaCpuConfig<Model>>(*options, &*prompt)
                           : run<mk::BuildConfig>(*options, &*prompt);
}
