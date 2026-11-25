// ref-dump: runs token IDs through llama.cpp one llama_decode call per token and dumps the
// per-layer intermediates (eval callback) and logits as F32 .bin files plus manifest.json.
// usage: ref-dump <model.gguf> <tokens.txt> <out-dir> [--threads N] [--generate N]
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static_assert(std::endian::native == std::endian::little, "the .bin files are little-endian");

namespace {

// What to dump: (llama.cpp node name without "-<layer>", op, name in the dump). Names repeat
// in the graph (three nodes are called "Qcur-L"), so the op picks the node.
struct Want {
    const char * base;
    ggml_op op;
    const char * dump;
};
constexpr Want layer_wants[] = {
    {"attn_norm", GGML_OP_MUL, "attn_norm"},
    {"Qcur", GGML_OP_RESHAPE, "Qcur"},  // before q_norm
    {"Kcur", GGML_OP_RESHAPE, "Kcur"},
    {"Vcur", GGML_OP_RESHAPE, "Vcur"},
    {"Qcur_normed", GGML_OP_MUL, "Qcur_normed"},
    {"Kcur_normed", GGML_OP_MUL, "Kcur_normed"},
    {"Qcur", GGML_OP_ROPE, "Qcur_rope"},
    {"Kcur", GGML_OP_ROPE, "Kcur_rope"},
    {"kqv_out", GGML_OP_RESHAPE, "kqv_out"},  // attention, before wo
    {"", GGML_OP_MUL_MAT, "attn_out"},  // after wo; unnamed in llama.cpp, matched by its weight
    {"ffn_inp", GGML_OP_ADD, "ffn_inp"},
    {"ffn_norm", GGML_OP_MUL, "ffn_norm"},
    {"ffn_gate", GGML_OP_MUL_MAT, "ffn_gate"},
    {"ffn_up", GGML_OP_MUL_MAT, "ffn_up"},
    {"ffn_swiglu", GGML_OP_GLU, "ffn_swiglu"},
    {"ffn_out", GGML_OP_MUL_MAT, "ffn_out"},
    {"l_out", GGML_OP_ADD, "l_out"},
};
constexpr const Want & attn_out_want = layer_wants[9];
constexpr Want global_wants[] = {
    {"embd", GGML_OP_GET_ROWS, "embd"},
    {"result_norm", GGML_OP_MUL, "result_norm"},
    {"result_output", GGML_OP_MUL_MAT, "result_output"},
};

struct Entry {
    std::string name, file;
    int layer, pos;
    int64_t ne[4];
    ggml_type type;
};

struct State {
    std::filesystem::path out;
    int n_layer = 0;
    int pos = 0;
    bool dump_layers = true;  // false: only result_output, for generated positions
    int captured = 0;
    std::vector<Entry> entries;
    std::vector<float> logits;  // result_output of the current step, checked against llama_get_logits
    long nonfinite = 0;
};

// Returns the dump entry for a node, or nullptr. Sets *layer (-1 for the global ones).
const Want * match(const ggml_tensor * t, const State & s, int * layer) {
    const std::string name = t->name;
    if (t->op == GGML_OP_MUL_MAT && t->src[0] != nullptr) {
        int il = -1;
        char expect[64];
        if (std::sscanf(t->src[0]->name, "blk.%d.", &il) == 1 && il >= 0 && il < s.n_layer &&
            std::snprintf(expect, sizeof expect, "blk.%d.attn_output.weight", il) > 0 &&
            std::strcmp(t->src[0]->name, expect) == 0) {
            *layer = il;
            return s.dump_layers ? &attn_out_want : nullptr;
        }
    }
    for (const Want & w : global_wants) {
        if (t->op == w.op && name == w.base) {
            *layer = -1;
            return s.dump_layers || std::strcmp(w.dump, "result_output") == 0 ? &w : nullptr;
        }
    }
    if (!s.dump_layers) return nullptr;
    const auto dash = name.rfind('-');
    if (dash == std::string::npos || dash + 1 == name.size()) return nullptr;
    char * end = nullptr;
    const long il = std::strtol(name.c_str() + dash + 1, &end, 10);
    if (*end != '\0' || il < 0 || il >= s.n_layer) return nullptr;
    const std::string base = name.substr(0, dash);
    for (const Want & w : layer_wants) {
        if (t->op == w.op && base == w.base) {
            *layer = static_cast<int>(il);
            return &w;
        }
    }
    return nullptr;
}

bool on_eval(ggml_tensor * t, bool ask, void * user) {
    auto & s = *static_cast<State *>(user);
    int layer = 0;
    const Want * w = match(t, s, &layer);
    if (ask) return w != nullptr;
    if (w == nullptr) return true;
    if (!ggml_is_contiguous(t)) {
        std::fprintf(stderr, "error: %s is not contiguous\n", t->name);
        std::exit(1);
    }

    const auto n = static_cast<size_t>(ggml_nelements(t));
    std::vector<char> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> f(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(f.data(), raw.data(), n * sizeof(float));
    } else if (const auto * tr = ggml_get_type_traits(t->type); tr->to_float != nullptr) {
        tr->to_float(raw.data(), f.data(), static_cast<int64_t>(n));
    } else {
        std::fprintf(stderr, "error: %s has type %s with no F32 conversion\n", t->name, ggml_type_name(t->type));
        std::exit(1);
    }
    for (float v : f) s.nonfinite += std::isfinite(v) ? 0 : 1;

    char rel[128];
    if (layer >= 0) {
        std::snprintf(rel, sizeof rel, "pos%03d/blk%02d.%s.bin", s.pos, layer, w->dump);
    } else {
        std::snprintf(rel, sizeof rel, "pos%03d/%s.bin", s.pos, w->dump);
    }
    const auto path = s.out / rel;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream os(path, std::ios::binary);
    os.write(reinterpret_cast<const char *>(f.data()), static_cast<std::streamsize>(n * sizeof(float)));
    if (!os) {
        std::fprintf(stderr, "error: cannot write %s\n", path.c_str());
        std::exit(1);
    }

    s.entries.push_back({w->dump, rel, layer, s.pos, {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}, t->type});
    ++s.captured;
    if (std::strcmp(w->dump, "result_output") == 0) s.logits = std::move(f);
    return true;
}

std::vector<llama_token> read_tokens(const char * path) {
    std::ifstream is(path);
    const std::string text{std::istreambuf_iterator<char>(is), {}};
    std::vector<llama_token> out;
    // Accepts llama-tokenize's --ids output, "[785, 32875, ...]": every run of digits is one ID.
    for (size_t i = 0; i < text.size();) {
        if (text[i] < '0' || text[i] > '9') {
            ++i;
            continue;
        }
        long v = 0;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9' && v < 100000000) v = v * 10 + (text[i++] - '0');
        out.push_back(static_cast<llama_token>(v));
    }
    return out;
}

llama_token argmax(const std::vector<float> & v) {
    size_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = i;  // first maximum wins, as in llama_sampler_init_greedy
    }
    return static_cast<llama_token>(best);
}

void print_ids(std::FILE * f, const std::vector<llama_token> & ids) {
    for (size_t i = 0; i < ids.size(); ++i) std::fprintf(f, "%s%d", i ? ", " : "", ids[i]);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tokens.txt> <out-dir> [--threads N] [--generate N]\n", argv[0]);
        return 2;
    }
    int threads = 8;
    int generate = 32;
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 < argc && std::strcmp(argv[i], "--threads") == 0) {
            threads = std::atoi(argv[i + 1]);
        } else if (i + 1 < argc && std::strcmp(argv[i], "--generate") == 0) {
            generate = std::atoi(argv[i + 1]);
        } else {
            std::fprintf(stderr, "error: bad option %s\n", argv[i]);
            return 2;
        }
    }
    const auto prompt = read_tokens(argv[2]);
    if (prompt.empty() || threads < 1 || generate < 1) {
        std::fprintf(stderr, "error: need at least one token, --threads >= 1 and --generate >= 1\n");
        return 2;
    }

    State s;
    s.out = argv[3];
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (model == nullptr) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    s.n_layer = llama_model_n_layer(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token t : prompt) {
        if (t >= n_vocab) {
            std::fprintf(stderr, "error: token %d is outside the vocabulary (%d)\n", t, n_vocab);
            return 1;
        }
    }
    const int n_prompt = static_cast<int>(prompt.size());
    const int n_steps = n_prompt + generate - 1;  // the last generated token is not decoded

    auto cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(n_steps);
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    // The defaults at the pinned tag, set explicitly so the dump doesn't follow a default change.
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = on_eval;
    cp.cb_eval_user_data = &s;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) return 1;

    const int per_step = s.n_layer * static_cast<int>(std::size(layer_wants)) + static_cast<int>(std::size(global_wants));
    std::vector<llama_token> argmaxes, generated;
    llama_token tok = prompt[0];
    for (int pos = 0; pos < n_steps; ++pos) {
        if (pos < n_prompt) tok = prompt[static_cast<size_t>(pos)];
        s.pos = pos;
        s.dump_layers = pos < n_prompt;
        s.captured = 0;
        s.logits.clear();
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (const int rc = llama_decode(ctx, batch); rc != 0) {
            std::fprintf(stderr, "error: llama_decode at pos %d returned %d\n", pos, rc);
            return 1;
        }
        // Guards against a graph change: every wanted node seen once, and the dumped logits are the real ones.
        const int expect = s.dump_layers ? per_step : 1;
        if (s.captured != expect) {
            std::fprintf(stderr, "error: pos %d captured %d tensors, expected %d\n", pos, s.captured, expect);
            return 1;
        }
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (s.logits.size() != static_cast<size_t>(n_vocab) ||
            std::memcmp(lg, s.logits.data(), s.logits.size() * sizeof(float)) != 0) {
            std::fprintf(stderr, "error: pos %d: dumped result_output differs from llama_get_logits\n", pos);
            return 1;
        }
        tok = argmax(s.logits);
        argmaxes.push_back(tok);
        if (pos + 1 >= n_prompt) generated.push_back(tok);
    }
    if (s.nonfinite != 0) {
        std::fprintf(stderr, "error: %ld non-finite values\n", s.nonfinite);
        return 1;
    }

    std::string text;
    for (llama_token t : generated) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
        if (n > 0) text.append(buf, static_cast<size_t>(n));
    }

    // One entry per line keeps the committed manifest diffable; run.sh adds the sha256 fields.
    std::FILE * m = std::fopen((s.out / "manifest.json").c_str(), "w");
    if (m == nullptr) return 1;
    std::fprintf(m, "{\n\"llama_tag\": \"%s\",\n\"model\": \"%s\",\n", LLAMA_TAG,
                 std::filesystem::path(argv[1]).filename().c_str());
    std::fprintf(m, "\"threads\": %d,\n\"seed\": \"none: greedy argmax in ref-dump, no sampler\",\n", threads);
    std::fprintf(m, "\"n_ctx\": %u,\n\"type_k\": \"%s\",\n\"type_v\": \"%s\",\n\"flash_attn\": true,\n",
                 llama_n_ctx(ctx), ggml_type_name(cp.type_k), ggml_type_name(cp.type_v));
    std::fprintf(m, "\"prompt_tokens\": [");
    print_ids(m, prompt);
    std::fprintf(m, "],\n\"argmax\": [");
    print_ids(m, argmaxes);
    std::fprintf(m, "],\n\"generated\": [");
    print_ids(m, generated);
    std::fprintf(m, "],\n\"tensors\": [\n");
    for (size_t i = 0; i < s.entries.size(); ++i) {
        const Entry & e = s.entries[i];
        std::fprintf(m,
                     "{\"name\": \"%s\", \"layer\": %d, \"pos\": %d, \"shape\": [%lld, %lld, %lld, %lld], "
                     "\"dtype\": \"%s\", \"file\": \"%s\"}%s\n",
                     e.name.c_str(), e.layer, e.pos, static_cast<long long>(e.ne[0]), static_cast<long long>(e.ne[1]),
                     static_cast<long long>(e.ne[2]), static_cast<long long>(e.ne[3]), ggml_type_name(e.type),
                     e.file.c_str(), i + 1 < s.entries.size() ? "," : "");
    }
    std::fprintf(m, "]\n}\n");
    if (std::fclose(m) != 0) return 1;

    std::printf("generated: [");
    print_ids(stdout, generated);
    std::printf("]\ntext: %s\ntensors: %zu\n", text.c_str(), s.entries.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
