#include "emitter.hpp"
#include "model_types.hpp"
#include "reader.hpp"
#include "toy.hpp" // generated from tests/data/toy.gguf by gguf2hpp while building

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

using namespace gguf;
using namespace std::string_view_literals;

// --- Compile time: the facts of toy.gguf, read through the generated header. ---

static_assert(model::file_size == 384u);
static_assert(model::data_start == 320u);
static_assert(model::alignment == 32u);
static_assert(model::hash == 0x2a6d1ca21af41e12u);

static_assert(model::meta::general::architecture == "toy");
static_assert(model::meta::arch::block_count == 2u); // arch is an alias of meta::toy
static_assert(model::meta::toy::scale == 0.5f);
static_assert(model::meta::toy::note == "hello");
static_assert(model::meta::toy::flag);
static_assert(model::meta::toy::list.element_type == mk::ValueType::Int32);
static_assert(model::meta::toy::list.count == 3u);
static_assert(model::meta::toy::list.pos == 0xd1u);  // the 10 of [10, 20, 30] in the file

// The constants have the width the file stored, not the widened type the reader holds.
static_assert(std::same_as<decltype(model::meta::toy::block_count), const std::uint32_t>);
static_assert(std::same_as<decltype(model::meta::toy::scale), const float>);

static_assert(model::tensors.size() == 2);

static_assert(model::tensors[0].name == "a");
static_assert(model::tensors[0].n_dims == 2u);
static_assert(std::ranges::equal(model::tensors[0].dims, std::array{3u, 2u, 1u, 1u}));
static_assert(model::tensors[0].dtype == mk::DType::F32);
static_assert(model::tensors[0].offset == 0u);
static_assert(model::tensors[0].bytes == 24u);

static_assert(model::tensors[1].name == "b");
static_assert(model::tensors[1].n_dims == 1u);
static_assert(std::ranges::equal(model::tensors[1].dims, std::array{4u, 1u, 1u, 1u}));
static_assert(model::tensors[1].dtype == mk::DType::F16);
static_assert(model::tensors[1].offset == 32u);
static_assert(model::tensors[1].bytes == 8u);

static_assert(mk::find(model::tensors, "b").offset == 32u);
// A name that is not in the table does not compile. Uncomment to see GCC say
// "call to non-'constexpr' function 'void mk::tensor_name_not_found()'":
// static_assert(mk::find(model::tensors, "nope").offset == 0u);

// The real model, when it was there at configure time (see CMakeLists.txt): toy.gguf has no
// layers. Wrapped in a namespace because its model:: would collide with toy's; that works only
// while every header it includes was already included above.
#if __has_include("qwen.hpp")
namespace qwen {
#include "qwen.hpp"
} // namespace qwen

static_assert(std::size(qwen::model::layers) == qwen::model::meta::arch::block_count);
// Numeric order: in the file, and so in tensors, "blk.10" comes before "blk.2".
static_assert(qwen::model::layers[2].q.name == "blk.2.attn_q.weight");
static_assert(qwen::model::layers[10].q.name == "blk.10.attn_q.weight");
static_assert(qwen::model::layers[27].down.name == "blk.27.ffn_down.weight");
static_assert(qwen::model::layers[5].o.name == "blk.5.attn_output.weight");
static_assert(qwen::model::layers[5].attn_norm.name == "blk.5.attn_norm.weight");
static_assert(qwen::model::layers[0].q.dims[0] == qwen::model::meta::arch::embedding_length);
static_assert(qwen::model::layers[0].q.offset
              == mk::find(qwen::model::tensors, "blk.0.attn_q.weight").offset);
#endif

// Generated strings end in sv so that they keep the bytes after a NUL.
static_assert("a\000b"sv.size() == 3 && std::string_view{"a\000b"}.size() == 1);

// --- Run time: emit() on hand-built models. The views point at literals; no file is needed. ---

namespace {
std::string emitted(const Model& m) {
    const auto r = emit(m, "t.gguf");
    if (!r) {
        ADD_FAILURE() << "emit failed: " << r.error();
        return {};
    }
    return *r;
}

std::string error_of(const Model& m) {
    const auto r = emit(m, "t.gguf");
    if (r) {
        ADD_FAILURE() << "emit succeeded:\n" << *r;
        return {};
    }
    return r.error();
}

Model with_key(std::string_view key, ValueType type, MetaValue val) {
    Model m{};
    m.metadata.push_back({key, type, val});
    return m;
}

Model with_tensor(std::string_view name, TensorType dtype = TensorType::F32) {
    Model m{};
    m.tensors.push_back({name, 1, {4, 1, 1, 1}, dtype, 0, 4, 16});
    return m;
}

Model with_tensors(std::initializer_list<std::string_view> names) {
    Model m{};
    for (const std::string_view name : names) {
        m.tensors.push_back({name, 1, {4, 1, 1, 1}, TensorType::F32, 0, 4, 16});
    }
    return m;
}

// The line of text that declares the one key of with_key(...), without its newline.
std::string key_line(std::string_view key, ValueType type, MetaValue val) {
    constexpr std::string_view opening = "namespace meta {\n";
    const std::string          text    = emitted(with_key(key, type, val));
    const std::size_t          at      = text.find(opening);
    if (at == std::string::npos) {
        return {};
    }
    const std::size_t start = at + opening.size();
    return text.substr(start, text.find('\n', start) - start);
}
} // namespace

TEST(Emit, ToyGolden) {
    auto f = MappedFile::open(TOY_GGUF);
    ASSERT_TRUE(f.has_value());
    auto m = parse(f->bytes());
    ASSERT_TRUE(m.has_value());
    auto text = emit(*m, "toy.gguf");
    ASSERT_TRUE(text.has_value()) << text.error();
    EXPECT_EQ(*text, R"(// Generated by gguf2hpp from toy.gguf. Do not edit.
#pragma once
#include "model_types.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace model {
using namespace std::string_view_literals;

inline constexpr std::uint64_t file_size  = 384u;
inline constexpr std::uint64_t data_start = 320u;
inline constexpr std::uint64_t alignment  = 32u;
inline constexpr std::uint64_t hash       = 0x2a6d1ca21af41e12u;

namespace meta {
namespace general { inline constexpr std::string_view architecture = "toy"sv; }
namespace toy { inline constexpr std::uint32_t block_count = 2u; }
namespace toy { inline constexpr float scale = 0.5f; }
namespace toy { inline constexpr std::string_view note = "hello"sv; }
namespace toy { inline constexpr bool flag = true; }
namespace toy { inline constexpr mk::Array list{mk::ValueType::Int32, 3u, 209u}; }
namespace arch = toy;
} // namespace meta

inline constexpr std::array<mk::Tensor, 2> tensors{{
    {"a"sv, 2u, {3u, 2u, 1u, 1u}, mk::DType::F32, 0u, 24u},
    {"b"sv, 1u, {4u, 1u, 1u, 1u}, mk::DType::F16, 32u, 8u},
}};
} // namespace model
)");
}

TEST(Emit, EmptyModel) {
    EXPECT_EQ(emitted(Model{}), R"(// Generated by gguf2hpp from t.gguf. Do not edit.
#pragma once
#include "model_types.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace model {
using namespace std::string_view_literals;

inline constexpr std::uint64_t file_size  = 0u;
inline constexpr std::uint64_t data_start = 0u;
inline constexpr std::uint64_t alignment  = 0u;
inline constexpr std::uint64_t hash       = 0x0000000000000000u;

namespace meta {
} // namespace meta

inline constexpr std::array<mk::Tensor, 0> tensors{};
} // namespace model
)");
}

TEST(Emit, SameInputSameOutput) {
    Model m = with_tensor("w");
    m.metadata.push_back({"a.b", ValueType::Float64, 0.1});
    EXPECT_EQ(emitted(m), emitted(m));
}

// Quote, backslash, newline, byte 0xFF, then AB: a \x escape would swallow the A and the B.
TEST(Emit, EscapesEveryString) {
    const std::string_view raw  = "q\"b\\n\nx\377AB";
    const std::string_view want = R"("q\042b\134n\012x\377AB"sv)";
    EXPECT_TRUE(emitted(with_tensor(raw)).contains(std::format("    {{{}, 1u,", want)));
    EXPECT_EQ(key_line("a.s", ValueType::String, raw),
              std::format("namespace a {{ inline constexpr std::string_view s = {}; }}", want));
}

// "??/" is a trigraph for a backslash: GCC warns about it, and -trigraphs would turn it on.
TEST(Emit, EscapesQuestionMarks) {
    EXPECT_TRUE(emitted(with_tensor("why?\?/")).contains(R"("why\077\077/"sv)"));
}

TEST(Emit, KeepsBytesAfterNul) {
    EXPECT_TRUE(emitted(with_tensor("a\000b"sv)).contains(R"("a\000b"sv)"));
}

// The file name lands in a // comment, where a newline would start a line of code.
TEST(Emit, EscapesSourceName) {
    const auto text = emit(Model{}, "x\n#error.gguf");
    ASSERT_TRUE(text.has_value());
    EXPECT_TRUE(
        text->starts_with("// Generated by gguf2hpp from x\\012#error.gguf. Do not edit.\n"));
    EXPECT_FALSE(text->contains("\n#error"));
}

TEST(Emit, KeyBecomesNestedNamespace) {
    EXPECT_EQ(key_line("a.b.c", ValueType::Bool, true),
              "namespace a::b { inline constexpr bool c = true; }");
    EXPECT_EQ(key_line("_a1.B_2", ValueType::Bool, false),
              "namespace _a1 { inline constexpr bool B_2 = false; }");
    // No dot: straight into meta.
    EXPECT_EQ(key_line("flag", ValueType::Bool, true), "inline constexpr bool flag = true;");
}

TEST(Emit, RejectsKeysThatAreNotIdentifiers) {
    for (const std::string_view key : {"a..b", "1a.b", "a.b-c", "a.", "", "a b", ".a", "a.1"}) {
        const std::string e = error_of(with_key(key, ValueType::Bool, true));
        EXPECT_TRUE(e.contains(std::format("\"{}\"", key))) << e;
        EXPECT_FALSE(e.contains('\n')) << e;
    }
}

// The message stays on one line even when the key does not.
TEST(Emit, ErrorEscapesTheKey) {
    const std::string e = error_of(with_key("a\nb", ValueType::Bool, true));
    EXPECT_TRUE(e.contains(R"("a\012b")")) << e;
    EXPECT_FALSE(e.contains('\n')) << e;
}

TEST(Emit, IntegersKeepTheirWidth) {
    EXPECT_EQ(key_line("a.x", ValueType::Uint8, std::uint64_t{255}),
              "namespace a { inline constexpr std::uint8_t x = 255u; }");
    EXPECT_EQ(key_line("a.x", ValueType::Int8, std::int64_t{-128}),
              "namespace a { inline constexpr std::int8_t x = -128; }");
    EXPECT_EQ(key_line("a.x", ValueType::Uint16, std::uint64_t{65535}),
              "namespace a { inline constexpr std::uint16_t x = 65535u; }");
    EXPECT_EQ(key_line("a.x", ValueType::Int16, std::int64_t{-1}),
              "namespace a { inline constexpr std::int16_t x = -1; }");
    EXPECT_EQ(key_line("a.x", ValueType::Int32, std::int64_t{-2147483647 - 1}),
              "namespace a { inline constexpr std::int32_t x = -2147483648; }");
    // Above INT64_MAX: only the u makes this an unsigned literal.
    EXPECT_EQ(key_line("a.x", ValueType::Uint64, std::numeric_limits<std::uint64_t>::max()),
              "namespace a { inline constexpr std::uint64_t x = 18446744073709551615u; }");
    EXPECT_EQ(key_line("a.x", ValueType::Int64, std::numeric_limits<std::int64_t>::max()),
              "namespace a { inline constexpr std::int64_t x = 9223372036854775807; }");
    EXPECT_EQ(key_line("a.x", ValueType::Int64, std::numeric_limits<std::int64_t>::min()),
              "namespace a { inline constexpr std::int64_t x = (-9223372036854775807 - 1); }");
}

TEST(Emit, FloatsAreValidLiterals) {
    EXPECT_EQ(key_line("a.x", ValueType::Float32, 2.0),
              "namespace a { inline constexpr float x = 2.f; }"); // not "2f"
    EXPECT_EQ(key_line("a.x", ValueType::Float64, 2.0),
              "namespace a { inline constexpr double x = 2.; }");
    EXPECT_EQ(key_line("a.x", ValueType::Float64, 1e-6),
              "namespace a { inline constexpr double x = 1.e-06; }");
    EXPECT_EQ(key_line("a.x", ValueType::Float32, -0.0),
              "namespace a { inline constexpr float x = -0.f; }");
    // The reader widened 0.1f to the double 0.10000000149011612; the float's own text is 0.1.
    EXPECT_EQ(key_line("a.x", ValueType::Float32, static_cast<double>(0.1f)),
              "namespace a { inline constexpr float x = 0.1f; }");
    EXPECT_EQ(key_line("a.x", ValueType::Float64, 0.1),
              "namespace a { inline constexpr double x = 0.1; }");
    // More digits than a float holds: a double must not pass through float.
    EXPECT_EQ(key_line("a.x", ValueType::Float64, 1.0 / 3),
              "namespace a { inline constexpr double x = 0.3333333333333333; }");
}

TEST(Emit, RejectsNonFiniteFloats) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const ValueType type : {ValueType::Float32, ValueType::Float64}) {
        for (const double v : {inf, -inf, nan}) {
            const std::string e = error_of(with_key("toy.scale", type, v));
            EXPECT_TRUE(e.contains("\"toy.scale\"")) << e;
        }
    }
}

TEST(Emit, ArrayIsTypeCountAndPosition) {
    EXPECT_EQ(key_line("a.list", ValueType::Array, ArrayRef{ValueType::String, 151936, 5000}),
              "namespace a { inline constexpr mk::Array list{mk::ValueType::String, 151936u, "
              "5000u}; }");
}

// A value that is not of the kind its type says is reported, not emitted.
TEST(Emit, RejectsValueThatDoesNotMatchItsType) {
    EXPECT_TRUE(error_of(with_key("a.x", ValueType::Uint32, "text"sv)).contains("\"a.x\""));
    EXPECT_TRUE(error_of(with_key("a.x", ValueType::String, std::uint64_t{1})).contains("\"a.x\""));
    EXPECT_TRUE(error_of(with_key("a.x", static_cast<ValueType>(13), true)).contains("\"a.x\""));
    EXPECT_TRUE(
        error_of(with_key("a.x", ValueType::Array, ArrayRef{static_cast<ValueType>(13), 0, 0}))
            .contains("\"a.x\""));
}

// The reader never widens a value past its own type; written as is, these would compile to
// another number (300 as a uint8_t is 44) or to no number at all ("inff").
TEST(Emit, RejectsValueThatDoesNotFitItsType) {
    EXPECT_TRUE(
        error_of(with_key("a.x", ValueType::Uint8, std::uint64_t{256})).contains("\"a.x\""));
    EXPECT_TRUE(error_of(with_key("a.x", ValueType::Int8, std::int64_t{-129})).contains("\"a.x\""));
    EXPECT_TRUE(
        error_of(with_key("a.x", ValueType::Int16, std::int64_t{32768})).contains("\"a.x\""));
    EXPECT_TRUE(
        error_of(with_key("a.x", ValueType::Uint32, std::uint64_t{1} << 32)).contains("\"a.x\""));
    EXPECT_TRUE(error_of(with_key("a.x", ValueType::Float32, 1e300)).contains("\"a.x\""));
    EXPECT_TRUE(error_of(with_key("a.x", ValueType::Float32, -1e300)).contains("\"a.x\""));
}

// The tensors come in file order, which is text order: blk.10 sits between blk.1 and blk.2.
TEST(Emit, LayersAreInNumericOrder) {
    const std::string text = emitted(with_tensors({"token_embd.weight",
                                                   "blk.0.attn_q.weight",
                                                   "blk.1.attn_q.weight",
                                                   "blk.10.attn_q.weight",
                                                   "blk.2.attn_q.weight",
                                                   "blk.3.attn_q.weight",
                                                   "blk.4.attn_q.weight",
                                                   "blk.5.attn_q.weight",
                                                   "blk.6.attn_q.weight",
                                                   "blk.7.attn_q.weight",
                                                   "blk.8.attn_q.weight",
                                                   "blk.9.attn_q.weight"}));
    EXPECT_TRUE(text.ends_with(R"(}};

struct Layer { mk::Tensor q; };

// Device code cannot read layers[i] directly, only through a constexpr local:
//     constexpr Layer L = layers[3];
// For a layer index known only at run time, copy the table into a static
// constexpr local. Without static, every thread rebuilds the copy on its stack
// at each launch.
inline constexpr Layer layers[11] = {
    {tensors[1]},
    {tensors[2]},
    {tensors[4]},
    {tensors[5]},
    {tensors[6]},
    {tensors[7]},
    {tensors[8]},
    {tensors[9]},
    {tensors[10]},
    {tensors[11]},
    {tensors[3]},
};
} // namespace model
)")) << text;
}

// Fields come in the order the file first names them. Listed names get the short form; the
// rest keep the GGUF name, minus ".weight", with dots as underscores.
TEST(Emit, LayerFieldsAreNamedAfterTheTensors) {
    const std::string text = emitted(with_tensors({"blk.1.ffn_up.weight",
                                                   "blk.1.attn_norm.weight",
                                                   "blk.1.attn_output.weight",
                                                   "blk.1.attn_q.bias",
                                                   "blk.0.attn_q.bias",
                                                   "blk.0.attn_output.weight",
                                                   "blk.0.attn_norm.weight",
                                                   "blk.0.ffn_up.weight"}));
    EXPECT_TRUE(text.contains("struct Layer { mk::Tensor up, attn_norm, o, attn_q_bias; };\n"));
    EXPECT_TRUE(text.contains("inline constexpr Layer layers[2] = {\n"
                              "    {tensors[7], tensors[6], tensors[5], tensors[4]},\n"
                              "    {tensors[0], tensors[1], tensors[2], tensors[3]},\n"
                              "};\n"))
        << text;
}

TEST(Emit, NoBlockTensorsNoLayers) {
    for (const std::string_view name : {"output.weight", "blk.x.w", "blk.3", "blk..w", "blk"}) {
        EXPECT_FALSE(emitted(with_tensors({name})).contains("Layer")) << name;
    }
}

// The point of the table: a hole fails the generator, with the name of the tensor it wants.
TEST(Emit, RejectsMissingLayerTensor) {
    // One tensor of one layer.
    EXPECT_TRUE(error_of(with_tensors(
                             {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.1.attn_q.weight"}))
                    .contains("\"blk.1.attn_k.weight\""));
    EXPECT_TRUE(error_of(with_tensors(
                             {"blk.0.attn_q.weight", "blk.1.attn_q.weight", "blk.1.attn_k.weight"}))
                    .contains("\"blk.0.attn_k.weight\""));
    // A whole layer. The second index must not make the tool count that far.
    EXPECT_TRUE(error_of(with_tensors({"blk.0.w", "blk.2.w"})).contains("\"blk.1.w\""));
    EXPECT_TRUE(
        error_of(with_tensors({"blk.0.w", "blk.18446744073709551615.w"})).contains("\"blk.1.w\""));
    EXPECT_TRUE(error_of(with_tensors({"blk.1.w"})).contains("\"blk.0.w\""));
}

TEST(Emit, RejectsTwoTensorsForOneLayerField) {
    const std::string e = error_of(with_tensors({"blk.0.a.weight", "blk.0.a"}));
    EXPECT_TRUE(e.contains("\"blk.0.a")) << e;
}

TEST(Emit, RejectsLayerFieldThatIsNotAnIdentifier) {
    for (const std::string_view name : {"blk.0.a-b.weight", "blk.0.1x", "blk.3."}) {
        const std::string e = error_of(with_tensors({name}));
        EXPECT_TRUE(e.contains(std::format("\"{}\"", name))) << e;
    }
}

// A layer missing from the end leaves no hole; only the file's own count shows it.
TEST(Emit, RejectsLayerCountThatDisagreesWithBlockCount) {
    Model m = with_tensors({"blk.0.w", "blk.1.w"});
    m.metadata.push_back({"general.architecture", ValueType::String, "toy"sv});
    m.metadata.push_back({"toy.block_count", ValueType::Uint32, std::uint64_t{2}});
    EXPECT_TRUE(emitted(m).contains("layers[2]"));
    m.metadata[1].val   = std::uint64_t{3};
    const std::string e = error_of(m);
    EXPECT_TRUE(e.contains("\"toy.block_count\"")) << e;
}

TEST(Emit, RejectsUnknownDtype) {
    const std::string e = error_of(with_tensor("blk.0.w", static_cast<TensorType>(4)));
    EXPECT_TRUE(e.contains("\"blk.0.w\"")) << e;
}

TEST(Emit, TensorNumbersAreUnsigned) {
    Model m{};
    m.tensors.push_back({"t", 4, {1ull << 63, 2, 3, 4}, TensorType::I8, 1ull << 63, 0, 5});
    EXPECT_TRUE(emitted(m).contains("    {\"t\"sv, 4u, {9223372036854775808u, 2u, 3u, 4u}, "
                                    "mk::DType::I8, 9223372036854775808u, 5u},\n"));
}

TEST(Emit, ArchAliasNeedsANamespaceToPointAt) {
    const auto with_arch = [](MetaValue arch, std::string_view other_key) {
        Model m{};
        m.metadata.push_back({"general.architecture", ValueType::String, arch});
        m.metadata.push_back({other_key, ValueType::Bool, true});
        return m;
    };
    EXPECT_TRUE(emitted(with_arch("qwen3"sv, "qwen3.x")).contains("\nnamespace arch = qwen3;\n"));
    // No key under "qwen3.": "qwen3x.y" only shares the letters.
    EXPECT_FALSE(emitted(with_arch("qwen3"sv, "qwen3x.y")).contains("namespace arch"));
    EXPECT_FALSE(emitted(with_arch("qwen3"sv, "qwen3")).contains("namespace arch"));
    // Not an identifier.
    EXPECT_FALSE(emitted(with_arch("my-arch"sv, "my.x")).contains("namespace arch"));
    // Not an identifier, yet a key does start with "a.b.".
    EXPECT_FALSE(emitted(with_arch("a.b"sv, "a.b.c")).contains("namespace arch"));
    // No general.architecture at all.
    EXPECT_FALSE(emitted(with_key("toy.x", ValueType::Bool, true)).contains("namespace arch"));
    // Not a string.
    Model m{};
    m.metadata.push_back({"general.architecture", ValueType::Uint32, std::uint64_t{7}});
    EXPECT_FALSE(emitted(m).contains("namespace arch"));
}

// mk:: enums are separate from gguf:: on purpose. Per enumerator: both ids are equal (compile
// time), and the emitter writes that name for that id (run time), so its switch has no typo.
#define CHECK_DTYPE(n)                                                                             \
    static_assert(std::to_underlying(mk::DType::n) == std::to_underlying(TensorType::n));          \
    EXPECT_TRUE(emitted(with_tensor("t", TensorType::n)).contains(" mk::DType::" #n ","))

#define CHECK_VALUE_TYPE(n)                                                                        \
    static_assert(std::to_underlying(mk::ValueType::n) == std::to_underlying(ValueType::n));       \
    EXPECT_TRUE(emitted(with_key("a.list", ValueType::Array, ArrayRef{ValueType::n, 0, 0}))        \
                    .contains("{mk::ValueType::" #n ","))

TEST(ModelTypes, DTypeMatchesTensorType) {
    CHECK_DTYPE(F32);
    CHECK_DTYPE(F16);
    CHECK_DTYPE(Q4_0);
    CHECK_DTYPE(Q4_1);
    CHECK_DTYPE(Q5_0);
    CHECK_DTYPE(Q5_1);
    CHECK_DTYPE(Q8_0);
    CHECK_DTYPE(Q8_1);
    CHECK_DTYPE(Q2_K);
    CHECK_DTYPE(Q3_K);
    CHECK_DTYPE(Q4_K);
    CHECK_DTYPE(Q5_K);
    CHECK_DTYPE(Q6_K);
    CHECK_DTYPE(Q8_K);
    CHECK_DTYPE(IQ2_XXS);
    CHECK_DTYPE(IQ2_XS);
    CHECK_DTYPE(IQ3_XXS);
    CHECK_DTYPE(IQ1_S);
    CHECK_DTYPE(IQ4_NL);
    CHECK_DTYPE(IQ3_S);
    CHECK_DTYPE(IQ2_S);
    CHECK_DTYPE(IQ4_XS);
    CHECK_DTYPE(I8);
    CHECK_DTYPE(I16);
    CHECK_DTYPE(I32);
    CHECK_DTYPE(I64);
    CHECK_DTYPE(F64);
    CHECK_DTYPE(IQ1_M);
    CHECK_DTYPE(BF16);
    CHECK_DTYPE(TQ1_0);
    CHECK_DTYPE(TQ2_0);
    CHECK_DTYPE(MXFP4);
    CHECK_DTYPE(NVFP4);
    CHECK_DTYPE(Q1_0);
}

TEST(ModelTypes, ValueTypeMatches) {
    CHECK_VALUE_TYPE(Uint8);
    CHECK_VALUE_TYPE(Int8);
    CHECK_VALUE_TYPE(Uint16);
    CHECK_VALUE_TYPE(Int16);
    CHECK_VALUE_TYPE(Uint32);
    CHECK_VALUE_TYPE(Int32);
    CHECK_VALUE_TYPE(Float32);
    CHECK_VALUE_TYPE(Bool);
    CHECK_VALUE_TYPE(String);
    CHECK_VALUE_TYPE(Array);
    CHECK_VALUE_TYPE(Uint64);
    CHECK_VALUE_TYPE(Int64);
    CHECK_VALUE_TYPE(Float64);
}
