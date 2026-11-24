#include "emitter.hpp"

#include "reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace gguf {
namespace {
// [A-Za-z_][A-Za-z0-9_]*. Spelled out in ASCII: <cctype> follows the locale.
bool is_identifier(std::string_view s) {
    const auto alpha = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    };
    const auto digit = [](char c) { return c >= '0' && c <= '9'; };
    return !s.empty()
           && alpha(s[0])
           && std::ranges::all_of(s, [&](char c) { return alpha(c) || digit(c); });
}

// The inside of a string literal that holds exactly the bytes of s. Strings come from an
// untrusted file and get compiled into the engine, so no byte of s may end the literal early.
std::string escape(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (const char ch : s) {
        // Bytes 0x80 to 0xFF must count as 128 to 255, not as negative.
        const unsigned b = static_cast<unsigned char>(ch);
        // '?' too: "??/" is a trigraph for a backslash. GCC warns about one under -Wall, and
        // -trigraphs would switch it on.
        if (b >= 0x20 && b <= 0x7E && ch != '"' && ch != '\\' && ch != '?') {
            out += ch;
        } else {
            // Octal stops after three digits. \x never stops: "\x41" then "B" would read as \x41B.
            out += std::format("\\{:03o}", b);
        }
    }
    return out;
}

// The mk::DType enumerator with the same id. Empty for an id that is not in the enum.
std::string_view name(TensorType t) {
    using enum TensorType;
    switch (t) {
    case F32:
        return "F32";
    case F16:
        return "F16";
    case Q4_0:
        return "Q4_0";
    case Q4_1:
        return "Q4_1";
    case Q5_0:
        return "Q5_0";
    case Q5_1:
        return "Q5_1";
    case Q8_0:
        return "Q8_0";
    case Q8_1:
        return "Q8_1";
    case Q2_K:
        return "Q2_K";
    case Q3_K:
        return "Q3_K";
    case Q4_K:
        return "Q4_K";
    case Q5_K:
        return "Q5_K";
    case Q6_K:
        return "Q6_K";
    case Q8_K:
        return "Q8_K";
    case IQ2_XXS:
        return "IQ2_XXS";
    case IQ2_XS:
        return "IQ2_XS";
    case IQ3_XXS:
        return "IQ3_XXS";
    case IQ1_S:
        return "IQ1_S";
    case IQ4_NL:
        return "IQ4_NL";
    case IQ3_S:
        return "IQ3_S";
    case IQ2_S:
        return "IQ2_S";
    case IQ4_XS:
        return "IQ4_XS";
    case I8:
        return "I8";
    case I16:
        return "I16";
    case I32:
        return "I32";
    case I64:
        return "I64";
    case F64:
        return "F64";
    case IQ1_M:
        return "IQ1_M";
    case BF16:
        return "BF16";
    case TQ1_0:
        return "TQ1_0";
    case TQ2_0:
        return "TQ2_0";
    case MXFP4:
        return "MXFP4";
    case NVFP4:
        return "NVFP4";
    case Q1_0:
        return "Q1_0";
    }
    return {};
}

// What a value type is called in the generated header. Both are empty for an unknown id.
struct ValueTypeNames {
    std::string_view enumerator; // in mk::ValueType
    std::string_view cpp;        // the type of a constant that holds one value
};

ValueTypeNames names(ValueType t) {
    using enum ValueType;
    switch (t) {
    case Uint8:
        return {"Uint8", "std::uint8_t"};
    case Int8:
        return {"Int8", "std::int8_t"};
    case Uint16:
        return {"Uint16", "std::uint16_t"};
    case Int16:
        return {"Int16", "std::int16_t"};
    case Uint32:
        return {"Uint32", "std::uint32_t"};
    case Int32:
        return {"Int32", "std::int32_t"};
    case Float32:
        return {"Float32", "float"};
    case Bool:
        return {"Bool", "bool"};
    case String:
        return {"String", "std::string_view"};
    case Array:
        return {"Array", "mk::Array"};
    case Uint64:
        return {"Uint64", "std::uint64_t"};
    case Int64:
        return {"Int64", "std::int64_t"};
    case Float64:
        return {"Float64", "double"};
    }
    return {};
}

// Key "a.b.c" as the C++ names it becomes.
struct KeyPath {
    std::string      ns;   // "a::b"; empty for a key with no dot
    std::string_view leaf; // "c"
};

// nullopt unless every dot-separated segment is an identifier. An empty segment is not one, so
// "a..b", "a." and "" are rejected too.
// shortcut: a segment that is a C++ keyword, a macro, or std or mk (names the header's own lines
// use), a first segment "arch" (the alias emit() adds), and a key that is also the prefix of
// another ("a.b" and "a.b.c"), pass; the generated header then fails to compile, which is loud.
// Reject them here when a real model has such a key.
std::optional<KeyPath> split_key(std::string_view key) {
    KeyPath path;
    for (;;) {
        const std::size_t      dot = key.find('.');
        const std::string_view seg = key.substr(0, dot); // no dot left: the rest of the key
        if (!is_identifier(seg)) {
            return std::nullopt;
        }
        if (dot == std::string_view::npos) {
            path.leaf = seg;
            return path;
        }
        if (!path.ns.empty()) {
            path.ns += "::";
        }
        path.ns += seg;
        key.remove_prefix(dot + 1);
    }
}

// One line that names the key. Escaped: the key is untrusted and may hold a newline.
std::unexpected<std::string> key_error(std::string_view key, std::string_view why) {
    return std::unexpected(std::format("metadata key \"{}\": {}", escape(key), why));
}

// Whether v fits the integer type t names. The reader widened every integer to 64 bits; one that
// does not fit its own type would compile to a different number.
bool fits(ValueType t, auto v) {
    switch (t) {
    case ValueType::Uint8:
        return std::in_range<std::uint8_t>(v);
    case ValueType::Int8:
        return std::in_range<std::int8_t>(v);
    case ValueType::Uint16:
        return std::in_range<std::uint16_t>(v);
    case ValueType::Int16:
        return std::in_range<std::int16_t>(v);
    case ValueType::Uint32:
        return std::in_range<std::uint32_t>(v);
    case ValueType::Int32:
        return std::in_range<std::int32_t>(v);
    default:
        return true; // 64 bits wide
    }
}

// The C++ text for kv's value. kv.type picks the case, so the original width decides the form.
// A value that is not of the kind kv.type says, or does not fit it, is an error.
std::expected<std::string, std::string> literal(const KeyValue& kv) {
    switch (kv.type) {
    case ValueType::Uint8:
    case ValueType::Uint16:
    case ValueType::Uint32:
    case ValueType::Uint64:
        if (const auto* v = std::get_if<std::uint64_t>(&kv.val); v && fits(kv.type, *v)) {
            // Without the u, a decimal literal above INT64_MAX has no unsigned type.
            return std::format("{}u", *v);
        }
        break;
    case ValueType::Int8:
    case ValueType::Int16:
    case ValueType::Int32:
    case ValueType::Int64:
        if (const auto* v = std::get_if<std::int64_t>(&kv.val); v && fits(kv.type, *v)) {
            // "-9223372036854775808" is a minus applied to a literal that does not fit in i64.
            if (*v == std::numeric_limits<std::int64_t>::min()) {
                return "(-9223372036854775807 - 1)";
            }
            return std::format("{}", *v);
        }
        break;
    case ValueType::Float32:
    case ValueType::Float64:
        if (const auto* v = std::get_if<double>(&kv.val)) {
            if (!std::isfinite(*v)) {
                return key_error(kv.key, "inf and NaN have no literal");
            }
            // The shortest text that reads back to the same value, always with a decimal
            // point: "2." is a floating literal, "2" is an int.
            if (kv.type == ValueType::Float32) {
                // Checked before narrowing: a double past the float range has no float value.
                if (std::abs(*v) > double{std::numeric_limits<float>::max()}) {
                    break;
                }
                // Back to float first: the shortest text for the widened double is longer.
                return std::format("{:#}f", static_cast<float>(*v));
            }
            return std::format("{:#}", *v);
        }
        break;
    case ValueType::Bool:
        if (const auto* v = std::get_if<bool>(&kv.val)) {
            return *v ? "true" : "false";
        }
        break;
    case ValueType::String:
        if (const auto* v = std::get_if<std::string_view>(&kv.val)) {
            // sv carries the length: a string_view made from a plain literal stops at a NUL.
            return std::format("\"{}\"sv", escape(*v));
        }
        break;
    case ValueType::Array:
        if (const auto* v = std::get_if<ArrayRef>(&kv.val)) {
            const std::string_view elem = names(v->element_type).enumerator;
            if (!elem.empty()) {
                return std::format("{{mk::ValueType::{}, {}u, {}u}}", elem, v->count, v->pos);
            }
        }
        break;
    }
    return key_error(kv.key, "value does not match its type");
}
} // namespace

std::expected<std::string, std::string> emit(const Model& m, std::string_view source) {
    // source is escaped too: a newline in a file name would end the comment and start code.
    // The header includes every standard header it uses itself, not through model_types.hpp.
    std::string out = std::format("// Generated by gguf2hpp from {}. Do not edit.\n"
                                  "#pragma once\n"
                                  "#include \"model_types.hpp\"\n"
                                  "\n"
                                  "#include <array>\n"
                                  "#include <cstdint>\n"
                                  "#include <string_view>\n"
                                  "\n"
                                  "namespace model {{\n"
                                  "using namespace std::string_view_literals;\n"
                                  "\n"
                                  "inline constexpr std::uint64_t file_size  = {}u;\n"
                                  "inline constexpr std::uint64_t data_start = {}u;\n"
                                  "inline constexpr std::uint64_t alignment  = {}u;\n"
                                  "inline constexpr std::uint64_t hash       = {:#018x}u;\n"
                                  "\n"
                                  "namespace meta {{\n",
                                  escape(source),
                                  m.file_size,
                                  m.data_start,
                                  m.alignment,
                                  m.hash);

    // One line per key, in file order. Each line reopens its namespace; nothing is grouped.
    const std::string_view* arch = nullptr; // general.architecture, when it is a string
    for (const KeyValue& kv : m.metadata) {
        const auto path = split_key(kv.key);
        if (!path) {
            return key_error(kv.key, "every part between dots must be a C++ identifier");
        }
        const auto value = literal(kv);
        if (!value) {
            return std::unexpected(value.error());
        }
        // Scalars read "T name = v;", arrays "mk::Array name{...};".
        const std::string decl = std::format("inline constexpr {} {}{}{};",
                                             names(kv.type).cpp,
                                             path->leaf,
                                             kv.type == ValueType::Array ? "" : " = ",
                                             *value);
        out += path->ns.empty() ? decl + "\n"
                                : std::format("namespace {} {{ {} }}\n", path->ns, decl);

        if (kv.key == "general.architecture") {
            arch = std::get_if<std::string_view>(&kv.val);
        }
    }

    // meta::arch is the architecture's own namespace ("qwen3"), so the engine need not name it.
    // Only when that namespace exists: an alias to an undeclared namespace does not compile.
    if (arch && is_identifier(*arch)) {
        const std::string prefix = std::format("{}.", *arch);
        if (std::ranges::any_of(m.metadata,
                                [&](const KeyValue& kv) { return kv.key.starts_with(prefix); })) {
            out += std::format("namespace arch = {};\n", *arch);
        }
    }
    out += "} // namespace meta\n\n";

    if (m.tensors.empty()) {
        // Not "{{ }}": that relies on how the library lays out a zero-size array.
        out += "inline constexpr std::array<mk::Tensor, 0> tensors{};\n";
    } else {
        out += std::format("inline constexpr std::array<mk::Tensor, {}> tensors{{{{\n",
                           m.tensors.size());
        for (const TensorInfo& t : m.tensors) {
            const std::string_view dtype = name(t.dtype);
            if (dtype.empty()) {
                return std::unexpected(std::format("tensor \"{}\": unknown dtype {}",
                                                   escape(t.name),
                                                   std::to_underlying(t.dtype)));
            }
            // u on every number: a dim can exceed INT64_MAX.
            out += std::format("    {{\"{}\"sv, {}u, {{{}u, {}u, {}u, {}u}}, "
                               "mk::DType::{}, {}u, {}u}},\n",
                               escape(t.name),
                               t.n_dims,
                               t.dims[0],
                               t.dims[1],
                               t.dims[2],
                               t.dims[3],
                               dtype,
                               t.offset,
                               t.byte_size);
        }
        out += "}};\n";
    }
    out += "} // namespace model\n";
    return out;
}
} // namespace gguf
