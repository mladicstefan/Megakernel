#pragma once
#include <array> // for the generated header's tensor table; it includes only this file
#include <cstdint>
#include <string_view>

// The types a generated model header is written in. CUDA code includes this too, and nvcc
// compiles it as C++20: nothing from C++23 in here.
namespace mk {
// Same names and ids as gguf::TensorType in the tool, gaps included (emitter_test checks).
enum class DType : std::uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
    MXFP4   = 39,
    NVFP4   = 40,
    Q1_0    = 41,
};

// Same names and ids as gguf::ValueType in the tool.
enum class ValueType : std::uint32_t {
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Uint32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    Uint64  = 10,
    Int64   = 11,
    Float64 = 12,
};

struct Tensor {
    std::string_view             name;
    std::uint32_t                n_dims;
    // ggml order; unused dims are 1. A plain array: device code cannot index a std::array.
    std::uint64_t                dims[4];
    DType                        dtype;
    std::uint64_t                offset; // from model::data_start
    std::uint64_t                bytes;
};

// A metadata array: its elements stay in the file, count of them starting at byte pos.
struct Array {
    ValueType     element_type;
    std::uint64_t count;
    std::uint64_t pos;
};

// Never defined: find() calls it for a missing name, so the compile error prints this name.
[[noreturn]] void tensor_name_not_found();

// The tensor called name. A name that is not in the table fails the compile.
// A copy, not a reference: a reference would make device code refer to the host's table.
consteval Tensor find(const auto& tensors, std::string_view name) {
    for (const Tensor& t : tensors) {
        if (t.name == name) {
            return t;
        }
    }
    tensor_name_not_found();
}
} // namespace mk
