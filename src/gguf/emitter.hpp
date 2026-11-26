#pragma once
#include "gguf/reader.hpp"

#include <expected>
#include <string>
#include <string_view>

namespace gguf {
// The text of the generated header for m. source is the input's file name, shown in the first
// line. The error is one line that names the key or tensor that cannot be written as C++.
std::expected<std::string, std::string> emit(const Model& m, std::string_view source);
} // namespace gguf
