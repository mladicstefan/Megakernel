#pragma once
#include "reader.hpp"

#include <string>

namespace gguf {
// The text of the generated header for m.
std::string emit(const Model& m);
} // namespace gguf
