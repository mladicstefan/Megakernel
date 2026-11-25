#include "model.hpp"

#include <print>

int main() {
    std::println("{}: {} layers, {} tensors", model::meta::general::architecture,
                 model::meta::arch::block_count, model::tensors.size());
}
