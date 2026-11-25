# Generate a header

A user runs `gguf2hpp <in.gguf> <out.hpp>` and gets a C++ header of constexpr model facts
that compiles as host C++23 and as CUDA C++20. Running it twice on the same input gives the
same bytes.

## Sub-features

- `gen-toy`: toy.gguf gives a header with 2 tensors and its metadata.
- `gen-qwen`: the Qwen model gives a header with 311 tensors and a 28-entry layer table.
- `gen-deterministic`: two runs give byte-identical output.
- `gen-compiles`: the header compiles with g++ (warnings as errors) and with nvcc, and
  `mk::find` resolves a tensor at compile time.

## How to get to it (user POV)

- Run `gguf2hpp <in.gguf> <out.hpp>` from a shell.
- CMake runs the same command for `toy.hpp` (tests) and `model.hpp` (engine); see
  build-integration.md.

## Driving it with verify.sh

Preconditions:

- Doctor passes. The Qwen entry needs `ok Qwen model present`.

- **Toy.** Run `RUN_ID=<id> $V gen tests/data/toy.gguf toy`. Expect `exit=0`, a 1097-byte
  output and `tmp-left=no`. The header contains `std::array<mk::Tensor, 2> tensors`,
  `"a"sv` with dims `{3u, 2u, 1u, 1u}`, and
  `namespace toy { inline constexpr float scale = 0.5f; }`.
- **Toy compiles.** Run `RUN_ID=<id> $V compile build/verify-evidence/<id>/toy.hpp toy`.
  Expect `g++=0 nvcc=0` and `probe: tensors=2 first=a bytes=24`.
- **Qwen.** Run `RUN_ID=<id> $V gen models/Qwen_Qwen3-0.6B-bf16.gguf qwen`. Expect `exit=0`
  and 43387 bytes at `ea2e28f`; the size changes with the emitter. The header contains
  `std::array<mk::Tensor, 311> tensors`, `struct Layer` and `layers[28]`.
- **Qwen compiles.** Run `RUN_ID=<id> $V compile build/verify-evidence/<id>/qwen.hpp qwen`.
  Expect `g++=0 nvcc=0` and `probe: tensors=311 first=output.weight bytes=311164928`.
- **Deterministic.** Run `gen` on the same input as `qwen-b`, then
  `cmp build/verify-evidence/<id>/qwen.hpp build/verify-evidence/<id>/qwen-b.hpp`. Expect
  no output and exit 0.

## Gotchas

- Tensors come in file order, so `first` for Qwen is `output.weight`, not
  `token_embd.weight`. Layers are in numeric order in `model::layers`, not in the tensor
  table.
- The header size is a fingerprint of the emitter version. Update the expected byte counts
  here when the emitter changes on purpose.
- The launch build is Debug. Don't report gguf2hpp timings from it as performance. On Qwen,
  one `gen` call takes about 50 ms of wall time, script included.
- `compile` checks that the header builds and that one consteval lookup works. It doesn't
  check the values. The value checks live in `tests/emitter_test.cpp`; this map proves the
  binary and the compilers agree.
