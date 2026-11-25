# Build integration

A user configures the project with `-DMODEL_GGUF=<file>` and builds the `engine`. CMake runs
gguf2hpp to write `generated/model.hpp` before compiling the engine, and later edits rerun
exactly the steps they affect.

## Sub-features

- `build-generate`: configuring with `MODEL_GGUF` and building `engine` runs
  `Generating model.hpp from <file>`.
- `build-run`: the engine binary runs and prints the architecture name, the layer count and
  the tensor count, all read from `model.hpp` at compile time.
- `build-regen-model`: touching the GGUF reruns generation and recompiles `main.cpp`.
- `build-regen-tool`: changing the tool's sources rebuilds gguf2hpp, reruns generation, then
  recompiles `main.cpp`.
- `build-main-only`: touching `src/main.cpp` rebuilds the engine without rerunning
  generation.

## How to get to it (user POV)

- Run `cmake -S . -B <dir> -DMODEL_GGUF=$PWD/models/Qwen_Qwen3-0.6B-bf16.gguf`, then
  `cmake --build <dir> --target engine`, then `<dir>/engine`.

## Driving it with verify.sh

Preconditions:

- Doctor passes.
- `S=${TMPDIR:-/tmp}/megakernel-verify/<id>`.
- Use a scratch copy of the model input for the touch steps:
  `cp tests/data/toy.gguf $S/toy-copy.gguf`. Never touch `tests/data/` itself.

- **Generate and run.** Run `RUN_ID=<id> $V engine $S/toy-copy.gguf`. Expect the line
  `Generating model.hpp from .../toy-copy.gguf`, and stdout `toy: 2 layers, 2 tensors`.
  For Qwen, pass the model path instead and expect `qwen3: 28 layers, 311 tensors`.
- **No change.** Run `cmake --build $S/engine-build --target engine`. Expect no
  `Generating`, `Building` or `Linking` lines.
- **Model changed.** Run `touch $S/toy-copy.gguf`, then the same build. Expect
  `Generating model.hpp`, then `Building ... main.cpp.o` and `Linking ... engine`. Without
  the `Building` line, the engine's dependency on the header is broken.
- **Tool changed.** Run `touch tools/gguf_header/emitter.cpp`, then the same build. Expect
  `Building ... emitter.cpp.o`, `Linking ... gguf2hpp`, `Generating model.hpp`, then
  `Building ... main.cpp.o` and `Linking ... engine`.
- **Main changed.** Run `touch src/main.cpp`, then the same build. Expect
  `Building ... main.cpp.o` and `Linking ... engine`, with no `Generating` line.
- **Proof.** `engine.build.log`, `engine.stdout`, and the build lines from each touch step.

## Gotchas

- `touch` on `src/main.cpp` or `emitter.cpp` changes only the file's mtime, but the user's
  own `build/` will rebuild those files on its next build. Say so in the report.
- The `engine` target exists only when `MODEL_GGUF` is set. Configuring without it prints a
  CMake warning, and must still build every test target and pass ctest.
- `main.cpp` must not use `model::layers`: a file without `blk.<n>.*` tensors (toy) has no
  layer table, and the engine must still build for it.
- Pass an absolute `MODEL_GGUF`. `engine` resolves it with `realpath`; by hand, use `$PWD/...`.
