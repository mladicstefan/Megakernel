# Build integration

A user configures the project with `-DMODEL_GGUF=<file>` and builds the `engine`. CMake runs
gguf2hpp to write `generated/model.hpp` before compiling the engine, and later edits rerun
exactly the steps they affect.

## Sub-features

- `build-generate`: configuring with `MODEL_GGUF` and building `engine` runs
  `Generating model.hpp from <file>`.
- `build-run`: the engine binary runs. Before milestone 8 it prints `Hello`. After it, it
  prints the layer and tensor counts (28 and 311 for Qwen).
- `build-regen-model`: touching the GGUF reruns generation.
- `build-regen-tool`: changing the tool's sources rebuilds gguf2hpp, then reruns generation.
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
  `Generating model.hpp from .../toy-copy.gguf`, and stdout `Hello` (before milestone 8).
  For Qwen, pass the model path instead, and after milestone 8 expect 28 and 311 in
  `engine.stdout`.
- **No change.** Run `cmake --build $S/engine-build --target engine`. Expect no
  `Generating`, `Building` or `Linking` lines.
- **Model changed.** Run `touch $S/toy-copy.gguf`, then the same build. Expect
  `Generating model.hpp`.
- **Tool changed.** Run `touch tools/gguf_header/emitter.cpp`, then the same build. Expect
  `Building ... emitter.cpp.o`, `Linking ... gguf2hpp`, then `Generating model.hpp`.
- **Main changed.** Run `touch src/main.cpp`, then the same build. Expect
  `Building ... main.cpp.o` and `Linking ... engine`, with no `Generating` line.
- **Proof.** `engine.build.log`, `engine.stdout`, and the build lines from each touch step.

## Gotchas

- Before milestone 8, `src/main.cpp` doesn't include `model.hpp`. A regenerated header then
  doesn't recompile the engine, which is expected. After milestone 8, "Model changed" must
  also show `Building ... main.cpp.o`; if it doesn't, the dependency is broken.
- `touch` on `src/main.cpp` or `emitter.cpp` changes only the file's mtime, but the user's
  own `build/` will rebuild those files on its next build. Say so in the report.
- Configuring without `MODEL_GGUF` prints a CMake warning. If milestone 8 moves `engine`
  inside `if(MODEL_GGUF)`, a plain configure must still build every test target.
- Pass an absolute `MODEL_GGUF`. `engine` resolves it with `realpath`; by hand, use `$PWD/...`.
