# Build integration

A user configures the project with `-DMODEL_GGUF=<file>` and builds the `engine`. CMake runs
gguf2hpp to write `generated/model.hpp` before compiling the engine, and later edits rerun
exactly the steps they affect.

## Sub-features

- `build-generate`: configuring with `MODEL_GGUF` and building `engine` runs
  `Generating model.hpp from <file>`.
- `build-run`: the engine binary runs. Its first line is the architecture name, the layer
  count and the tensor count, all read from `model.hpp` at compile time. The lines after it
  report the weights upload; see [weights-upload.md](./weights-upload.md).
- `build-regen-model`: touching the GGUF reruns generation and recompiles `weights.cu` and
  `main.cpp`.
- `build-regen-tool`: changing the tool's sources rebuilds gguf2hpp, reruns generation, then
  recompiles `weights.cu` and `main.cpp`.
- `build-main-only`: touching `apps/engine/main.cpp` rebuilds the engine without rerunning
  generation or recompiling `weights.cu`.
- `build-weights-only`: touching `src/weights/weights.cu` recompiles it and relinks the
  engine without recompiling `main.cpp`.
- `build-gpu`: configuring prints `MK_GPU: Rtx4070TiSuper, CUDA architectures: 89`, and
  every `.cu` file compiles for that architecture alone. `-DMK_GPU=Rtx5070` prints `120`.
  An unknown `MK_GPU` stops the configure with `MK_GPU=<name> is not a GPU this build
  knows`. A `CMAKE_CUDA_ARCHITECTURES` other than the GPU type's fails to compile with
  `nvcc compiles device code for another CUDA architecture than MK_GPU's cuda_arch`.

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
  `Generating model.hpp from .../toy-copy.gguf`, and a stdout whose first line is
  `toy: 2 layers, 2 tensors`. For Qwen, pass the model path instead and expect the first
  line `qwen3: 28 layers, 311 tensors`.
- **No change.** Run `cmake --build $S/engine-build --target engine`. Expect no
  `Generating`, `Building` or `Linking` lines.
- **Model changed.** Run `touch $S/toy-copy.gguf`, then the same build. Expect
  `Generating model.hpp`, then `Building ... weights.cu.o`, `Linking ... libweights.a`,
  `Building ... main.cpp.o` and `Linking ... engine`. Without a `Building` line, that
  file's dependency on the header is broken.
- **Tool changed.** Run `touch src/gguf/emitter.cpp`, then the same build. Expect
  `Building ... emitter.cpp.o`, `Linking ... gguf2hpp`, `Generating model.hpp`, then the
  same four lines as for a changed model.
- **Main changed.** Run `touch apps/engine/main.cpp`, then the same build. Expect
  `Building ... main.cpp.o` and `Linking ... engine`, with no `Generating` line and no
  `weights.cu.o` line.
- **Weights changed.** Run `touch src/weights/weights.cu`, then the same build. Expect
  `Building ... weights.cu.o`, `Linking ... libweights.a` and `Linking ... engine`, with no
  `Generating` line and no `main.cpp.o` line.
- **GPU.** After `engine`, run
  `grep MK_GPU build/verify-evidence/<id>/engine.configure.log` and
  `grep -o -- '--generate-code=[^ "]*' $S/engine-build/compile_commands.json | sort | uniq -c`.
  Expect the 89 line and one `compute_89,code=[compute_89,sm_89]` entry per `.cu` file.
  Configure a second directory by hand with `-DCMAKE_CUDA_ARCHITECTURES=120`, or with
  `-DMK_GPU=Rtx9999`, and expect the messages above.
- **Proof.** `engine.build.log`, `engine.stdout`, and the build lines from each touch step.

## Gotchas

- `touch` on `apps/engine/main.cpp`, `src/weights/weights.cu` or `src/gguf/emitter.cpp`
  changes only the file's mtime, but the user's own `build/` will rebuild those files on its
  next build. Say so in the report.
- The engine uploads the model to the GPU, so running it needs a CUDA device. Without one
  it prints the first line and the `file:` line, then exits 1 with the failed CUDA call.
- The `engine` target exists only when `MODEL_GGUF` is set. Configuring without it prints a
  CMake warning, and must still build every test target and pass ctest.
- A file without `blk.<n>.*` tensors (toy) has no `model::layers`, and the engine must
  still build for it. `src/weights/weights.hpp` gives the name an empty table in that case.
  Engine code that names `model::layers` must include it and must read a layer only when
  `mk::layer_count > 0`, at compile time.
- Pass an absolute `MODEL_GGUF`. `engine` resolves it with `realpath`; by hand, use `$PWD/...`.
