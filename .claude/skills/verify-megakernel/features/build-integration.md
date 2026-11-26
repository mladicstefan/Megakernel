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
- `build-arch`: configuring prints `MODEL_GGUF architecture: '<name>'`, the file's
  `general.architecture` exactly, wherever the key is in the file. The project's reader
  reads it: CMake compiles `gguf_architecture` from the reader's sources at the first
  configure (about 2.5 s, and again when they change) and runs it on every configure. A
  name in the table in `CMakeLists.txt` (`qwen3`) gets its model type and the decoders. Any
  other name, or none, prints `No forward pass for this architecture: only the engine is
  built`. A file the reader rejects stops the configure with one `MODEL_GGUF: <path>: <what>
  at byte <n>`, as does a `general.architecture` that is not a string or holds a NUL byte.
- `build-gpu`: the default is `MK_GPU=detect`. `gpu_probe` (`apps/gpu_probe/main.cu`,
  compiled once by `try_compile`, about 2 s, and run at every configure, about 0.25 s) asks
  the CUDA runtime for device 0 and writes `build/generated/build_gpu.hpp`. Configuring on
  the development machine prints `GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 66 SMs, sm_89,
  15988 MiB, the catalogue's type` and `MK_GPU: detect, selects Rtx4070TiSuper, CUDA
  architectures: 89`, and every `.cu` file compiles for that architecture alone. The header
  then holds `static_assert`s of the catalogue type's numbers against the driver's.
  `-DMK_GPU=detected` selects `DetectedGpu`, a type the header defines from the driver's
  numbers (`blocks_per_sm` unmeasured), and builds the same megakernel instructions.
  `-DMK_GPU=Rtx5070` asks no device and prints `120`. An unknown `MK_GPU`, or `detect`
  with no usable device (`CUDA_VISIBLE_DEVICES=`), stops the configure with one message
  that names `-DMK_GPU=<type>` and lists the catalogue. `CMAKE_CUDA_ARCHITECTURES` is
  ignored; a `-arch` in `CMAKE_CUDA_FLAGS` fails to compile with `nvcc compiles device code
  for another CUDA architecture than MK_GPU's cuda_arch`.

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
- **Architecture.** `engine.configure.log` of the toy run has `MODEL_GGUF architecture:
  'toy'` and `No forward pass for this architecture`. In a run of its own, `RUN_ID=<id2> $V
  craft qwen3 m 512 1536 4 2 128 2 1024 arch-last` writes a qwen3 file whose
  `general.architecture` is its last key; `RUN_ID=<id2> $V engine $S2/craft/m.gguf` prints
  `'qwen3'` and `cmake --build $S2/engine-build --target forward` builds. Configure a
  directory by hand with `-DMODEL_GGUF` set to a truncated copy of toy (`head -c 100`):
  expect exit 1 and one `CMake Error` block ending in `truncated at byte 98`. Measured on
  2026-11-26, with files made by `craft` and by hand: a random file gives `not a GGUF file
  (bad magic) at byte 0`, a missing one `cannot open`, a `uint32` architecture `is not a
  string`, one with a NUL byte `holds a NUL byte`, and `qwen3;x` with a newline configures
  with no forward pass.
- **GPU.** After `engine`, run
  `grep MK_GPU build/verify-evidence/<id>/engine.configure.log` and
  `grep -o -- '--generate-code=[^ "]*' $S/engine-build/compile_commands.json | sort | uniq -c`.
  Expect the 89 line and one `compute_89,code=[compute_89,sm_89]` entry per `.cu` file.
  Configure a second directory by hand with `-DMK_GPU=Rtx5070` (expect 120 and
  `compute_120`), with `-DMK_GPU=Rtx9999`, and with `CUDA_VISIBLE_DEVICES=` in a fresh
  directory, and expect the messages above. In a scratch copy, set `Rtx4070TiSuper::sms` in
  `src/config/gpu_catalogue.hpp` to 64: the configure passes and the build stops with
  `gpu_catalogue.hpp's Rtx4070TiSuper::sms is not the 66 this machine's GPU reports`.
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
- The architecture is read when CMake configures, not when the GGUF changes. Replacing the
  file with one of another architecture needs a reconfigure; until then `src/arch/qwen3.hpp`
  fails a `static_assert` on `general.architecture` rather than building the wrong decoder.
