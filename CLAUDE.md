# comptime-megakernel

An LLM inference engine compiled for one specific model. A build-time tool, `gguf2hpp`,
reads the front of a GGUF file and writes `model.hpp`: constexpr facts about the model
(tensor names, shapes, dtypes, offsets, hyperparameters, a hash). The engine compiles
against that header, so the model is known at compile time. The engine's goal is a single
persistent CUDA kernel that runs a whole decode step, with its schedule computed at compile
time.

First target: decode only, batch size 1, Qwen3-0.6B BF16 (`models/Qwen_Qwen3-0.6B-bf16.gguf`),
on an RTX 4070 Ti SUPER (66 SMs, sm_89). Correctness reference: llama.cpp.

Status and next steps: `ROADMAP.md` (milestones) and `docs/PLAN.md` (engine phases).

## Layout

Components live in directories under `src/`, executables under `apps/`, and tests mirror
`src/`. `src/` is the one include root: write every project include relative to it
(`#include "weights/weights.hpp"`). One top-level `CMakeLists.txt` builds everything.

- `src/base/`: `host_device.hpp`, `numeric.hpp` (BF16), `cuda_try.hpp` (`MK_TRY`).
- `src/gguf/`: `reader.{hpp,cpp}` (Cursor, MappedFile, `parse()`), `emitter.{hpp,cpp}`
  (`emit()`).
- `src/model/`: `model_types.hpp` (the types generated headers use).
- `src/config/`: `config.hpp` (the configuration types every kernel is a template over),
  `gpu.hpp` (GPUs as types: who each is, what it holds, the megakernel's measured blocks
  per SM), `kv_format.hpp` (KV cache formats as types), `row_kernel.hpp` (row kernels as
  types).
- `src/weights/`: `weights.{hpp,cu}` (the weights on the GPU), `verified_file.hpp`,
  `verified_device.hpp` (the device check `Megakernel::make` requires), `checksum.hpp`.
- `src/megakernel/`: `megakernel.{hpp,cu}` (one kernel launch per decode step, qwen3 only),
  `schedule.hpp` (the row split, the phase list and every proof the kernel asserts, as
  constexpr functions that need no model and no GPU).
- `src/reference/`: `forward.{hpp,cu}` (the reference forward pass the megakernel is
  checked against), `compare.hpp`.
- `src/decode/`: `greedy.hpp` (token IDs in, token IDs out), `positions.hpp`.
- `apps/engine/main.cpp`: the engine. `apps/gguf2hpp/main.cpp`: the `gguf2hpp` CLI.
  `apps/forward/main.cpp`: the `forward` CLI, which decodes token IDs with the megakernel
  or the reference pass, checks the first against the second, and the second against a
  llama.cpp dump.
- `tools/reference/`: the llama.cpp harness that makes the dumps (its own CMake project).
- `tests/`: `gguf/reader_test.cpp` and `gguf/emitter_test.cpp` (one gtest binary,
  `reader_test`); `weights/checksum_test.cpp`, `base/numeric_test.cpp`,
  `reference/compare_test.cpp`, `decode/positions_test.cpp` and
  `megakernel/schedule_test.cpp` (one binary, `host_test`);
  `weights/weights_test.cpp` (needs `MODEL_GGUF` and a GPU); `smoke_test.cpp`.
- `tests/data/`: `toy.gguf` (384 bytes, the main fixture), `make_test_gguf.py`, and a uv
  venv with gguf-py (`tests/data/.venv/bin/python -I`).
- `models/`: real models, gitignored, may be missing. Tests that need one skip.
- `build/generated/`: headers gguf2hpp writes (`toy.hpp`, `qwen.hpp` if the model exists,
  `model.hpp` for the engine).
- `.claude/skills/verify-megakernel/`: the project's verification skill.

## Build and test

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
```

- **Engine:** the target exists only when configured for a model:
  `-DMODEL_GGUF=$PWD/models/Qwen_Qwen3-0.6B-bf16.gguf`. Use a separate build directory
  for that.
- **GPU:** `-DMK_GPU=<type>` names the GPU, a type in `src/config/gpu.hpp` (default
  `Rtx4070TiSuper`; `Rtx5070` compiles but has never run). CMake's table next to it gives
  the CUDA architecture, and the build compiles that one only. Another Nvidia GPU is a type
  in `gpu.hpp`, a line in that table and a recompile.
- **Sanitizers:** use a separate build directory with `-DENABLE_SANITIZERS=ON` (ASan and
  UBSan, host code only; use `compute-sanitizer` for kernels).
- **Offline configure of a new build directory:** reuse the fetched sources with
  `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$PWD/build/_deps/googletest-src` and
  `-DFETCHCONTENT_SOURCE_DIR_XXHASH=$PWD/build/_deps/xxhash-src`.
- **Real files and the real build:** use the **verify-megakernel** skill. It runs gguf2hpp
  on toy, Qwen and hostile GGUFs, compiles the generated headers with g++ and nvcc, and
  builds the engine. When a change alters user-facing behaviour, update its
  `features/*.md` in the same change.

## Language and style

- Host code is C++23 with GCC. **Anything nvcc compiles or includes is C++20:**
  `model_types.hpp`, generated headers, and all device code. A build compiles device code
  for one CUDA architecture, its GPU type's `cuda_arch`, and a `static_assert` on
  `__CUDA_ARCH__` in `gpu.hpp` rejects any other.
- Warnings are strict (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  -Wold-style-cast` and more). Keep the build warning-free.
- Compile time only. A choice that the model or the build fixes is a type or a constant,
  never a run-time value or a flag. Add it to a configuration type in
  `src/config/config.hpp`, give its interface a concept, and prove what can be proven with a
  `static_assert`.
- Structs with public fields and member functions. No virtual, no inheritance for
  polymorphism, no exceptions (use `std::expected`), no `std::function` or `shared_ptr`.
  Templates and concepts where they earn it.
- Use the standard library when it is just as fast (for example, `unique_ptr` with a
  deleter rather than hand-written move operations). No abstraction nobody asked for.
- The GGUF file is untrusted:
  - Validate at the boundary.
  - Compare a length against the bytes remaining; never compute `pos + len`.
  - Use `memcpy` for unaligned reads.
  - Check an id before casting it to an enum.
- Comments say why, in one line. A known, accepted limitation is marked
  `// shortcut: <the limit>, <when to upgrade>`.
- Match the surrounding code's naming, comment density and idiom. Don't refactor code
  outside the task.

## Verification

- A change is done when the build is warning-free, ctest passes, the sanitizer build
  passes, and the behaviour is proven on the real artifact (the verify skill), not just
  "it compiles".
- At the end of a milestone, report each measurement next to what was expected, and
  explain any difference. Label numbers as measured or estimated.
- New non-trivial logic gets a test that fails when that logic breaks.

## Working agentically

Agents write the code; the user directs, reviews pull requests and merges them.

- **Subagents:** any that write, edit or review code run on `model: "opus"`. Any that
  only research or read run on `model: "sonnet"`.
- **One feature, one branch, one pull request.**
  - A feature is the smallest change that stands on its own: it builds, its tests pass,
    and it can be explained in one paragraph. A refactor, a new kernel, a default change
    and a doc update are four features, not one.
  - Branch from `master`. When a feature builds on a pull request that is still open,
    branch from that pull request's branch and use it as the base. This is a stack.
  - Keep one commit per pull request. Fold review fixes into that commit, then restack
    the branches above it.
  - The commit message is the pull request's description. Title: `type(scope): subject`.
    Body, with these headings and under 40 lines: `## Why`, `## What changed`,
    `## Verification`. Add `## Tradeoffs` or `## Blast radius` only when a reviewer would
    ask.
  - Work in a git worktree, never in the main checkout.
  - Push the feature branch and open the pull request with `gh pr create --base <base>`.
    A session may keep building the next feature on top without waiting for review.
  - Never push to `master` and never merge. The user does both.
  - While `gh` is not installed there are no pull requests. Then a session pushes each
    feature branch and merges it into its local `master` with `git merge --no-ff`, in
    stack order, so that every feature stays one merge commit. The user still pushes
    `master`.
- **Two sessions never edit the same files in the same checkout.** Parallel work goes in
  separate worktrees.
- At the end of a milestone, update `ROADMAP.md`. For phase-level changes, update
  `docs/PLAN.md`.
- **Reports:** top down (a one-paragraph overview, then the parts, then the details). Say
  what you skipped or couldn't check. Don't quiz the user. When a session leaves work
  behind, say which branch or files hold it.
