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

- `tools/gguf_header/`: `reader.{hpp,cpp}` (Cursor, MappedFile, `parse()`),
  `emitter.{hpp,cpp}` (`emit()`), `main.cpp` (the `gguf2hpp` CLI).
- `src/`: the engine (`main.cpp`) and `model_types.hpp` (the types generated headers use).
- `tests/`: `reader_test.cpp` and `emitter_test.cpp` (one gtest binary, `reader_test`),
  `smoke_test.cpp`.
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
  `model_types.hpp`, generated headers, and all device code. CUDA architectures are 89
  and 120.
- Warnings are strict (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  -Wold-style-cast` and more). Keep the build warning-free.
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

Agents write the code; the user directs, reviews and commits.

- **Subagents:** any that write, edit or review code run on `model: "opus"`. Any that
  only research or read run on `model: "sonnet"`.
- **Commits:**
  - In the main checkout, commit only when the user asks.
  - A session in its own git worktree may commit on its own branch.
  - Never merge into `master`, and never push. The user does both. The sandbox can't
    reach the SSH key anyway.
- **Two sessions never edit the same files in the same checkout.** Parallel work goes in
  separate worktrees.
- At the end of a milestone, update `ROADMAP.md`. For phase-level changes, update
  `docs/PLAN.md`.
- **Reports:** top down (a one-paragraph overview, then the parts, then the details). Say
  what you skipped or couldn't check. Don't quiz the user. When a session leaves work
  behind, say which branch or files hold it.
