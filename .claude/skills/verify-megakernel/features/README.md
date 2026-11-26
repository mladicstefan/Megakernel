# Megakernel verification map

This directory is the maintained source for verifying the user-facing behavior of gguf2hpp
and the engine build. Read this index before driving, then use the matching feature file as
the recipe.

## Baseline preconditions

- `V=.claude/skills/verify-megakernel/scripts/verify.sh`, run from the repo root.
- A fresh `RUN_ID` for the run, the same one on every call.
- `RUN_ID=<id> $V launch`, then `RUN_ID=<id> $V doctor` exits 0.
- Fixtures: `tests/data/toy.gguf` (384 bytes, 2 tensors, 6 keys) and, if present,
  `models/Qwen_Qwen3-0.6B-bf16.gguf` (1.5 GB, 311 tensors, 27 keys, 28 layers).
- Never drive a build tree that this run did not create.

## Driving conventions

- Start every recipe from the baseline state.
- Run the tool through `gen` and headers through `compile`, so the evidence lands in the
  same place every time.
- Treat commands as literal.
- Put crafted and copied inputs in the run's scratch directory, never in `tests/data/`.

## Proof and skip reporting

- Tool proof: `<name>.cmd`, `<name>.stderr`, `<name>.exit` and the output file (or its
  absence).
- Header proof: `<name>.gcc.log` and `<name>.nvcc.log`, both empty with exit 0, plus
  `<name>.probe.stdout`.
- Build proof: the `cmake --build` lines that show what reran.
- If the Qwen model or nvcc is missing, report that entry point as skipped, with the
  doctor line. Don't count a toy-only run as Qwen coverage.

## Feature entry contract

Each feature file has an H1 title and one paragraph of user-visible behavior, then exactly
these H2s, in order: `Sub-features`, `How to get to it (user POV)`,
`Driving it with verify.sh`, `Gotchas`.

## Features

- [Generate a header](./generate-header.md): toy and Qwen headers, byte-identical reruns,
  and compiling on g++ and nvcc.
- [Error reporting](./error-reporting.md): usage, unreadable and malformed inputs, an
  unwritable output, exit codes, no partial files.
- [Untrusted input](./untrusted-input.md): hostile strings are escaped byte-exactly;
  invalid keys and non-finite values are rejected.
- [Build integration](./build-integration.md): `MODEL_GGUF` generates `model.hpp`, the
  engine builds, and edits rerun the right steps.
- [Reference dumps](./reference-dumps.md): llama.cpp's per-layer tensors and logits for a
  committed prompt, reproducible checksums, and greedy agreement with `llama-simple`.
- [Weights upload](./weights-upload.md): the engine verifies its GGUF, uploads the tensor
  data to the GPU, proves the copy with per-tensor checksums, and rejects any other file.
- [Forward pass](./forward-pass.md): the reference CUDA forward pass decodes token IDs,
  matches llama.cpp's greedy tokens, and is compared with its dump tensor by tensor.
- [Megakernel](./megakernel.md): one kernel launch per decode step, checked phase by phase
  against the reference pass, with its tokens per second.
- [Perf gate](./perf-gate.md): the megakernel's registers, spills, shared memory, SASS
  count and ms per step, with the baseline and the limits a change stays inside.
