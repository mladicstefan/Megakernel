---
name: verify-megakernel
description: Drive comptime-megakernel's build-time tool gguf2hpp (GGUF file in, C++ model header out) and the CMake engine build the way a user does, and capture proof. Use to prove a change to the reader, emitter, model_types.hpp, CMakeLists.txt or src/main.cpp works on real files, hostile files and the real build, beyond what ctest covers.
---

# Verify megakernel

The user-facing surface is a CLI, `gguf2hpp <in.gguf> <out.hpp>`, plus the CMake path that
runs it: configuring with `-DMODEL_GGUF=<file>` generates `model.hpp` and compiles the
`engine` against it. The generated header is itself a product: it must compile as C++23
(host) and as CUDA C++20 (nvcc). ctest covers the library; this skill covers the binary, its
files and the build.

All commands go through `scripts/verify.sh` in this skill's folder. Shell state does not
persist between calls, so prefix every call with the same `RUN_ID`:

```bash
V=.claude/skills/verify-megakernel/scripts/verify.sh   # from the repo root
RUN_ID=20261125-1200 $V launch
```

Paths per run:
- Scratch (builds, crafted files, probe programs): `${TMPDIR:-/tmp}/megakernel-verify/$RUN_ID`.
  Removed by cleanup.
- Evidence: `build/verify-evidence/$RUN_ID/` in the repo (gitignored). Kept after cleanup.

## Launch

`RUN_ID=<id> $V launch` configures a fresh build in scratch and builds only `gguf2hpp`
(about 7 s; googletest is configured, not built). It reuses `build/_deps/*-src` when an
earlier configure of `build/` fetched them, so it works offline. Log: `launch.log`.
Ready means `doctor` passes. There is no server: every drive is one process run.

## Doctor

`RUN_ID=<id> $V doctor` is read-only. It checks:
- the binary exists
- it was built from the current tree (HEAD plus uncommitted diff; untracked files are not
  counted)
- a no-argument run prints `usage:` and exits 2
- `tests/data/toy.gguf` is 384 bytes
- the Qwen model, g++, nvcc and the gguf-py venv (`tests/data/.venv`) are present
It exits non-zero on a hard failure. Rerun `launch` after any source change.

## Drive

| Command | What it does |
|---|---|
| `gen <in.gguf> <name>` | Runs the tool with output `<evidence>/<name>.hpp`. Records `<name>.cmd`, `.stderr` and `.exit`. Prints `exit=`, the output size, and `tmp-left=` (a stray `.hpp.tmp` is a bug). |
| `compile <header> <name> [check.hpp]` | Writes a probe program that includes the header (and the check file), uses `mk::find` on the first tensor, compiles it with `g++ -std=c++23 -Wall -Wextra -Wpedantic -Werror` and with `nvcc -std=c++20 -c`, then runs the g++ build. Evidence: `<name>.gcc.log`, `.nvcc.log`, `.probe.stdout`. |
| `craft <kind> <name>` | Writes `<scratch>/craft/<name>.gguf` and prints its path. Kinds: `evil-strings` (also writes `<name>.gguf.check.hpp` with byte-exact `static_assert`s), `bad-key`, `digit-key`, `nonfinite`, `zero-tensors`. |
| `engine <in.gguf>` | Configures `<scratch>/engine-build` with `MODEL_GGUF`, builds `engine`, runs it. Evidence: `engine.configure.log`, `engine.build.log`, `engine.stdout`. Prints the `Generating model.hpp` line. |
| `run-engine <name> [args]` | Runs the engine that `engine` built, with the given arguments (a GGUF path, or none). Records `<name>.cmd`, `.stdout`, `.stderr` and `.exit`. Prints `exit=` and both streams. |
| `gate` | For the engine that `engine` built: ptxas's registers, spills and static shared memory for the megakernel on sm_89, its SASS instruction count, and the median and range of ms per step over three `forward --megakernel --generate 128` runs. Evidence: `gate.txt`, `gate.ptxas.log`, `gate.sm_89.sass`, `gate.run{1,2,3}.stdout`. |

The recipes per feature are in [features/README.md](features/README.md).

## Evidence

A proof has:
- the command
- stderr
- the exit code
- the produced file
- for headers, that the header compiles on both compilers, and what the probe printed

Test the real binary on real files. Never call `parse()` or `emit()` directly here; ctest
does that. For a mutation of the build (touching a file), show the build lines that prove
what reran. Name the feature ID and the `RUN_ID` with each claim.

## Cleanup

`RUN_ID=<id> $V cleanup` removes only that run's scratch directory and never kills a process
by name. Evidence stays in `build/verify-evidence/$RUN_ID/`. After a failed attempt, run
cleanup too, so stale builds don't mislead the next doctor.

## Isolation

Runs with different `RUN_ID`s share nothing but the repo's sources. Don't point `gen` at the
same output name twice in one run unless you mean to overwrite it. gguf2hpp writes
`<out>.tmp` and renames it over `<out>`. Never drive the user's own `build/` tree.
