# Roadmap

Status as of 2026-11-25. The header tool, the weights loader and the reference forward pass
are done. The engine cannot generate text yet. Nothing below "Done" is started.

## The rule: compile time only

The engine is compiled for one model, one GPU and one configuration. Anything the model or
the build fixes is a type or a constant, never a run-time value:

- tensor offsets, shapes and dtypes (from `model.hpp`)
- the layer count and every dimension
- the KV cache format (`mk::Kv`, a type that satisfies `mk::KvFormat`)
- the GPU (SM count, architecture) and the context length
- the schedule: which SM runs which piece of work, in which order

Run-time values are what the user supplies: the file's bytes, the token IDs and the
position. A new choice is added as a type or a template parameter, not as a flag or a
branch.

## Done

| What | Where | Proof |
|---|---|---|
| `gguf2hpp`: GGUF to a constexpr `model.hpp` (milestones 1 to 8) | `src/gguf/`, `src/model/model_types.hpp` | header compiles on g++ (C++23) and nvcc (C++20); hostile files rejected |
| llama.cpp reference dumps (Phase A) | `tools/reference/` | byte-identical reruns; greedy tokens equal `llama-simple` |
| Weights on the GPU (Phase B) | `src/weights/weights.{hpp,cu}`, `src/weights/verified_file.hpp` | 311 of 311 device checksums match the file; a wrong file is rejected |
| Reference forward pass (Phase C) | `src/reference/forward.{hpp,cu}`, `apps/forward/` | 32 greedy tokens equal llama.cpp's; every op within 1e-4 of its dump |
| KV cache format as a type, BF16 for the engine | `src/config/kv_format.hpp`, `EngineConfig` | F16, BF16 and F32 all generate llama.cpp's 32 tokens |

Measured on the RTX 4070 Ti SUPER (details in `docs/PLAN.md`):

| Measure | Value |
|---|---|
| upload of 1.5 GB, warm page cache | 0.167 s, 9.0 GB/s |
| memory read rate of a simple kernel | 623 GB/s, 93% of nominal |
| llama.cpp `b12200`, CUDA backend | 381 tokens/s |
| reference forward pass | 31 tokens/s (it is a test oracle) |
| ceiling for one decode step | about 520 tokens/s (estimated: 1.19 GB at 623 GB/s) |

What is still decided at run time, and so breaks the rule today:

- The reference pass passes each matrix's address, rows and columns to its kernel as
  arguments, and walks the layers in a host loop.

## Targets

Each target ends in something a user can run, with its speed and correctness measured. There
are no intermediate phases: the design, the code and the measurements of a target land
together.

### 1. The megakernel decodes (replaces phases D, E and F)

One persistent CUDA kernel runs a whole decode step, with its schedule computed at compile
time.

**State on 2026-11-25:** a first megakernel exists (`src/megakernel/megakernel.{hpp,cu}`).
One launch runs a step as 170 phases with grid-wide barriers between them, on a grid of 132
blocks (2 per SM). Every phase matches the
reference pass (within 6e-6, and 1.3e-3 for attention through the BF16 cache), and the
three sanitizers are clean. Measured with `forward --megakernel` on the `planets` prompt:

| Positions decoded | ms per step | tokens/s | llama.cpp CUDA |
|---|---|---|---|
| 141 | 2.31 | 433 | 381 (`tg128`) |
| 1,037 | 2.40 | 416 | 369 (`tg1024`) |
| 4,013 | 2.69 | 372 | not measured |

It is past llama.cpp and short of the goal of 450. At the measured read rate one step's
weights take 1.91 ms. Where the 2.27 ms of a step over 141 positions go, measured by
building the kernel with parts switched off (scratch copies, three runs each):

| Part | Time | At the read rate it would take |
|---|---|---|
| launch, token read-back and all 170 barriers | 0.14 ms | 0 |
| the output matrix (311 MB) | 0.51 ms | 0.50 ms |
| the layers' matrices (881 MB) | 1.57 ms | 1.41 ms |
| attention, at about 70 positions | 0.08 ms | 0.01 ms |

So the output matrix is at the limit, the layers are 11% above it, and 0.22 ms is not
weight reading at all. The row kernel is not what limits the layers: kernels that read 8,
16 or 32 bytes per lane all give 2.42 to 2.47 ms on the old grid.

What is a type or a checked constant now: the configuration (`MegakernelConfig`: KV
format, context, GPU, grid, row kernel), the phases of a layer (`LayerPhases`, checked by
`well_ordered`), and the row schedule (`rows_of`, checked by `partitions`).
`engine --tokens <tokens.txt>` decodes through it. Still open from the list below: the
watchdog test.

- **Configuration is a type.** One `struct` names the build's choices: the KV format, the
  GPU, the context length. The kernel is a template over it. `model.hpp` supplies the rest.
- **Ops are templates over what they read.** A matrix-vector product takes its tensor as a
  compile-time value, so its address, rows and columns are constants in the kernel. The
  reference pass moves to the same templates.
- **The schedule is a constexpr value**, checked by `static_assert`: no cycle, no two
  writers of one buffer.
- **`engine --tokens <tokens.txt>` prints generated IDs**, through the megakernel.

Done when, on the 4070 Ti SUPER and the Qwen3-0.6B BF16 file:

- every tensor matches the reference pass (exact arithmetic) within the reduction-order
  tolerance, and the 32 greedy tokens of `planets` equal llama.cpp's
- `compute-sanitizer` memcheck, racecheck and synccheck report 0 errors, and a watchdog
  test shows no hang over 10,000 steps
- measured tokens/s at context 1, 1,024 and 4,096 are reported next to the ceiling, with
  the limiter named. The bar is llama.cpp's 381 tokens/s. The goal is 450 or more.

### 2. Text in, text out

- the tokenizer and the chat template, read from the GGUF arrays that `model.hpp` already
  locates
- sampling (greedy, temperature, top-k, top-p) with a seed
- a prefill path for the prompt, so a long prompt does not pay one decode step per token
- `engine "prompt"` streams text

Done when the greedy text for a set of prompts equals llama.cpp's, time to first token and
tokens/s are measured, and a prompt of the full context length runs without error.

### 3. A second GPU and a second model, by recompiling

- the RTX 5070 (48 SMs, sm_120) as a second GPU type
- a second Qwen3 size, and one quantized format

Done when each needs only a different `model.hpp` and configuration type, with no edit to
kernel code, and each has its own measured table.

## Open items (accepted for now)

- `BadTensorExtent` reports `data_start`, not which tensor failed.
- `OpenFailed` does not keep `errno`.
- The toy hash test value was recorded from this code, not an outside tool.
- Key segments that are C++ keywords, and leaf/prefix key clashes, are not rejected by
  the emitter; the generated header fails to compile instead.
- Device code indexing `model::layers` with a run-time index needs a copy through a
  `static constexpr` local. A compile-time index has no such problem.
- Only the GGUF header is hashed. A file with the same header and other tensor data
  passes the engine's check.
- The checksum reads whole 8-byte words: a tensor that is not a whole number of them
  fails a `static_assert`.
- The reference pass is checked on one prompt.
- CMake finds the architecture by reading the first 4096 bytes of the GGUF.
