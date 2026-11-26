# comptime-megakernel

An LLM inference engine compiled for one model and one GPU. Everything the model or the
build fixes becomes a constant or a type before the program runs: tensor offsets, shapes,
the KV cache format, the GPU's SM count, and the order in which the GPU does its work.

First target: decode at batch size 1 of Qwen3-0.6B (BF16) on an RTX 4070 Ti SUPER, checked
against llama.cpp.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/speed-dark.svg">
  <img src="docs/img/speed.svg" alt="Tokens per second: this engine 436 and llama.cpp 381 at 141 positions; 416 and 369 at 1,037 positions; goal 450; estimated ceiling 520.">
</picture>

| Positions decoded | This engine | llama.cpp, CUDA backend |
|---|---|---|
| 141 | 433 to 440 tokens/s | 381 tokens/s |
| 1,037 | 416 tokens/s | 369 tokens/s |
| 4,013 | 372 tokens/s | not measured |

All numbers are measured on the RTX 4070 Ti SUPER. The ceiling is an estimate: one step
reads 1.19 GB of weights, and a simple kernel on this GPU reads 623 GB/s.

In every diagram below, teal marks what is fixed at compile time and orange marks what
arrives at run time.

## Build and run

```bash
cmake -S . -B build -DMODEL_GGUF=$PWD/models/Qwen_Qwen3-0.6B-bf16.gguf
cmake --build build -j
ctest --test-dir build

# Token IDs in, token IDs out.
build/engine --tokens tests/data/reference/planets/tokens.txt --generate 32
```

The engine takes and prints token IDs. It has no tokenizer yet.

## How it works

### 1. The model becomes a header

A build tool, `gguf2hpp`, reads the front of the GGUF file and writes `model.hpp`. That
header holds every tensor's name, shape, dtype and byte offset as `constexpr` tables. The
engine is compiled against it, so a tensor's address is a constant in the kernel.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/pipeline-dark.svg">
  <img src="docs/img/pipeline.svg" alt="The GGUF header is turned into model.hpp, which the compilers combine with config.hpp into one engine binary.">
</picture>

Both inputs to the compiler are fixed before the engine exists. Only token IDs arrive
later.

### 2. Start-up refuses anything it was not built for

The engine maps the file and compares its size and header hash with the constants in
`model.hpp`. Then it compares the GPU's SM count with the GPU type in the configuration.
Only after both checks does it copy the tensor data into one device allocation.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/startup-dark.svg">
  <img src="docs/img/startup.svg" alt="Start-up checks the file, then the GPU, then uploads; a failed check exits before anything reaches the GPU.">
</picture>

A wrong file or a wrong GPU stops the engine before any weight is uploaded. Without
`--tokens` the engine proves the upload with a checksum per tensor.

### 3. The configuration is a type

One struct names every build choice. Each kernel is a template over that struct, so a
choice is a constant inside the kernel and never a flag. Three of the members are types
themselves, and each has a small interface that the compiler checks.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/config-dark.svg">
  <img src="docs/img/config.svg" alt="EngineConfig has six members; three are types checked by a concept, and the grid is derived from the GPU's SM count.">
</picture>

To swap a component, name another type in the struct. A grid that cannot run the model
fails a `static_assert` when the kernel is compiled.

### 4. The megakernel runs a whole step in one launch

A kernel launch costs more than reading a small matrix, so one launch per operation can
never be fast. The megakernel is launched once per token. All 132 blocks walk the same
list of phases, and a barrier after each phase makes every block wait until all of them
have finished it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/megakernel-dark.svg">
  <img src="docs/img/megakernel.svg" alt="One launch runs embed, 28 layers and the logits; a layer is six phases that pass buffers to each other, with a barrier after each.">
</picture>

A thick bar is a barrier, where all 132 blocks arrive before any continues. The phase
order is a type list. A `static_assert` rejects any order in which a phase reads a buffer
that no earlier phase wrote.

### 5. A matrix is split by a proven schedule

One `constexpr` function gives each of the grid's 1,056 warps a contiguous run of a
matrix's rows. A compile-time check walks every warp and proves that each row has exactly
one. Inside a warp, each of the 32 lanes reads 16 bytes, so the warp reads 512 contiguous
bytes per step.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/row-schedule-dark.svg">
  <img src="docs/img/row-schedule.svg" alt="Rows of a BF16 matrix are divided among warps; a warp reads one row as 32 lanes of 16 bytes.">
</picture>

The split depends only on the row count and the warp count, and both are compile-time
constants.

### 6. Each implementation is checked against the one before it

llama.cpp is the outside reference. A slow reference pass, with one kernel per operation,
is compared with a llama.cpp dump of every tensor. The megakernel is compared with the
reference pass, phase by phase, on the same inputs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/checks-dark.svg">
  <img src="docs/img/checks.svg" alt="The reference pass is checked against llama.cpp and the megakernel against the reference pass; the engine uses the megakernel.">
</picture>

All three generate the same 32 tokens for the test prompt. The megakernel's attention
differs more because K and V are rounded to BF16 on their way into the cache. With a
32-bit cache the difference is 3.7e-6.

### 7. Where a step's time goes

One decode step over 141 positions takes 2.30 ms. The parts were measured by building the
kernel with pieces switched off.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/step-time-dark.svg">
  <img src="docs/img/step-time.svg" alt="A 2.30 ms step: 0.14 ms launch and barriers, 0.51 ms output matrix, 1.57 ms layers, 0.08 ms attention; at the memory read rate the same work would take 1.92 ms.">
</picture>

| Part | Measured | At the memory read rate |
|---|---|---|
| launch, token read-back and all 170 barriers | 0.14 ms | 0 |
| the output matrix (311 MB) | 0.51 ms | 0.50 ms |
| the 28 layers' matrices (881 MB) | 1.57 ms | 1.41 ms |
| attention, at about 70 positions | 0.08 ms | 0.01 ms |

The output matrix already runs at the memory limit. The layers are 11% above it.

## What is still open

| Item | State |
|---|---|
| Another model architecture | The forward pass is written for the qwen3 graph only. |
| Another GPU | A GPU is one type. A 48-SM type compiles. It has not been run. |
| KV cache below 16 bits | Would be a new format type. None exists. |
| One launch for many tokens | Still one launch per token. |
| Text in, text out | No tokenizer yet. |

## Where to read more

- `ROADMAP.md`: the targets and the measurements so far.
- `docs/PLAN.md`: how each finished phase was built and proven.
- `CLAUDE.md`: the layout of the repository, the build options and the working rules.
