# Architecture: one engine per model and GPU

This is the target shape of the engine and the review that led to it. Three adversaries
tried to break the claim "the engine compiles to one model and one GPU, and adding
another of either is cheap", and three designers each wrote a target architecture. The
findings below were reproduced by rerunning the adversaries' commands. The design is the
shape all three designers converged on.

## The seven paradigms

Every rule about structure in this codebase is one of these.

| # | Paradigm | What it is | Where it exists today |
|---|---|---|---|
| P1 | Description type | A struct of `static constexpr` members behind a concept of three requirements or fewer. Dispatch is static because the kernel is a template over it. | `KvFormat`, `GpuSpec`, `RowKernel` |
| P2 | Typed handle | The type carries what is the same for every instance (shape, dtype). The value carries what differs (offset). | nowhere |
| P3 | Policy by overload | A choice inside an algorithm is a tag type, and each implementation overloads on it. No flag, no branch. | nowhere: `llama_cpu_rounding` is a `bool` |
| P4 | Type list, folded | An ordered list of types run by a fold expression. The schedule is a type. | `LayerPhases`, `run_layer` |
| P5 | Proof as a constant | A `constexpr` function over plain numbers, asserted where it is used, and fed failing numbers by a host test. | `schedule.hpp`, tested by `schedule_test` |
| P6 | Proof-carrying value | A run-time fact is checked once, and the result is a type only that check can make. | `VerifiedFile`, `VerifiedDevice`, `Token`, `HostWeights` |
| P7 | One build type | One type names one of each choice, and one translation unit instantiates the templates for it. | `EngineConfig`, without the model and with the GPU fixed |

## What the review found

For Qwen3-0.6B on the RTX 4070 Ti SUPER the engine does what it claims: the documented
tokens, every check within tolerance, 101 of 101 tests. Swapping the KV format, the row
kernel or `blocks_per_sm` inside `EngineConfig` is one edit and stays correct.

The claim breaks as soon as the model or the GPU changes. Each row was reproduced.

| Finding | What happens | Missing paradigm |
|---|---|---|
| Attention is wrong when a KV head has more blocks than a block has dimensions | A model with head size 32 or 64 passes every `static_assert`, then gives wrong tokens from position 64 or 512 on. `attention_merge` loads one value per thread and runs out of threads. | P5: the split has no proof |
| The shared buffer is sized by the feed-forward width only | A model whose attention width exceeds it writes past the buffer: `illegal memory access`. | P2: shapes are loose constants |
| The Qwen3-4B shape cannot be built | `threads` must be 512, and then the kernel's static shared memory is 76 KB against a limit of 48 KB. No configuration fixes it. | P1: the GPU type has no capacity |
| `blocks_per_sm = 5`, a context of a million positions, a GPU type built for the wrong CUDA architecture | Each compiles and fails only at run time. | P1, P5 |
| A second configuration type | It satisfies the concept and then fails to link: the instantiations are named inside the kernel files. | P7 |
| The megakernel ignores `llama_cpu_rounding` | Its concept requires the field and its code never reads it. | P3 |
| The device check is written twice and names `EngineConfig` | A second configuration with another GPU skips it and waits 26 s on a barrier. | P6 |
| CMake reads the architecture by scanning for strings | A valid GGUF with another key order reads as `qwen3&`, and no decoder is built. | one owner per fact |
| `gguf2hpp` rejects a file for one key it cannot name | `general.base_model.0.name`, or an architecture with a hyphen, stops the whole header. | the tool is stricter than the engine needs |
| A file with tied output weights, or RoPE scaling | The first fails to compile without naming the tensor. The second is ignored. | P3, P5 |
| Tests are registered per architecture but expect 0.6B's numbers | A second qwen3 model fails three tests that are not about it. | fixtures keyed by model |
| `--generate` near 2^64, or too large to parse | Wraps to a short run, or silently means 32. Exit 0 both times. | boundary check |
| One assertion message for five conditions | `this grid cannot run this model` does not say which condition failed. | P5: one assert per fact |

None of these affects the shipped build. All of them are what "adding a model or a GPU"
would hit first.

## The target structure

Each fact gets one owner.

| Owner | Holds | Paradigm |
|---|---|---|
| `model.hpp` (generated) | The file's numbers: every tensor and every metadata value. | |
| `Matrix<S>`, `Vector<S>` | A tensor handle whose type is its shape and dtype and whose value is its offset. A kernel reads rows, columns and format from the type. The 28 layers share one instantiation. | P2 |
| `arch::qwen3` | What the numbers mean: a `DecoderShape` read from `model.hpp`, and two policy types, `QkNorm` and `Rope`. About 20 lines. | P1, P3 |
| `Rtx4070TiSuper`, `Rtx5070` | The hardware: SM count, CUDA architecture, memory, shared memory per SM, and the megakernel's `blocks_per_sm`, a measurement on that hardware. | P1 |
| `EngineConfig<Arch, Gpu>` | One of each, plus the KV format, the context length and the row kernel. `threads` is derived from the model, not chosen. | P7 |
| `schedule.hpp` | Every proof, as functions over plain numbers with one named assertion per fact: the row split, the attention split, the phase order, the barrier count, shared memory, device memory. | P5 |
| `VerifiedDevice<G>` | The device check, made once, as a type the megakernel requires. | P6 |

What is a template over what: `Megakernel<B>` and its kernel over a build. The reference
pass over a build and a rounding policy. `Matrix<S>` over a shape. Nothing is a template
over a tensor's offset or over the layer index.

What adding things then costs:

- **A second size of the same architecture:** another `MODEL_GGUF`. No code.
- **The RTX 5070:** a type in `gpu.hpp`, one line in CMake's table and
  `-DMK_GPU=Rtx5070`. The build compiles for that GPU's CUDA architecture only, and a
  `static_assert` on `__CUDA_ARCH__` rejects a mismatch.
- **A Llama-style decoder:** one architecture header and one overload per policy.
- **Another weight dtype:** one format type and one row kernel.

## The migration

One feature, one branch, one commit each, in this order. Every step passes the gate.

**The gate:** warning-free builds, the whole ctest, and for a step that touches the kernel:
the same register count (123), no spills, the same shared memory (23,008 bytes), a SASS
instruction count within 1% of 5,848, and a median of at most 2.33 ms per step over three
runs of `forward --megakernel --generate 128`.

| # | Step | Opens |
|---|---|---|
| 0 | **Done.** The gate as a script in the verify skill, with today's numbers: `verify.sh gate`. | every later "same speed" claim |
| 1 | **Done.** Fix the bugs that give wrong output: the attention merge loads every member's partial, the shared buffer is sized by the widest vector, the row kernel's alignment is asserted, and `--generate` is range-checked. The repros are the verify skill's "Other shapes" recipe and three tests. | correctness on other shapes |
| 2 | **Done.** The proofs live in `src/megakernel/schedule.hpp`, and `megakernel.cu` asserts each fact with its own message. `tests/megakernel/schedule_test.cpp` feeds every proof failing numbers. | P5 |
| 3 | **Done.** The GPU type carries its CUDA architecture, its capacity and the megakernel's measured `blocks_per_sm`. `MK_GPU` picks it, the build compiles one architecture, a `static_assert` on `__CUDA_ARCH__` rejects another, and `schedule.hpp` proves shared memory, device memory and the context length. `Rtx5070` is a compile-only target. | the hardware axis |
| 4 | **Done.** `VerifiedDevice<G>` replaces the two copies of the SM check. `Megakernel<C>::make` takes a `VerifiedDevice<C::Gpu>`, which only `VerifiedDevice<G>::check()` makes, so a caller that skips the check does not compile. The upload and the reference pass do not take one: neither depends on the SM count. | P6 |
| 5 | Typed tensor handles replace the offset tables and the dtype casts. | P2 |
| 6 | `arch::qwen3` and `EngineConfig<Arch, Gpu>`. The globals in `forward.hpp` go, `threads` is derived, and CMake parses the architecture from the GGUF's bytes exactly. | a second size of the same architecture |
| 7 | The megakernel gets its own library and its own sequence state. The engine no longer links the reference pass. The rounding flag becomes the reference pass's policy. | P3, P7 |
| 8 | Q/K norm and the RoPE pairing become the architecture's policies. Tied output weights fall back to the embedding. `gguf2hpp` skips a key it cannot name. | a second architecture |
| 9 | Test fixtures are keyed by the model. | a second model's tests |

Then the first real uses, each a feature of its own: a Llama model with its llama.cpp
dump, and the RTX 5070 measured on the card.

## What this design leaves out

- **Unrolling the layers, or a tensor's offset as a template argument.** That is 196
  instantiations of the row loop for one table load saved per phase.
- **A graph description shared by the reference pass and the megakernel.** The reference
  pass is useful because it is independent. A shared description would fail in both at once.
- **Teaching `gguf2hpp` about architectures.** The tool stays ignorant of the engine.
- **A binary for two GPUs.** The schedule would become run-time values in the hot loop.
- **A barrier interface.** One implementation exists, and all 170 barriers plus the launch
  cost 0.14 ms of a 2.30 ms step.

## How this was produced

- **Adversaries:** three, on different models, each in its own worktree. Every finding in
  the table was rerun by the session that wrote this file, except tied output weights and
  RoPE scaling, which were confirmed by reading the source.
- **Designers:** three, independently. All three chose the same shape: one value for the
  model's dimensions, typed tensors, an architecture type, a GPU type that owns its CUDA
  architecture, proofs in a host header. The base is the design with the seven paradigms.
  From the second came "threads is derived" and the per-step SASS check. From the third
  came the gate, the capacity facts on the GPU type and one assertion per fact.
- **Checked:** the base design's constructs compile under nvcc 13.5 as C++20 for sm_89 and
  sm_120 and under g++ 16 as C++23, and six wrong combinations each fail on their own
  assertion.
- **Not checked:** nothing was run on a second GPU or with a real second model. The other
  models' dimensions come from memory and from model cards, not from files.
