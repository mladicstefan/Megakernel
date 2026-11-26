# Perf gate

A change to the megakernel proves that it kept the kernel's size and speed. `gate` prints,
for the engine build of a run, the registers, spill bytes and static shared memory that
ptxas reports for the megakernel on sm_89, its SASS instruction count, and the median and
range of ms per step over three runs of `forward --megakernel --generate 128`. The change
reports that output next to the baseline below and stays inside the limits.

## Sub-features

- `gate-size`: `registers`, `spills` and `smem` come from `ptxas -v` on the sm_89 PTX the
  build embedded in `megakernel.cu.o`. The gate fails if the SASS of that rerun differs
  from the build's, so the numbers are the build's.
- `gate-sass`: `sass` counts the instructions of the megakernel in the build's sm_89 SASS
  (`cuobjdump -sass`). The listing is kept as `gate.sm_89.sass`, to diff against a baseline
  run's.
- `gate-speed`: `ms/step` is the median of three runs, with the range and the step count.
  Each run must exit 0 and start with the tokens of `tests/data/reference/planets/greedy.txt`,
  or the gate fails.
- `gate-limits`: a change stays inside these, against the baseline:

  | Number | Baseline (measured on `master` 38bb3ad, 2026-11-26) | Limit |
  |---|---|---|
  | registers | 123 | the same, and at most 128 (a block of this grid can have no more) |
  | spills | 0 bytes stored, 0 bytes loaded | 0 |
  | smem | 23,008 bytes static | the same |
  | sass | 5,848 instructions | within 1% (5,790 to 5,906) |
  | ms/step | 2.269 median (2.268 to 2.275), 141 steps | median at most 2.33 |

  A change that moves a number on purpose says why in its commit, and that number becomes
  the next baseline.

## How to get to it (user POV)

- There is no user command. A developer runs the gate before committing a change to the
  kernel, and pastes its output into the commit message.

## Driving it with verify.sh

Preconditions:

- Doctor passes, the Qwen model is present, nvcc's directory has `ptxas` and `cuobjdump`,
  and a CUDA GPU is visible to `nvidia-smi`.
- `RUN_ID=<id> $V engine models/Qwen_Qwen3-0.6B-bf16.gguf` has built the engine in this run.

- **Gate.** Run `RUN_ID=<id> $V gate`. It builds `forward` if needed (`gate.build.log`) and
  prints six lines: the tree it measured, then `registers`, `spills`, `smem`, `sass` and
  `ms/step`. Expect exit 0 and the baseline numbers above on an unchanged kernel.
- **Repeat.** Run the gate in a second fresh run (`launch`, `engine`, `gate`). The first
  five lines are identical; only `ms/step` moves.
- **Proof.** `gate.txt` (the printed lines), `gate.ptxas.log`, `gate.sm_89.sass` and
  `gate.run{1,2,3}.stdout` in the run's evidence directory.

## Gotchas

- The GPU must be idle for the timing. The last line shows its utilisation just before the
  runs; another job on the GPU makes every step slower, so rerun on an idle GPU before
  judging a speed limit.
- The first step of a process also loads the kernel. Every run here is one process of 141
  steps, so compare only runs of the same length.
- The static numbers do not depend on the GPU or its load; only `ms/step` does.
- The gate measures sm_89 only. The build also compiles sm_120 (124 registers at the
  baseline), which this GPU cannot run.
- The engine build is a Debug build, which only changes host code: device code is
  optimised the same either way (no `-G`).
