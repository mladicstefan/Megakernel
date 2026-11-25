# Megakernel

A user runs `forward --megakernel <tokens.txt>` and gets the token IDs that the megakernel
generates greedily, with its time per step and tokens per second. One kernel launch runs a
whole decode step. With `--check` the reference pass decodes the same tokens next to it,
the megakernel runs each phase on the reference's operands, and the tool prints the worst
difference per tensor and exits 1 when a limit is exceeded.

## Sub-features

- `mega-greedy`: for `tests/data/reference/planets/tokens.txt`, the `generated:` line
  equals `tests/data/reference/planets/greedy.txt`.
- `mega-check`: `--megakernel --check` compares the nine tensors the megakernel writes
  (`embd`, `Qcur`, `Kcur`, `Vcur`, `kqv_out`, `ffn_inp`, `ffn_swiglu`, `l_out`,
  `result_output`) at all 45 positions, and its token with the reference's. Limits: rel L2
  2e-3, 1-cosine 5e-7, KL 1e-9 nats, equal tokens.
- `mega-speed`: the `decode:` line reports steps, ms per step and tokens/s.
- `mega-sanitizers`: `compute-sanitizer` with `memcheck`, `racecheck` and `synccheck`
  reports 0 errors and 0 hazards.
- `mega-gpu`: on a GPU with another SM count than the build's, it exits 1 with `built for
  the RTX 4070 Ti SUPER (66 SMs), this GPU has <n> SMs`.

## How to get to it (user POV)

- Configure with `-DMODEL_GGUF=$PWD/models/Qwen_Qwen3-0.6B-bf16.gguf`, build, and run
  `<dir>/forward --megakernel tests/data/reference/planets/tokens.txt`.
- `--generate <n>` sets how many tokens to generate. `--check` adds the comparison.
- ctest runs the comparison as `megakernel.each_phase_matches_reference`. It needs the
  model and a GPU, but no llama.cpp dump.

## Driving it with verify.sh

Preconditions:

- Doctor passes, the Qwen model is present, and a CUDA GPU is visible to `nvidia-smi`.
- `S=${TMPDIR:-/tmp}/megakernel-verify/<id>`, `EV=build/verify-evidence/<id>`,
  `P=tests/data/reference/planets` and `F=$S/engine-build/forward`.
- Build as in [forward-pass.md](./forward-pass.md): `RUN_ID=<id> $V engine
  models/Qwen_Qwen3-0.6B-bf16.gguf`, then `cmake --build $S/engine-build -j --target
  forward`.

- **Greedy.** Run `$F --megakernel $P/tokens.txt > $EV/mega.stdout; echo exit=$?`. Expect
  `exit=0`, and `diff <(sed -n 's/^generated: //p' $EV/mega.stdout) $P/greedy.txt` to
  print nothing.
- **Check.** Run `$F --megakernel --check $P/tokens.txt > $EV/mega-check.stdout; echo
  exit=$?`. Expect `exit=0`, nine table rows, `the megakernel's token is the reference's
  at 45 of 45 positions`, and `result: within tolerance`. `embd` is exactly 0, `kqv_out`
  is the largest at about 4e-4, and every other row is below 6e-6.
- **Speed.** Run `$F --megakernel --generate 128 $P/tokens.txt` three times and keep the
  `decode:` lines. Measured on the RTX 4070 Ti SUPER on 2026-11-25: 3.11 ms per step, 322
  tokens/s, the same to 0.01 ms in three runs. With `--generate 1024`: 8.15 ms per step.
- **Sanitizers.** For each of `memcheck`, `racecheck` and `synccheck`, run
  `compute-sanitizer --tool <tool> $F --megakernel --generate 3 $P/tokens.txt >
  $EV/mega-<tool>.log 2>&1`. Expect `generated: [25, 362, 8]` and `0 errors` or `0
  hazards`. `racecheck` takes about 4 minutes; the others take seconds.
- **Proof.** The `.stdout` files and the three sanitizer logs.

## Gotchas

- The speed falls as the context grows: attention runs one block per query head and walks
  the cached positions one after another. Report the number of positions with every
  tokens/s figure.
- The first step of a process also loads the kernel, so a short run reads slower per step
  than a long one would at the same context. Compare runs of the same length.
- With `--check` the `decode:` time includes uploading the reference's trace for every
  step. Never report it as the megakernel's speed.
- With `--check` both decoders follow the reference's tokens. The megakernel's own chain
  is tested by the run without `--check`.
- `kqv_out` differs more than the other tensors because K and V are rounded to BF16 on
  their way into the cache. A sum in another order can move a value across a rounding
  boundary, which changes one cache entry by one BF16 step.
- The grid is 264 blocks of 256 threads, and every barrier waits for all of them.
  `Megakernel::make` fails with `the GPU cannot keep the whole grid resident` if the
  device cannot run 4 blocks per SM at once. A barrier that waits about a second gives up,
  and the step fails with `a barrier gave up waiting for the rest of the grid`.
- `--megakernel` and `--compare` cannot be combined. Compare the megakernel with the
  reference pass, and the reference pass with llama.cpp.
