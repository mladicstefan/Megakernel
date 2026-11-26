# Megakernel

A user runs `engine --tokens <tokens.txt>` and gets the token IDs that the megakernel
generates greedily, with its time per step and tokens per second. One kernel launch runs a
whole decode step. `forward --megakernel <tokens.txt>` does the same from the test tool. With `--check` the reference pass decodes the same tokens next to it,
the megakernel runs each phase on the reference's operands, and the tool prints the worst
difference per tensor and exits 1 when a limit is exceeded.

## Sub-features

- `mega-greedy`: for `tests/data/reference/planets/tokens.txt`, the `generated:` line
  equals `tests/data/reference/planets/greedy.txt`.
- `mega-engine`: `engine [model.gguf] --tokens <tokens.txt> [--generate <n>]` prints the
  first three lines of the engine, then `generated:` and `decode:`. It skips the checksum
  proof. A build for a model without a forward pass (toy) exits 1 with `this build cannot
  decode: it has no forward pass for a 'toy' model`.
- `mega-check`: `--megakernel --check` compares the nine tensors the megakernel writes
  (`embd`, `Qcur`, `Kcur`, `Vcur`, `kqv_out`, `ffn_inp`, `ffn_swiglu`, `l_out`,
  `result_output`) at every position, and its token with the reference's. Limits: rel L2
  5e-5, 1-cosine 1e-10, KL 2e-9 nats, equal tokens. `kqv_out` has its own: rel L2 5e-3,
  1-cosine 5e-6.
- `mega-speed`: the `decode:` line reports steps, ms per step and tokens/s.
- `mega-sanitizers`: `compute-sanitizer` with `memcheck`, `racecheck` and `synccheck`
  reports 0 errors and 0 hazards.
- `mega-gpu`: on a GPU with another SM count than the build's, it exits 1 with `built for
  the RTX 4070 Ti SUPER (66 SMs), this GPU has <n> SMs`.

## How to get to it (user POV)

- Configure with `-DMODEL_GGUF=$PWD/models/Qwen_Qwen3-0.6B-bf16.gguf`, build, and run
  `<dir>/forward --megakernel tests/data/reference/planets/tokens.txt`.
- `--generate <n>` sets how many tokens to generate. `--check` adds the comparison.
- ctest runs the engine as `engine.generates_the_reference_tokens`, and the comparison as `megakernel.each_phase_matches_reference` (45 positions)
  and `megakernel.matches_reference_at_333_positions`. They need the
  model and a GPU, but no llama.cpp dump.

## Driving it with verify.sh

Preconditions:

- Doctor passes, the Qwen model is present, and a CUDA GPU is visible to `nvidia-smi`.
- `S=${TMPDIR:-/tmp}/megakernel-verify/<id>`, `EV=build/verify-evidence/<id>`,
  `P=tests/data/reference/planets` and `F=$S/engine-build/forward`.
- Build as in [forward-pass.md](./forward-pass.md): `RUN_ID=<id> $V engine
  models/Qwen_Qwen3-0.6B-bf16.gguf`, then `cmake --build $S/engine-build -j --target
  forward`.

- **Engine.** Run `RUN_ID=<id> $V run-engine decode --tokens $P/tokens.txt`. Expect
  `exit=0`, a `generated:` line equal to `$P/greedy.txt`, and a `decode:` line. Run it with
  `--tokens /nonexistent.txt`, with a file holding `[1, 999999]`, and with `--generate
  5000`: each exits 1 with `cannot open`, `the ID at byte 4 is outside the vocabulary of
  151936`, or `5013 positions are more than the 4096 this build's KV cache holds`.
- **Greedy.** Run `$F --megakernel $P/tokens.txt > $EV/mega.stdout; echo exit=$?`. Expect
  `exit=0`, and `diff <(sed -n 's/^generated: //p' $EV/mega.stdout) $P/greedy.txt` to
  print nothing.
- **Check.** Run `$F --megakernel --check $P/tokens.txt > $EV/mega-check.stdout; echo
  exit=$?`. Expect `exit=0`, nine table rows, `the megakernel's token is the reference's
  at 45 of 45 positions`, and `result: within tolerance`. `embd` is exactly 0, `kqv_out`
  is the largest at about 4e-4, and every other row is below 6e-6.
- **Check over many positions.** Run the same with `--generate 587` (600 positions, about
  25 s). Expect `600 of 600 positions` and `within tolerance`, with `kqv_out` near 1.3e-3.
  Up to 128 positions every attention warp sums at most one position, so only a longer
  run exercises the sum over several.
- **Speed.** Run `$F --megakernel --generate 128 $P/tokens.txt` three times and keep the
  `decode:` lines. Measured on the RTX 4070 Ti SUPER on 2026-11-26: 2.30 to 2.31 ms per
  step, 433 to 436 tokens/s. With `--generate 1024`: 2.40 to 2.41 ms, 414 to 416
  tokens/s. With `--generate 4000`: 2.69 ms, 372 tokens/s.
- **Sanitizers.** For each of `memcheck`, `racecheck` and `synccheck`, run
  `compute-sanitizer --tool <tool> $F --megakernel --generate 3 $P/tokens.txt >
  $EV/mega-<tool>.log 2>&1`. Expect `generated: [25, 362, 8]` and `0 errors` or `0
  hazards`. `racecheck` takes about 4 minutes; the others take seconds.
- **Other shapes.** Give each model its own `RUN_ID`. Run `RUN_ID=<id> $V craft qwen3 m
  <embd> <ff> <heads> <kv_heads> <head_dim> <layers> <vocab> [align=<n>] [pad8]`, then
  `RUN_ID=<id> $V engine $S/craft/m.gguf` and `cmake --build $S/engine-build -j --target
  forward`. Run `$F --megakernel --check --generate <n> $S/craft/m.gguf.tokens.txt`, and
  compare the `generated:` line of `RUN_ID=<id> $V run-engine decode --tokens
  $S/craft/m.gguf.tokens.txt --generate <n>` with that of `$F --generate <n>
  $S/craft/m.gguf.tokens.txt`, the reference pass. The grid needs `heads / kv_heads *
  head_dim` to be 256 and every width to be a multiple of 256. Measured on 2026-11-26:
  - `512 1536 4 2 128 2 1024`: within tolerance with `--generate 700`, and the same 700
    tokens.
  - `512 1536 8 2 64 2 1024`: within tolerance with `--generate 498`, `kqv_out` outside it
    with `--generate 700` (rel L2 1.6e-1). From position 512 on, `attention_merge` has 4
    query heads of more than 64 members to load, and 256 threads.
  - `512 512 8 4 128 2 1024`: `--generate 32` fails with `an illegal memory access was
    encountered`. The shared buffer `xs` holds `ff` floats, and the output projection
    loads `heads * head_dim`.
  - `512 1536 4 2 128 2 1024 align=8 pad8`: the engine builds and its checksums match,
    then `--tokens` fails with `misaligned address`. The row kernel loads 16-byte words, and
    every matrix starts 8 bytes past a multiple of 16.
- **Proof.** The `.stdout` files and the three sanitizer logs.

## Gotchas

- The speed falls as the context grows, because attention reads the whole KV cache: 0.10
  ms per step more for every 1,000 positions (measured between 1,037 and 4,013). Report
  the number of positions with every tokens/s figure.
- To see where a step's time goes, build scratch copies with parts switched off: `run_layer`
  without the `Ps::run(s)` call leaves the barriers, and an empty row range in the logits
  block removes the output matrix. Measured over 141 positions: 0.14 ms with neither (launch,
  read-back and barriers), 0.65 ms with the output matrix only, 1.79 ms with the layers
  only, 2.27 ms in full.
- The first step of a process also loads the kernel, so a short run reads slower per step
  than a long one would at the same context. Compare runs of the same length.
- With `--check` the `decode:` time includes uploading the reference's trace for every
  step. Never report it as the megakernel's speed.
- With `--check` both decoders follow the reference's tokens. The megakernel's own chain
  is tested by the run without `--check`.
- `kqv_out` differs more than the other tensors because K and V are rounded to BF16 on
  their way into the cache. A sum in another order can move a value across a rounding
  boundary, which changes one cache entry by one BF16 step. With an F32 cache
  (`EngineConfig::Kv = KvF32`, in a scratch copy) `kqv_out` is at 3.7e-6.
- The grid is 132 blocks of 256 threads (`EngineConfig::blocks_per_sm` is 2, on 66 SMs),
  and every barrier waits for all of them.
  `Megakernel::make` fails with `the GPU cannot keep the whole grid resident` if the
  device cannot run that many blocks per SM at once: 5 or more do not fit on this GPU. A barrier that waits about a second gives up,
  and the step fails with `a barrier gave up waiting for the rest of the grid`.
- `--megakernel` and `--compare` cannot be combined. Compare the megakernel with the
  reference pass, and the reference pass with llama.cpp.
