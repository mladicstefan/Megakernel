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
  the <name> (<n> SMs), this GPU has <m> SMs`. A `-DMK_GPU=Rtx5070` build drives it on the
  development machine: `built for the RTX 5070 (48 SMs), this GPU has 66 SMs`.
  `Megakernel<C>::make` takes the `VerifiedDevice<C::Gpu>` that this check makes
  (`src/weights/verified_device.hpp`), so a caller that skips it does not compile.
- `mega-capacity`: a configuration the GPU cannot hold fails to build, one message per
  fact: blocks per SM whose shared memory does not fit on an SM, a KV cache that does not
  fit in device memory next to the weights, and an `n_ctx` past the model's
  `context_length`.

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
  `--tokens /nonexistent.txt`, with a file holding `[1, 999999]`, with `--generate 5000`
  and with `--generate 18446744073709551613`: each exits 1 with `cannot open`, `the ID at
  byte 4 is outside the vocabulary of 151936`, `5013 positions are more than the 4096 this
  build's KV cache holds`, or `18446744073709551615 positions are more than the 4096 this
  build's KV cache holds` (the count saturates rather than wrapping). `--generate
  99999999999999999999999` (past SIZE_MAX) and `--generate ''` print the usage line and
  exit 2. ctest runs the first two as `engine.refuses_a_count_that_wraps` and
  `engine.refuses_a_count_past_size_max`.
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
  <embd> <ff> <heads> <kv_heads> <head_dim> <layers> <vocab> [options]` (see `craft` in
  SKILL.md), then
  `RUN_ID=<id> $V engine $S/craft/m.gguf` and `cmake --build $S/engine-build -j --target
  forward`. Run `$F --megakernel --check --generate <n> $S/craft/m.gguf.tokens.txt`, and
  compare the `generated:` line of `RUN_ID=<id> $V run-engine decode --tokens
  $S/craft/m.gguf.tokens.txt --generate <n>` with that of `$F --generate <n>
  $S/craft/m.gguf.tokens.txt`, the reference pass. No source edit: the model type reads
  the shape from `model.hpp`, a block has `heads / kv_heads * head_dim` threads (the PTX's
  `.maxntid` in `megakernel.cu.o`), and the KV cache holds 4,096 positions or the
  `context_length` if fewer (the `positions are more than the <n>` refusal prints it). Every
  width must be a multiple of 256 and `head_dim` of 32. Each fact the grid relies on is its
  own `static_assert` in `megakernel.cu`, over a proof in `schedule.hpp`, so a shape that
  breaks one fails to build with that fact's message. Measured on 2026-11-26, after the
  model became a type:
  - `512 1536 4 2 64 2 1024`: 128 threads. Within tolerance with `--generate 700`
    (`kqv_out` rel L2 6.4e-5), and the same 700 tokens.
  - `512 1536 4 4 128 2 1024`, one query head per KV head: 128 threads. Within tolerance
    with `--generate 700` (`kqv_out` 5.9e-5), and the same 700 tokens.
  - `512 1536 16 2 64 2 1024`: 512 threads. Within tolerance with `--generate 700`
    (`kqv_out` 7.6e-7), and the same 700 tokens.
  - `512 1536 4 2 128 2 1024 ctx=2048`: 256 threads, a cache of 2,048 positions. Within
    tolerance with `--generate 2000` (`kqv_out` 4.4e-4), and the same 2,000 tokens.
  - `2560 9728 32 8 128 36 151936 sparse`, Qwen3-4B's shape: `engine` fails to build with
    `static assertion failed with "the megakernel's static shared memory for this model is
    more than the 48 KB CUDA gives a block: ..."` and no other error. 512 threads and an
    `xs` of 9,728 floats make 76 KB.
  Measured on 2026-11-26 before the model became a type; each derives 256 threads, the
  number typed by hand then:
  - `512 1536 4 2 128 2 1024`: within tolerance with `--generate 700`, and the same 700
    tokens.
  - `512 1536 8 2 64 2 1024`: within tolerance with `--generate 700` (`kqv_out` rel L2
    1.6e-6), and the same 700 tokens. From position 512 on, `attention_merge` loads more
    partials than a block has threads: 4 query heads of up to 66 members.
  - `512 512 8 4 128 2 1024`: within tolerance with `--generate 32`, and the same 32
    tokens. Its attention width, 1,024, is wider than `ff`: the output projection's vector
    is the longest one the shared buffer `xs` holds.
  - `512 1536 4 2 128 2 1024 align=8 pad8`: `engine` fails to build with `static
    assertion failed with "general.alignment is not a multiple of the row kernel's
    alignment"`. The row kernel loads 16-byte words, and every matrix starts 8 bytes past
    a multiple of 16.
- **Capacity.** In a scratch copy (`git archive HEAD | tar -x -C $S/cap`), make one edit,
  then configure it for Qwen and build `engine`:
  - `blocks_per_sm = 5` in `Rtx4070TiSuper` (`src/config/gpu_catalogue.hpp`): `static assertion
    failed with "blocks_per_sm blocks of the megakernel's static shared memory do not fit
    in the shared memory of one SM: lower the GPU type's blocks_per_sm"`. Five blocks of
    23,408 bytes plus 1,024 reserved each are 122,160, above 102,400.
  - `n_ctx = 65536` in `EngineConfig` (`src/config/config.hpp`), in place of the `std::min`:
    `"n_ctx is more positions than the model's context_length: lower n_ctx"`. Qwen's
    `context_length` is 32,768.
  - `n_ctx = 1u << 20`: that message and `"the weights and a KV cache of n_ctx positions do
    not fit in the GPU's device memory: lower n_ctx"`. The cache alone would be 120 GB.
  Measured on 2026-11-26. `schedule_test` feeds each proof failing numbers.
- **GPU.** Configure with `-DMK_GPU=Rtx5070` and build `engine` and `forward`. Both exit 1
  on this machine with `built for the RTX 5070 (48 SMs), this GPU has 66 SMs`, after the
  `file:` line and before the upload.
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
- The grid is 132 blocks (`Rtx4070TiSuper::blocks_per_sm` is 2, on 66 SMs) of 256 threads
  for Qwen3-0.6B, and every barrier waits for all of them. The threads come from the model,
  so `blocks_per_sm` was measured with 256 threads only. 5 or more blocks per SM fail to build: their
  shared memory does not fit in an SM (see Capacity). `Megakernel::make` still fails with
  `the GPU cannot keep the whole grid resident` if the device cannot run that many blocks
  per SM at once, and with `its static shared memory is not what megakernel_shared_bytes in
  schedule.hpp counts` if a `__shared__` buffer was added without counting it. A barrier
  that waits about a second gives up, and the step fails with `a barrier gave up waiting
  for the rest of the grid`.
- `--megakernel` and `--compare` cannot be combined. Compare the megakernel with the
  reference pass, and the reference pass with llama.cpp.
