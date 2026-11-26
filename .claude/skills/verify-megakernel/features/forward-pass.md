# Forward pass

A user runs `forward <tokens.txt>` and gets the token IDs that the reference CUDA forward
pass generates greedily after the prompt. With `--compare <dump-dir>` it also checks every
intermediate tensor and the logits against a dump that `tools/reference/run.sh` made with
llama.cpp, prints the worst difference per tensor name, and exits 1 when a limit is
exceeded. The target exists only when `MODEL_GGUF` is a `qwen3` model.

## Sub-features

- `fwd-greedy`: for `tests/data/reference/planets/tokens.txt`, the `generated:` line equals
  `tests/data/reference/planets/greedy.txt`, which is llama.cpp's greedy output.
- `fwd-compare`: `--compare <dump>` decodes in exact arithmetic and compares as it runs: 20
  tensor names for the 14 prompt positions, and the logits for all 45 positions. Limits:
  rel L2 5e-2, 1-cosine 1e-3, KL 1e-3 nats, and equal top-1 tokens.
- `fwd-per-op`: `--compare <dump> --per-op` recomputes each op on the dump's operands, with
  llama.cpp's CPU rounding reproduced, for the 14 prompt positions. Limits: rel L2 5e-4,
  1-cosine 1e-7, KL 1e-9 nats.
- `fwd-deterministic`: two runs print the same lines, apart from the `decode:` timing.
- `fwd-arch-gate`: configuring for a GGUF whose `general.architecture` is not `qwen3`
  prints `No forward pass for this architecture` and builds the engine without `forward`.
  A qwen3 file gets `forward` wherever the key is in the file (see `build-arch` in
  [build-integration.md](./build-integration.md)).
- `fwd-errors`: bad arguments print the usage line and exit 2. An unreadable token file, an
  ID outside the vocabulary, a file without IDs, a missing or wrong-size dump tensor, a
  wrong model file, and more positions than the build's KV cache holds each print one line
  that names the problem, and exit 1.
- `fwd-memcheck`: `compute-sanitizer --tool memcheck` on `forward` reports 0 errors.

## How to get to it (user POV)

- Configure with `-DMODEL_GGUF=$PWD/models/Qwen_Qwen3-0.6B-bf16.gguf` and build. Run
  `<dir>/forward tests/data/reference/planets/tokens.txt`.
- `--generate <n>` sets how many tokens to generate (default 32). `--model <gguf>` loads
  another path; it must be the file the build was configured with.
- For `--compare`, first run `tools/reference/run.sh`. The dump is then in
  `~/.cache/megakernel/reference/b12200/planets`.
- With the dump present at configure time, ctest runs both comparisons as
  `forward.each_op_matches_llama` and `forward.exact_stays_close_to_llama`.

## Driving it with verify.sh

Preconditions:

- Doctor passes, the Qwen model is present, and a CUDA GPU is visible to `nvidia-smi`.
- The reference dump exists: `tools/reference/run.sh` exits 0 (see
  [reference-dumps.md](./reference-dumps.md)). Without it, report `fwd-compare` and
  `fwd-per-op` as skipped.
- `S=${TMPDIR:-/tmp}/megakernel-verify/<id>`, `EV=build/verify-evidence/<id>`,
  `R=~/.cache/megakernel/reference/b12200/planets`,
  `P=tests/data/reference/planets` and `F=$S/engine-build/forward`.

- **Build.** Run `RUN_ID=<id> $V engine models/Qwen_Qwen3-0.6B-bf16.gguf`, then
  `cmake --build $S/engine-build -j --target forward > $EV/forward.build.log 2>&1`. Expect
  exit 0 and no `warning` in the log. `engine.configure.log` has
  `MODEL_GGUF architecture: 'qwen3'`.
- **Greedy.** Run `$F $P/tokens.txt > $EV/greedy.stdout; echo exit=$?`. Expect `exit=0`
  and a `generated:` line whose list equals `$P/greedy.txt`:
  `diff <(sed -n 's/^generated: //p' $EV/greedy.stdout) $P/greedy.txt` prints nothing.
- **Compare as it runs.** Run `$F --compare $R $P/tokens.txt > $EV/compare.stdout; echo
  exit=$?`. Expect `exit=0`, 20 table rows, `top-1 agrees at 45 of 45 positions`, and
  `result: within tolerance`. Measured worst values, with the engine's BF16 cache: rel L2
  3.0e-2, 1-cosine 3.7e-4, KL 3.0e-4.
- **Compare op by op.** Run `$F --compare $R --per-op $P/tokens.txt > $EV/per-op.stdout;
  echo exit=$?`. Expect `exit=0` and `result: within tolerance`. `embd`, `Qcur_rope`,
  `Kcur_rope`, `ffn_inp` and `l_out` are exactly 0. `kqv_out` is the largest at about
  1e-4. Every other row is below 3e-6.
- **Deterministic.** Run the compare command again into `$EV/compare.2.stdout`. Expect
  `diff <(grep -v '^decode:' $EV/compare.stdout) <(grep -v '^decode:' $EV/compare.2.stdout)`
  to print nothing.
- **The comparison catches a bug.** Copy the repo's tracked files to `$S/mut` with
  `git archive HEAD | tar -x -C $S/mut`, change `kNormEps` in `$S/mut/src/reference/forward.cu` to
  `1e-5f`, and build `forward` from that copy. Expect `--per-op` to exit 1 with
  `attn_norm` near 9e-3. Expect the comparison without `--per-op` to exit 0: this bug is
  smaller than llama.cpp's own rounding. Never edit the checkout itself for this.
- **Errors.** Run each of these with `; echo exit=$?`, stderr into `$EV/errors.log`:
  - `$F` gives the usage line and `exit=2`.
  - `$F --per-op $P/tokens.txt` gives the usage line and `exit=2`: `--per-op` needs
    `--compare`.
  - `$F /nonexistent.txt` gives `cannot open` and `exit=1`.
  - A file holding `[1, 2, 999999]` gives `the ID at byte 7 is outside the vocabulary of
    151936` and `exit=1`.
  - `$F --compare /nonexistent $P/tokens.txt` gives `/nonexistent/pos000/embd.bin: cannot
    read` and `exit=1`.
  - `$F --model tests/data/toy.gguf $P/tokens.txt` gives the engine's size-mismatch line
    and `exit=1`.
  - `$F --generate 5000 $P/tokens.txt` gives `5013 positions are more than the 4096 this
    build's KV cache holds` and `exit=1`.
  - `$F --generate 18446744073709551613 $P/tokens.txt`, with and without `--megakernel`,
    gives `18446744073709551615 positions are more than the 4096 this build's KV cache
    holds` and `exit=1`: the count saturates rather than wrapping.
  - `$F --generate 99999999999999999999999 $P/tokens.txt` (past SIZE_MAX) and `$F
    --generate '' $P/tokens.txt` give the usage line and `exit=2`.
- **Memcheck.** Run `compute-sanitizer --tool memcheck $F --generate 3 $P/tokens.txt >
  $EV/memcheck.log 2>&1`. Expect `generated: [25, 362, 8]` and `ERROR SUMMARY: 0 errors`.
  It takes about 15 s.
- **Toy has no forward pass.** In a run of its own, `RUN_ID=<id2> $V engine
  $S2/toy-copy.gguf`. Expect `No forward pass for this architecture` in
  `engine.configure.log`, and `cmake --build $S2/engine-build --target forward` to fail
  with `No rule to make target`.
- **Proof.** The `.stdout` files, `forward.build.log`, `errors.log` and `memcheck.log`.

## Gotchas

- The pass is a correctness reference, not the engine's decode path. It takes about 32 ms
  per step, against about 2.6 ms for llama.cpp's CUDA backend. Never report its speed as
  the engine's.
- Without `--per-op` the differences are llama.cpp's, not the pass's. llama.cpp's CPU
  backend rounds a vector to BF16 before every BF16 matrix product, rounds Q to F16, and
  sums attention in F16. One such product differs by up to 6e-3 from the exact one, and
  the differences grow over 28 layers. So that comparison cannot see a bug smaller than
  about 1e-2. Use `--per-op` to judge an op.
- With `--per-op` the logits always agree, because they are computed from the dump's own
  `result_norm`. Top-1 agreement means something only without `--per-op`.
- `--per-op` stops after the prompt, since a dump has the layers of the prompt's positions
  only. It prints `1 to generate` whatever `--generate` says.
- Top-1 agreement is not guaranteed for other prompts. On `planets` the largest logit
  difference is 0.16 and the smallest gap between llama.cpp's best two tokens is 0.13.
- CMake reads `general.architecture` with the project's reader at configure time, and
  names the model type `Qwen3` (`src/arch/qwen3.hpp`) for `qwen3`. A stale configure
  cannot build the wrong pass: `qwen3.hpp` has a `static_assert` on the architecture.
- The ctest entries are added only when the dump exists at configure time. After creating
  the dump, rerun `cmake`.
- A model whose matrices are not BF16, or whose norm weights are not F32, fails a
  `static_assert` in `src/arch/qwen3.hpp`.
- The KV cache holds 4,096 positions, or the model's `context_length` if that is fewer:
  `EngineConfig<Model, Gpu>::n_ctx`. The 4,096 is CMake's `MK_N_CTX`. The `positions are more than the <n>` errors print it.
