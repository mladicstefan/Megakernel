# Reference dumps

A user runs `tools/reference/run.sh` and gets llama.cpp's per-layer intermediate tensors
and logits for a committed prompt, one `llama_decode` call per token, as F32 `.bin` files
with a `manifest.json`. The script builds llama.cpp at a pinned tag on first use, checks
every file against the checksums committed in `tests/data/reference/<prompt-id>/`, and
checks that the dump's greedy chain equals `llama-simple`'s output. Phases C and E compare
their tensors with these files.

## Sub-features

- `ref-build`: `tools/reference/build.sh` clones llama.cpp `b12200` into
  `~/.cache/megakernel/llama.cpp-b12200`, builds `ref-dump`, `llama-simple` and
  `llama-tokenize` (CPU only), and prints their paths. A second run prints
  `ninja: no work to do.` and the same four paths.
- `ref-tokens`: `llama-tokenize` on `prompt.txt` gives exactly `tokens.txt` (14 IDs, no
  BOS, no chat template).
- `ref-dump`: 6737 tensors: 479 per prompt position (17 per layer × 28, plus `embd`,
  `result_norm` and `result_output`) for positions 0 to 13, and `result_output` alone for
  the 31 generated positions 14 to 44. All values are finite, and every file size matches
  its shape.
- `ref-checksums`: every sha256 equals the committed `manifest.json`. A second run gives
  a byte-identical `manifest.json`.
- `ref-greedy`: the 32 argmax tokens equal `greedy.txt`, which holds `llama-simple`'s
  greedy output re-tokenized, and decode to the same text.

## How to get to it (user POV)

- From the repo root, run `tools/reference/run.sh` (prompt `planets`) or
  `tools/reference/run.sh <prompt-id>`. It needs network access the first time only.
- The dump lands in `~/.cache/megakernel/reference/b12200/<prompt-id>/`:
  `pos<PPP>/blk<LL>.<name>.bin`, `pos<PPP>/<name>.bin` for the global tensors, and
  `manifest.json`.
- `MODEL=<file>` overrides the GGUF. The default is `models/Qwen_Qwen3-0.6B-bf16.gguf`
  in the main checkout, so it works from a worktree too.
- `run.sh <prompt-id> --update` rewrites `tokens.txt`, `manifest.json` and `greedy.txt`.
  To add a prompt, create `tests/data/reference/<id>/prompt.txt` and run with `--update`.

## Driving it with verify.sh

There is no `verify.sh` subcommand: `run.sh` is already one command that prints every
check. Keep its output as evidence.

Preconditions:

- `models/Qwen_Qwen3-0.6B-bf16.gguf` exists in the main checkout (doctor shows
  `ok Qwen model present`). Without it, report this feature as skipped.
- `E=build/verify-evidence/<id>`, and `mkdir -p $E`.

- **Build.** Run `tools/reference/build.sh > $E/ref-build.stdout`. On a warm cache, expect
  exit 0 in well under a second, `ninja: no work to do.` on stderr, and four lines naming
  `llama.cpp:`, `ref-dump:`, `llama-simple:` and `llama-tokenize:`. A cold cache clones
  and builds for about 2 minutes on 16 threads.
- **Run.** Run `tools/reference/run.sh > $E/ref-run.log; echo exit=$?`. Expect `exit=0`
  and these lines:
  - `tokens: match tokens.txt ([785, 32875, ...])`
  - `dump: <ms> ms, tensors: 6737`
  - `tensors: 6737, all sizes match their shapes, all values finite`
  - `checksums: all 6737 match .../manifest.json`
  - `greedy: llama-simple:` and `greedy: ref-dump:` with the same 32 IDs, starting
    `[25, 362, 8, 43562`
  - `greedy: match greedy.txt (32 tokens), same text: : A) Mercury, Venus, Earth, ...`
  - `dump dir: ... (70M)`
- **Rerun is identical.** Copy `~/.cache/megakernel/reference/b12200/planets/manifest.json`
  to `$E/manifest.1.json`, run `run.sh` again, and `cmp` the new manifest with it. Expect
  no output.
- **The check catches a change.** Copy the dump directory to `$S/mut`, overwrite one byte
  of `$S/mut/pos005/blk13.Kcur_rope.bin` with `dd ... conv=notrunc`, and run
  `python3 -I tools/reference/check.py $S/mut tests/data/reference/planets/manifest.json`.
  Expect exit 1 and `checksums: 1 of 6737 tensors differ`. Never mutate the cache dump
  itself.

## Gotchas

- `run.sh` never needs the repo's CMake build, and the repo build never needs llama.cpp.
  `tools/reference/CMakeLists.txt` is a separate project that `build.sh` configures.
- `tensors[].dtype` is the type of the graph node, which is `f32` for every dumped tensor.
  The weights are BF16. The KV cache is F16 (llama.cpp's default, set explicitly in
  `dump.cpp`), so attention reads K and V rounded to F16, while `Kcur_rope` and `Vcur` are
  the F32 values before that rounding.
- llama.cpp names three nodes `Qcur-<layer>` (the matmul, its reshape, the RoPE output).
  The dump picks them by op and calls them `Qcur` and `Qcur_rope`. The output projection
  is unnamed in llama.cpp; the dump calls it `attn_out`. `ref-dump` exits non-zero if a
  step captures more or fewer than 479 tensors, which catches a renamed node after a tag
  bump.
- The dump doesn't depend on the thread count: 1, 4, 8 and 16 threads gave identical files.
  `threads` in the manifest is still 8, so a thread change shows up as a manifest
  difference.
- `n_ctx` in the manifest is 256: llama.cpp pads the requested 45 up to 256.
- `llama-simple` prefills the prompt as one batch, while `ref-dump` decodes it one token
  at a time. The greedy tokens still agree for `planets`. A new prompt could diverge on a
  near tie; check the logit margin at that position before calling it a bug.
- `llama-simple` ends its output with a newline. The script strips it before
  re-tokenizing, because `",\n"` is one token (345), not `","` (11).
- `du` is aliased to `sudo du` in the user's shell, so use `command du`.
- Changing `LLAMA_TAG` in `build.sh` means a new cache directory and new checksums: rerun
  with `--update` and explain every difference in the commit.
