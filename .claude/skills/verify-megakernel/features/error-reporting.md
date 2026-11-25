# Error reporting

When gguf2hpp cannot do its job, it prints one line to stderr that names the input and the
problem, exits non-zero, and leaves no output file and no `.tmp` file behind.

## Sub-features

- `err-usage`: wrong argument count prints `usage: gguf2hpp <in.gguf> <out.hpp>` and exits 2.
- `err-open`: a missing file, a directory, or a device exits 1 with `cannot open (...)`.
- `err-parse`: a truncated file or a non-GGUF file exits 1, naming the error and the byte
  position.
- `err-write`: an output path in a missing directory exits 1 with `cannot write <out>.tmp`.
- `err-no-partial`: no failure leaves `<out>` or `<out>.tmp` behind.

## How to get to it (user POV)

- Run `gguf2hpp` with bad arguments, or on a bad file, from a shell.
- In a build, the same messages appear in the `Generating ...` step, and the build fails.

## Driving it with verify.sh

Preconditions:

- Doctor passes.
- `S=${TMPDIR:-/tmp}/megakernel-verify/<id>` is the run's scratch directory.

- **Usage.** Doctor already checks it. For evidence, run
  `$S/build/gguf2hpp; echo exit=$?`. Expect the usage line and `exit=2`.
- **Missing file.** Run `RUN_ID=<id> $V gen /nonexistent.gguf missing`. Expect `exit=1`,
  `output=none`, and stderr
  `... /nonexistent.gguf: cannot open (missing, not a regular file, or mmap failed) at byte 0`.
- **Directory and device.** Run `gen` with `/` and with `/dev/null` as input. Expect the
  same `cannot open` line, `exit=1`.
- **Truncated.** Run `head -c 57 tests/data/toy.gguf > $S/cut.gguf`, then
  `RUN_ID=<id> $V gen $S/cut.gguf cut`. Expect `exit=1` and `truncated at byte 56`. The
  first value's 8-byte length starts at byte 56, and only 1 byte of it is left.
- **Not GGUF.** Run `RUN_ID=<id> $V gen CMakeLists.txt notgguf`. Expect `exit=1` and
  `not a GGUF file (bad magic) at byte 0`.
- **Unwritable output.** Run
  `$S/build/gguf2hpp tests/data/toy.gguf /nonexistent-dir/x.hpp; echo exit=$?`. Expect
  `cannot write /nonexistent-dir/x.hpp.tmp` and `exit=1`.
- **Proof.** For each case, `tmp-left=no` and `output=none`, and the `.stderr` and `.exit`
  files in the evidence directory.

## Gotchas

- `cannot open` doesn't say why: the reader doesn't keep `errno`. That's accepted (see
  ROADMAP.md), not a failure.
- The byte position is where the failing read started, not the file size.
- Emitter errors (bad keys, NaN) are a different message shape, with no byte position;
  see untrusted-input.md.
