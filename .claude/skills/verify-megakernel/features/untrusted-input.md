# Untrusted input

A GGUF file is untrusted, and its text ends up compiled into the engine. gguf2hpp either
writes every string from the file byte-exactly inside a literal, or refuses the file with an
error that names the key. It never emits a header that compiles into something other than
the file's content.

## Sub-features

- `ut-escape`: tensor names and string values containing `"`, `\`, a newline, `*/`, `??=`,
  NUL, or a literal `\x41B` round-trip byte-exactly.
- `ut-bad-key`: a key segment that is not an identifier (`evil.b-c`, `evil.1a`) is refused.
- `ut-nonfinite`: a NaN or inf float value is refused.
- `ut-zero-tensors`: a file with no tensors gives a valid header with an empty table.

## How to get to it (user POV)

- Run gguf2hpp on a GGUF file from someone else: a download, or a file written by another
  tool.

## Driving it with verify.sh

Preconditions:

- Doctor passes, including `ok gguf-py venv`.
- `S=${TMPDIR:-/tmp}/megakernel-verify/<id>`.

- **Hostile strings.** Run `f=$(RUN_ID=<id> $V craft evil-strings evil)`, then
  `RUN_ID=<id> $V gen $f evil`. Expect `exit=0`.
- **Byte-exact.** Run
  `RUN_ID=<id> $V compile build/verify-evidence/<id>/evil.hpp evil $f.check.hpp`. Expect
  `g++=0 nvcc=0` and `probe: tensors=7 first=q"uote bytes=4`. The check file holds the
  expected bytes as char arrays and `static_assert`s each tensor name against them.
- **Negative control.** Change one byte in a copy of the check file:
  `sed 's/{113, 34/{113, 35/' $f.check.hpp > $S/broken.check.hpp`. Then run `compile` with
  it as `evilneg`. Expect `g++=1`, and `static assertion failed` in `evilneg.gcc.log`. This
  proves the check is included and can fail.
- **Bad keys.** For each of `bad-key` and `digit-key`, run `craft`, then `gen`. Expect
  `exit=1`, `output=none`, and stderr
  `metadata key "evil.b-c": every part between dots must be a C++ identifier` (or
  `"evil.1a"`).
- **Non-finite.** Run `craft nonfinite`, then `gen`. Expect `exit=1` and
  `metadata key "evil.scale": inf and NaN have no literal`.
- **Zero tensors.** Run `craft zero-tensors`, `gen`, then `compile`. Expect `exit=0`,
  `g++=0 nvcc=0`, and `probe: tensors=0`.

## Gotchas

- gguf-py writes strings as UTF-8, so `craft` can't produce invalid UTF-8 or a lone byte
  0xFF. Those are covered by `tests/emitter_test.cpp` on hand-built models. A file-level
  check would need patching the bytes after writing.
- Key segments that are C++ keywords, and keys that are both a leaf and a prefix, are
  accepted shortcuts: the header fails to compile instead of the tool refusing. Don't
  report them as new findings.
- The negative control must fail on g++. If it compiles, the check file wasn't included,
  and the byte-exact proof is void.
