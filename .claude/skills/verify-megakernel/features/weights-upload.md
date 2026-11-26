# Weights upload

The engine loads the GGUF it was compiled for and refuses any other file. It checks the
file's size and header hash against `model.hpp`, uploads the tensor data to the GPU in one
allocation, and proves the copy: a kernel checksums every tensor at
`base + model::tensors[i].offset`, the host checksums the same bytes in the file, and the
engine prints how many of them match.

## Sub-features

- `weights-default-path`: `engine` with no argument loads the `MODEL_GGUF` it was
  configured with. `engine <path>` loads that path.
- `weights-verify`: the second line is
  `file: <path> verified (<size> bytes, header hash 0x...)`.
- `weights-upload`: the third line is `upload: <bytes> bytes in <s> s, <rate> GB/s`.
- `weights-match`: the fourth line is `checksums: 311/311 tensors match, layers[27].q
  matches through model::layers` for Qwen, and `checksums: 2/2 tensors match` for toy,
  which has no layer table.
- `weights-kernel-time`: the fifth line is `checksum kernel: <ms> ms, <rate> GB/s (median of
  5 runs, <min> to <max> ms; a first measurement, not tuned)`.
- `weights-reject-hash`: a file of the right size with a different header exits 1 with
  `header hash 0x... differs from the 0x... this engine was built for`.
- `weights-reject-size`: a file of another size exits 1 with
  `file size <n> differs from the <n> this engine was built for`.
- `weights-reject-parse`: a file of the right size that is not a valid GGUF exits 1 with the
  message gguf2hpp prints for it, for example `not a GGUF file (bad magic) at byte 0`.
- `weights-reject-missing`: a missing path exits 1 with `cannot open (...) at byte 0`.
- `weights-usage`: a second model path or an unknown option prints `usage: engine
  [model.gguf] [--tokens <tokens.txt> [--generate <n>]]` and exits 2.
- `weights-gpu`: on a GPU whose SM count is not the one the build targets (`mk::BuildGpu`,
  66 for the RTX 4070 Ti SUPER), the engine exits 1 with `built for the RTX 4070 Ti SUPER
  (66 SMs), this GPU has <n> SMs` before it uploads anything. On the development machine,
  which has only that GPU, drive it with a `-DMK_GPU=Rtx5070` build (the default detects this GPU): `built for the RTX
  5070 (48 SMs), this GPU has 66 SMs`. The check is `mk::VerifiedDevice<G>::check()`, and
  `weights_test` drives its refusal with a GPU type of 0 SMs.
- `weights-negative-control`: `weights_test` uploads the file's bytes starting 32 bytes
  early, and no tensor's device checksum matches the file's.
- `weights-memcheck`: `compute-sanitizer --tool memcheck` on the engine reports 0 errors.

## How to get to it (user POV)

- Configure with `-DMODEL_GGUF=$PWD/models/Qwen_Qwen3-0.6B-bf16.gguf`, build, and run
  `<dir>/engine` or `<dir>/engine <other.gguf>`.
- The negative control is a ctest test in that build directory: `<dir>/weights_test`.

## Driving it with verify.sh

Preconditions:

- Doctor passes, the Qwen model is present, and a CUDA GPU is visible to `nvidia-smi`.
- `S=${TMPDIR:-/tmp}/megakernel-verify/<id>`, `M=$PWD/models/Qwen_Qwen3-0.6B-bf16.gguf` and
  `EV=build/verify-evidence/<id>`.
- The two wrong copies of Qwen take 1.5 GB each in `$S`. Cleanup removes them.

- **Upload and match.** Run `RUN_ID=<id> $V engine $M`, then
  `RUN_ID=<id> $V run-engine qwen`. Expect `exit=0` and the five lines above, with
  `311/311 tensors match, layers[27].q matches`.
- **Path argument.** Run `cp $M $S/same.gguf`, then
  `RUN_ID=<id> $V run-engine same $S/same.gguf`. Expect `exit=0`, the `file:` line naming
  `$S/same.gguf`, and `311/311`.
- **Wrong header.** Run `cp $M $S/flip.gguf`, then
  `printf X | dd of=$S/flip.gguf bs=1 seek=32 conv=notrunc status=none`, then
  `RUN_ID=<id> $V run-engine reject-hash $S/flip.gguf`. Byte 32 is the first character of
  the first metadata key, so the file still parses. Expect `exit=1`, a stdout of only the
  first line, and stderr `engine: .../flip.gguf: header hash 0x... differs from the
  0xcdbb83aa1306c816 this engine was built for`.
- **Truncated.** Run `cp $M $S/short.gguf`, then `truncate -s -1 $S/short.gguf`, then
  `RUN_ID=<id> $V run-engine reject-size $S/short.gguf`. Expect `exit=1` and stderr
  `file size 1509347487 differs from the 1509347488 this engine was built for`.
- **Not a GGUF.** Run `printf X | dd of=$S/flip.gguf bs=1 seek=0 conv=notrunc status=none`,
  then `RUN_ID=<id> $V run-engine reject-parse $S/flip.gguf`. Expect `exit=1` and
  `not a GGUF file (bad magic) at byte 0`.
- **Missing.** Run `RUN_ID=<id> $V run-engine reject-missing /nonexistent.gguf`. Expect
  `exit=1` and `cannot open (missing, not a regular file, or mmap failed) at byte 0`.
- **Negative control.** Run `cmake --build $S/engine-build -j --target weights_test`, then
  `$S/engine-build/weights_test > $EV/weights_test.log 2>&1; echo exit=$?`. Expect `exit=0`
  and `[  PASSED  ] 2 tests.`
- **Memcheck.** Run
  `compute-sanitizer --tool memcheck $S/engine-build/engine > $EV/memcheck.log 2>&1`.
  Expect `311/311` and `ERROR SUMMARY: 0 errors`. It takes a few seconds.
- **Toy.** Run the build-integration recipe on `$S/toy-copy.gguf` in a run of its own.
  Expect `checksums: 2/2 tensors match` and no `layers` text.
- **Measurements.** Run `run-engine` seven times for the warm numbers. For a cold number,
  evict the file first and show that it worked:
  `dd if=$M iflag=nocache count=0 of=/dev/null; fincore $M` must print `RES 0B`. Measured
  on the RTX 4070 Ti SUPER on 2026-11-25:
  - upload, warm page cache: 0.167 s, 9.0 GB/s (median of 7, 0.163 to 0.201 s)
  - upload, cold page cache: 1.55 s, 0.97 GB/s (3 runs, 1.54 to 1.60 s)
  - checksum kernel: 2.41 ms, 623 GB/s (7 engine runs of 5 launches each)
- **Proof.** For each `run-engine` case, the `.cmd`, `.stdout`, `.stderr` and `.exit` files.
  For the rest, `weights_test.log` and `memcheck.log`.

## Gotchas

- Only the header is hashed: bytes `[0, data_start)`, seeded with the file size. A file
  with the same header and different tensor data passes the check. The checksums compare
  the device copy with that file, so they still match.
- The size is checked before the file is parsed. A truncated copy reports a size mismatch,
  not the parse error gguf2hpp gives for it.
- The first stdout line is printed before the file is checked, so a rejected file still
  prints it.
- The cold upload is limited by reading the file, not by the copy. On the development
  machine the model is stored zstd-compressed by btrfs under LUKS, and a cold `dd` of it
  alone reads at 1.0 GB/s.
- `fincore` reports a non-zero `RES` after eviction if another process has the file mapped.
  Then the cold number is not cold. Evict a private copy of the model instead.
- The idle PCIe link reports generation 1. During the upload it reports generation 3:
  `nvidia-smi --query-gpu=pcie.link.gen.current --format=csv` in a loop while the engine
  runs.
- Under memcheck the kernel time is about 25 times longer. Never report it as a measurement.
- A model with a tensor that is not a whole number of 8-byte words, or with
  `general.alignment` below 8, fails a `static_assert` in `src/weights/weights.hpp`. That is
  the accepted limit of the checksum, not a build bug.
