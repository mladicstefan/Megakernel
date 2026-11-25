# Roadmap

Status as of 2026-11-25, `master` at `ea2e28f`. 74/74 tests pass, clean under ASan + UBSan.

## GGUF-Header (build-time tool `gguf2hpp`)

| # | Milestone | Status | Where |
|---|-----------|--------|-------|
| 1 | Cursor: bounds-checked `read<T>()`, `read_string()` | done | `reader.hpp` |
| 2 | MappedFile: mmap, move-only via `unique_ptr` | done | `reader.hpp/.cpp` |
| 3 | Fixed header: magic, version, counts | done | `parse()` |
| 4 | Metadata walk, all 13 value types (nested arrays rejected) | done | `reader.cpp` |
| 5 | Tensor infos, alignment, data_start, sizes, extents, duplicates | done | `reader.cpp` |
| 6 | Hash: XXH3 over `[0, data_start)` seeded with file size | done | `parse()` |
| 7 | Emitter + generated `toy.hpp` checked by `static_assert` | done | `emitter.cpp`, `src/model_types.hpp` |
| 8 | Engine `main` includes `model.hpp`, prints layer and tensor counts | **next** | `src/main.cpp` |

Beyond the plan, done in milestone 7: a per-layer table (`model::layers[i].q`, numeric
order) and a generated `qwen.hpp` checked in `emitter_test` when the model is present.

### Milestone 7 measurements (Qwen3-0.6B BF16)

| Measure | Expected | Measured |
|---------|----------|----------|
| Generated header size | ~35 KB | 43.4 KB: tensor table 30.7, chat template 5.2, layer table 4.8 (not in the estimate), rest 2.7 |
| Output deterministic | yes | yes (`cmp` of two runs) |
| Compiles with g++ `-std=c++23` and nvcc `-std=c++20` | yes | yes; documented `constexpr Layer L = model::layers[3];` runs correctly on the GPU |
| Tensor names in a binary that uses one tensor via `mk::find` | the used one or none | none |

## Next: milestone 8

- `src/main.cpp` includes `model.hpp` and prints `model::meta::arch::block_count` and
  `model::tensors.size()` (expect 28 and 311 for Qwen).
- Configure with `-DMODEL_GGUF=models/Qwen_Qwen3-0.6B-bf16.gguf`; CMake already
  generates `model.hpp` and puts `generated/` and `src/` on the engine's include path.

## Open items (accepted for now)

- Tensor offsets are not checked to be multiples of the alignment. Add before the
  engine relies on aligned loads.
- `BadTensorExtent` reports `data_start`, not which tensor failed.
- `OpenFailed` does not keep `errno`.
- The toy hash test value was recorded from this code, not an outside tool.
- Key segments that are C++ keywords, and leaf/prefix key clashes, are not rejected by
  the emitter; the generated header fails to compile instead.
- Device code indexing `model::layers` with a run-time index needs a copy through a
  `__host__ __device__ constexpr` helper, which nvcc warns about without
  `--expt-relaxed-constexpr`. A compile-time index has no such problem.
- Once the engine loads weights, it must link `xxhash` and recompute the hash at
  startup to check the file is the one it was compiled for.

## Later (out of scope until the header tool is done)

The engine itself: a single persistent CUDA kernel running the whole forward pass, with
the schedule computed at compile time (see `README.md`).
