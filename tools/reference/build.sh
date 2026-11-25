#!/usr/bin/env bash
# Builds llama.cpp (CPU only) at a pinned tag plus ref-dump, in a cache outside the repo.
# Idempotent: a second run clones nothing, configures nothing and ninja has no work to do.
set -euo pipefail

# b12200 (2026-11-07): the newest tag over two weeks old on 2026-11-25; it has the Qwen3 graph
# (src/models/qwen3.cpp) and BF16 CPU kernels. The SHA guards against a moved tag.
LLAMA_TAG=b12200
LLAMA_SHA=c19043f67e52f1f1e00d797246e19643bb2f168e

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cache=${XDG_CACHE_HOME:-$HOME/.cache}/megakernel
src=$cache/llama.cpp-$LLAMA_TAG
build=$src/build-ref

if [[ ! -d $src ]]; then
  mkdir -p "$cache"
  rm -rf "$src.tmp"
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$LLAMA_TAG" \
    https://github.com/ggml-org/llama.cpp.git "$src.tmp"
  mv "$src.tmp" "$src"
fi
head=$(git -C "$src" rev-parse HEAD)
if [[ $head != "$LLAMA_SHA" ]]; then
  echo "error: $src is at $head, expected $LLAMA_SHA ($LLAMA_TAG); delete it and rerun" >&2
  exit 1
fi

if [[ ! -f $build/CMakeCache.txt ]]; then
  cmake -S "$here" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$src" -DLLAMA_TAG="$LLAMA_TAG" \
    > "$build.configure.log" 2>&1 || { cat "$build.configure.log" >&2; exit 1; }
fi
cmake --build "$build" --target ref-dump llama-simple llama-tokenize >&2

echo "llama.cpp:      $src ($LLAMA_TAG)"
echo "ref-dump:       $build/ref-dump"
echo "llama-simple:   $build/bin/llama-simple"
echo "llama-tokenize: $build/bin/llama-tokenize"
