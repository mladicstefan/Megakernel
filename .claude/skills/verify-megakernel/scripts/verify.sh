#!/usr/bin/env bash
# Drives gguf2hpp and the engine build the way a user does, and keeps the evidence.
# Every call needs RUN_ID set, because each shell call starts fresh:
#   RUN_ID=<id> verify.sh <command> [args]
# Commands: launch | doctor | gen | compile | craft | engine | cleanup (see SKILL.md).
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
HERE=$(cd "$(dirname "$0")" && pwd)
: "${RUN_ID:?set RUN_ID, e.g. RUN_ID=\$(date +%Y%m%d-%H%M%S)}"
SCRATCH=${VERIFY_SCRATCH:-${TMPDIR:-/tmp}/megakernel-verify/$RUN_ID} # removed by cleanup
EVIDENCE=$REPO/build/verify-evidence/$RUN_ID                         # kept: build/ is gitignored
BUILD=$SCRATCH/build
NVCC=${NVCC:-/opt/cuda/bin/nvcc}
PY=$REPO/tests/data/.venv/bin/python

# Reuse sources an earlier configure of build/ fetched, so launch works offline.
fetch_flags() {
    local d=$REPO/build/_deps
    [[ -d $d/googletest-src ]] && echo "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$d/googletest-src"
    [[ -d $d/xxhash-src ]] && echo "-DFETCHCONTENT_SOURCE_DIR_XXHASH=$d/xxhash-src"
    true
}

tree_rev() { # HEAD plus a hash of uncommitted changes: a binary built from another tree is stale
    echo "$(git -C "$REPO" rev-parse HEAD) $(git -C "$REPO" diff HEAD | sha1sum | cut -c1-12)"
}

cmd=${1:-}
shift || true
mkdir -p "$EVIDENCE"

case $cmd in
launch)
    mkdir -p "$SCRATCH"
    # shellcheck disable=SC2046
    cmake -S "$REPO" -B "$BUILD" $(fetch_flags) >"$EVIDENCE/launch.log" 2>&1
    cmake --build "$BUILD" -j --target gguf2hpp >>"$EVIDENCE/launch.log" 2>&1
    tree_rev >"$SCRATCH/tree-rev"
    echo "launched: $BUILD/gguf2hpp (log: $EVIDENCE/launch.log)"
    ;;
doctor)
    ok=1
    if [[ -x $BUILD/gguf2hpp ]]; then echo "ok   binary $BUILD/gguf2hpp"; else echo "FAIL no binary: run launch"; ok=0; fi
    if [[ -f $SCRATCH/tree-rev && $(cat "$SCRATCH/tree-rev") == "$(tree_rev)" ]]; then
        echo "ok   built from the current tree"
    else
        echo "FAIL tree changed since launch (or never launched): rerun launch"; ok=0
    fi
    if [[ -x $BUILD/gguf2hpp ]]; then
        set +e; usage=$("$BUILD/gguf2hpp" 2>&1); code=$?; set -e
        if [[ $code == 2 && $usage == usage:* ]]; then echo "ok   no-arg run prints usage, exit 2"; else echo "FAIL no-arg run: exit $code, '$usage'"; ok=0; fi
    fi
    size=$(stat -c %s "$REPO/tests/data/toy.gguf" 2>/dev/null || echo missing)
    if [[ $size == 384 ]]; then echo "ok   toy.gguf 384 bytes"; else echo "FAIL toy.gguf: $size"; ok=0; fi
    if [[ -f $REPO/models/Qwen_Qwen3-0.6B-bf16.gguf ]]; then echo "ok   Qwen model present"; else echo "info Qwen model missing: Qwen recipes report a skip"; fi
    command -v g++ >/dev/null && echo "ok   g++ $(g++ -dumpversion)" || { echo "FAIL no g++"; ok=0; }
    if [[ -x $NVCC ]]; then echo "ok   nvcc at $NVCC"; else echo "info no nvcc: CUDA compile checks report a skip"; fi
    [[ -x $PY ]] && echo "ok   gguf-py venv" || echo "info no tests/data/.venv: craft unavailable"
    echo "evidence: $EVIDENCE"
    [[ $ok == 1 ]]
    ;;
gen) # gen <in.gguf> <name>: run the tool as a user would; record command, stderr, exit, output
    in=$1 name=$2 out=$EVIDENCE/$2.hpp
    echo "$BUILD/gguf2hpp $in $out" >"$EVIDENCE/$name.cmd"
    set +e; "$BUILD/gguf2hpp" "$in" "$out" 2>"$EVIDENCE/$name.stderr"; code=$?; set -e
    echo "$code" >"$EVIDENCE/$name.exit"
    left=$([[ -e $out.tmp ]] && echo "yes" || echo "no")
    echo "exit=$code output=$([[ -f $out ]] && echo "$out ($(stat -c %s "$out") bytes)" || echo none) tmp-left=$left"
    [[ -s $EVIDENCE/$name.stderr ]] && echo "stderr: $(cat "$EVIDENCE/$name.stderr")"
    true
    ;;
compile) # compile <header.hpp> <name> [check.cpp]: does the header build, as C++23 and as CUDA C++20?
    hpp=$(realpath "$1") name=$2 check=${3:+$(realpath "$3")}
    mkdir -p "$SCRATCH/probe-$name"
    tu=$SCRATCH/probe-$name/probe.cpp
    {
        echo "#include \"$hpp\""
        [[ -n $check ]] && echo "#include \"$check\""
        echo '#include <cstdio>'
        if grep -q 'std::array<mk::Tensor, 0>' "$hpp"; then
            echo 'int main() { std::printf("tensors=%zu\n", model::tensors.size()); }'
        else
            # mk::find is consteval: this line is the compile-time lookup a kernel would do.
            echo 'constexpr mk::Tensor first = mk::find(model::tensors, model::tensors[0].name);'
            echo 'int main() { std::printf("tensors=%zu first=%.*s bytes=%llu\n", model::tensors.size(), int(first.name.size()), first.name.data(), static_cast<unsigned long long>(first.bytes)); }'
        fi
    } >"$tu"
    set +e
    g++ -std=c++23 -Wall -Wextra -Wpedantic -Werror -I"$REPO/src" "$tu" -o "$SCRATCH/probe-$name/probe" >"$EVIDENCE/$name.gcc.log" 2>&1
    gcc_code=$?
    [[ $gcc_code == 0 ]] && "$SCRATCH/probe-$name/probe" >"$EVIDENCE/$name.probe.stdout" 2>&1
    if [[ -x $NVCC ]]; then
        cp "$tu" "$SCRATCH/probe-$name/probe.cu"
        "$NVCC" -std=c++20 -I"$REPO/src" -c "$SCRATCH/probe-$name/probe.cu" -o "$SCRATCH/probe-$name/probe.o" >"$EVIDENCE/$name.nvcc.log" 2>&1
        nvcc_code=$?
    else
        nvcc_code=skipped
    fi
    set -e
    echo "g++=$gcc_code nvcc=$nvcc_code logs=$EVIDENCE/$name.{gcc,nvcc}.log"
    [[ -f $EVIDENCE/$name.probe.stdout ]] && echo "probe: $(cat "$EVIDENCE/$name.probe.stdout")"
    true
    ;;
craft) # craft <kind> <name>: write a hostile or edge-case GGUF into scratch, print its path
    mkdir -p "$SCRATCH/craft"
    "$PY" -I "$HERE/craft_gguf.py" "$1" "$SCRATCH/craft/$2.gguf"
    ;;
engine) # engine <in.gguf>: configure the engine for a model, build it, run it
    gguf=$(realpath "$1")
    eb=$SCRATCH/engine-build
    # shellcheck disable=SC2046
    cmake -S "$REPO" -B "$eb" -DMODEL_GGUF="$gguf" $(fetch_flags) >"$EVIDENCE/engine.configure.log" 2>&1
    cmake --build "$eb" -j --target engine >"$EVIDENCE/engine.build.log" 2>&1
    "$eb/engine" >"$EVIDENCE/engine.stdout" 2>&1
    echo "engine-build=$eb generated=$eb/generated/model.hpp"
    grep -E 'Generating model.hpp' "$EVIDENCE/engine.build.log" || echo "(no generation line in this build)"
    echo "stdout: $(cat "$EVIDENCE/engine.stdout")"
    ;;
cleanup)
    rm -rf "$SCRATCH"
    echo "removed $SCRATCH; evidence kept in $EVIDENCE"
    ;;
*)
    sed -n '2,5p' "$0" >&2
    exit 2
    ;;
esac
