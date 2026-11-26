#!/usr/bin/env bash
# Drives gguf2hpp and the engine build the way a user does, and keeps the evidence.
# Every call needs RUN_ID set, because each shell call starts fresh:
#   RUN_ID=<id> verify.sh <command> [args]
# Commands: launch | doctor | gen | compile | craft | engine | run-engine | cleanup (see SKILL.md).
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
craft) # craft <kind> <name> [args]: write a hostile or edge-case GGUF into scratch, print its path
    mkdir -p "$SCRATCH/craft"
    "$PY" -I "$HERE/craft_gguf.py" "$1" "$SCRATCH/craft/$2.gguf" "${@:3}"
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
run-engine) # run-engine <name> [args]: run the engine that `engine` built; record command, stdout, stderr, exit
    name=$1
    shift
    eb=$SCRATCH/engine-build
    echo "$eb/engine $*" >"$EVIDENCE/$name.cmd"
    set +e; "$eb/engine" "$@" >"$EVIDENCE/$name.stdout" 2>"$EVIDENCE/$name.stderr"; code=$?; set -e
    echo "$code" >"$EVIDENCE/$name.exit"
    echo "exit=$code"
    [[ -s $EVIDENCE/$name.stdout ]] && echo "stdout: $(cat "$EVIDENCE/$name.stdout")"
    [[ -s $EVIDENCE/$name.stderr ]] && echo "stderr: $(cat "$EVIDENCE/$name.stderr")"
    true
    ;;
gate) # gate: the megakernel's size on sm_89 and its speed, for the engine that `engine` built
    eb=$SCRATCH/engine-build g=$SCRATCH/gate cuda=$(dirname "$NVCC") P=$REPO/tests/data/reference/planets
    obj=$eb/CMakeFiles/forward_pass.dir/src/megakernel/megakernel.cu.o
    [[ -f $obj ]] || { echo "FAIL no $obj: run engine with the Qwen model first"; exit 1; }
    cmake --build "$eb" -j --target forward >"$EVIDENCE/gate.build.log" 2>&1
    rm -rf "$g" && mkdir -p "$g"
    sass() { "$cuda/cuobjdump" -sass "$@" | awk '/Function :/ { k = /megakernel/ } k && /^ +\/\*[0-9a-f]{4,}\*\/ +[^ \/].*;/'; }
    # ptxas reruns on the PTX the build embedded; equal SASS below proves it is the build's compile.
    (cd "$g" && "$cuda/cuobjdump" -xptx all "$obj" >/dev/null)
    "$cuda/ptxas" -arch=sm_89 -v "$g"/*.sm_89.ptx -o "$g/k.cubin" >"$EVIDENCE/gate.ptxas.log" 2>&1
    sass -arch sm_89 "$obj" >"$EVIDENCE/gate.sm_89.sass"
    sass "$g/k.cubin" | cmp -s - "$EVIDENCE/gate.sm_89.sass" || { echo "FAIL ptxas rerun differs from the build's SASS"; exit 1; }
    read -r regs smem stores loads < <(awk '/Compiling entry function/ { k = /megakernel/ }
        k && /spill stores/ { st = $5; ld = $9 } k && /Used [0-9]+ registers/ { r = $5; for (i = 1; i < NF; i++) if ($(i + 1) == "bytes" && $(i + 2) ~ /^smem/) s = $i }
        END { print r, s, st, ld }' "$EVIDENCE/gate.ptxas.log")
    busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null || echo unknown)
    ms=()
    for i in 1 2 3; do
        out=$EVIDENCE/gate.run$i.stdout
        "$eb/forward" --megakernel --generate 128 "$P/tokens.txt" >"$out" 2>&1 || { echo "FAIL run $i exited $?: $out"; exit 1; }
        # A fast step that decodes the wrong tokens is no measurement.
        grep -qF "generated: $(tr -d ']\n' <"$P/greedy.txt")," "$out" || { echo "FAIL run $i: not the reference's greedy tokens: $out"; exit 1; }
        ms+=("$(sed -n 's/^decode: .*, \([0-9.]*\) ms per step.*/\1/p' "$out")")
    done
    read -r lo mid hi < <(printf '%s\n' "${ms[@]}" | sort -n | paste -sd' ')
    {
        echo "gate: megakernel, sm_89, $(tree_rev)"
        echo "registers  $regs"
        echo "spills     $stores bytes stored, $loads bytes loaded"
        echo "smem       $smem bytes static"
        echo "sass       $(wc -l <"$EVIDENCE/gate.sm_89.sass") instructions"
        echo "ms/step    $mid median of 3 ($lo to $hi), --generate 128, $(grep -o '^decode: [0-9]* steps' "$EVIDENCE/gate.run1.stdout" | cut -d' ' -f2) steps, GPU $busy busy before"
    } >"$EVIDENCE/gate.txt"
    cat "$EVIDENCE/gate.txt"
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
