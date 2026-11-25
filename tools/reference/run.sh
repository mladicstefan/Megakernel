#!/usr/bin/env bash
# Dumps llama.cpp's per-layer tensors for one committed prompt and checks them against the fixtures.
# usage: run.sh [prompt-id] [--update]   (prompt-id defaults to planets; --update rewrites the fixtures)
# MODEL overrides the GGUF path. Exit 0 only when every check passes.
set -euo pipefail

THREADS=8     # the dumps are byte-identical at 1, 4, 8 and 16 threads; 8 is the physical core count
GENERATE=32

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
id=planets update=
for a in "$@"; do
  case $a in --update) update=--update ;; *) id=$a ;; esac
done
fixtures=$here/../../tests/data/reference/$id
# models/ is gitignored, so a worktree finds it in the main checkout, next to the common git dir.
main=$(cd "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
model=${MODEL:-$main/models/Qwen_Qwen3-0.6B-bf16.gguf}
[[ -f $fixtures/prompt.txt ]] || { echo "error: no $fixtures/prompt.txt" >&2; exit 1; }
[[ -f $model ]] || { echo "error: no model at $model (set MODEL)" >&2; exit 1; }
prompt=$(<"$fixtures/prompt.txt")  # drops a trailing newline, which would be a token of its own

paths=$("$here/build.sh")
path() { awk -v k="$1:" '$1 == k { print $2 }' <<<"$paths"; }
tag=$(awk '$1 == "llama.cpp:" { gsub(/[()]/, "", $3); print $3 }' <<<"$paths")
out=${XDG_CACHE_HOME:-$HOME/.cache}/megakernel/reference/$tag/$id
fail=0

# 1. The committed token IDs are what llama.cpp's tokenizer gives for the prompt (no BOS, no template).
tokenize() { "$(path llama-tokenize)" -m "$model" -p "$1" --ids --log-disable 2>/dev/null; }
tokens=$(tokenize "$prompt")
if [[ -n $update ]]; then printf '%s\n' "$tokens" > "$fixtures/tokens.txt"; fi
if [[ $tokens == "$(cat "$fixtures/tokens.txt")" ]]; then
  echo "tokens: match tokens.txt ($tokens)"
else
  echo "tokens: MISMATCH: llama-tokenize gives $tokens, tokens.txt has $(cat "$fixtures/tokens.txt")"; fail=1
fi

# 2. The dump, written next to its final place and renamed in, so a crash leaves no half dump.
rm -rf "$out.tmp"
mkdir -p "$out.tmp"
start=$(date +%s%N)
"$(path ref-dump)" "$model" "$fixtures/tokens.txt" "$out.tmp" --threads $THREADS --generate $GENERATE \
  > "$out.tmp/ref-dump.stdout" 2> "$out.tmp/ref-dump.stderr" || { cat "$out.tmp/ref-dump.stderr" >&2; exit 1; }
echo "dump: $(( ($(date +%s%N) - start) / 1000000 )) ms, $(grep '^tensors:' "$out.tmp/ref-dump.stdout")"
python3 -I "$here/check.py" "$out.tmp" "$fixtures/manifest.json" $update || fail=1
rm -rf "$out"
mv "$out.tmp" "$out"

# 3. Greedy. llama-simple prints text, not IDs, so its output is re-tokenized; greedy.txt holds
# those IDs after the prompt, and the dump's argmax chain must equal them and decode to the same text.
simple=$("$(path llama-simple)" -m "$model" -n $GENERATE "$prompt" 2>/dev/null)
simple_ids=$(tokenize "$simple")
expect="[${simple_ids#"${tokens%]}, "}"
if [[ -n $update ]]; then printf '%s\n' "$expect" > "$fixtures/greedy.txt"; fi
generated=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["generated"])' "$out/manifest.json")
text=$(sed -n 's/^text: //p' "$out/ref-dump.stdout")
echo "greedy: llama-simple: $expect"
echo "greedy: ref-dump:     $generated"
if [[ $expect == "$(cat "$fixtures/greedy.txt")" && $generated == "$expect" && $simple == "$prompt$text" ]]; then
  echo "greedy: match greedy.txt ($GENERATE tokens), same text: $text"
else
  echo "greedy: MISMATCH (greedy.txt: $(cat "$fixtures/greedy.txt"))"
  echo "        llama-simple text: ${simple#"$prompt"}"
  echo "        ref-dump text:     $text"
  fail=1
fi

echo "dump dir: $out ($(command du -sh "$out" | cut -f1))"
exit $fail
