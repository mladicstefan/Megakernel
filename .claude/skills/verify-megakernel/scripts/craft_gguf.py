"""Writes hostile or edge-case GGUF files for driving gguf2hpp.

Usage: python -I craft_gguf.py <kind> <out.gguf> [args]
Kinds: evil-strings, bad-key, digit-key, nonfinite, zero-tensors, qwen3.

evil-strings also writes <out>.check.hpp: static_asserts that the generated header holds
exactly these bytes. The expected bytes are spelled as char arrays, not as escaped literals,
so the check does not reuse the emitter's escaping.

qwen3 takes <embd> <ff> <heads> <kv_heads> <head_dim> <layers> <vocab> [align=<n>] [pad8]
[ctx=<n>] [arch-last] [sparse] and writes a qwen3 model of that shape with random weights,
for building the engine for it. align=<n> sets general.alignment, pad8 puts an 8-byte tensor
before the others, ctx=<n> sets qwen3.context_length (default 40960), arch-last writes
general.architecture after every other key, and sparse writes no tensor data: the file is
extended to its size, reads as zeros, and is only for building. It also writes
<out>.tokens.txt: 14 token IDs inside the vocabulary.
"""

import math
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import quantize

EVIL = ['q"uote', "back\\slash", "new\nline", "end*/comment", "tri??=graph", "nul\0byte", "\\x41B"]


def char_array(name: str, s: str) -> str:
    data = s.encode()
    body = ", ".join(str(b if b < 128 else b - 256) for b in data) or "0"
    return f"inline constexpr char {name}[] = {{{body}}};"


# Adds the keys and tensors the engine reads for a qwen3 model, and returns a prompt.
def qwen3(w: GGUFWriter, args: list[str]) -> str:
    embd, ff, heads, kv_heads, head_dim, layers, vocab = map(int, args[:7])
    options = args[7:]
    rng = np.random.default_rng(0)
    ctx = 40960
    sparse = "sparse" in options
    for o in options:
        if o.startswith("align="):
            w.add_custom_alignment(int(o.removeprefix("align=")))
        elif o.startswith("ctx="):
            ctx = int(o.removeprefix("ctx="))
        elif o not in ("pad8", "arch-last", "sparse"):
            sys.exit(f"unknown qwen3 option {o}")
    w.add_uint32("qwen3.block_count", layers)
    w.add_uint32("qwen3.context_length", ctx)
    w.add_uint32("qwen3.embedding_length", embd)
    w.add_uint32("qwen3.feed_forward_length", ff)
    w.add_uint32("qwen3.attention.head_count", heads)
    w.add_uint32("qwen3.attention.head_count_kv", kv_heads)
    w.add_uint32("qwen3.attention.key_length", head_dim)
    w.add_uint32("qwen3.attention.value_length", head_dim)
    w.add_float32("qwen3.rope.freq_base", 1e6)
    w.add_float32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
    if "arch-last" in options:
        kv = w.kv_data[0]
        kv["general.architecture"] = kv.pop("general.architecture")

    def matrix(name: str, cols: int, rows: int, scale: float) -> None:
        if sparse:
            w.add_tensor_info(name, (rows, cols), np.dtype(np.uint16), rows * cols * 2,
                              raw_dtype=GGMLQuantizationType.BF16)
            return
        a = rng.standard_normal((rows, cols), dtype=np.float32) * scale
        w.add_tensor(
            name, quantize(a, GGMLQuantizationType.BF16), raw_dtype=GGMLQuantizationType.BF16
        )

    def norm(name: str, n: int) -> None:
        if sparse:
            w.add_tensor_info(name, (n,), np.dtype(np.float32), n * 4)
            return
        w.add_tensor(name, (1 + 0.1 * rng.standard_normal(n)).astype(np.float32))

    if "pad8" in options:
        w.add_tensor("pad8", np.zeros(1, dtype=np.float64))
    q, kv = heads * head_dim, kv_heads * head_dim
    matrix("token_embd.weight", embd, vocab, 1.0)
    for layer in range(layers):
        p = f"blk.{layer}."
        norm(p + "attn_norm.weight", embd)
        matrix(p + "attn_q.weight", embd, q, embd**-0.5)
        matrix(p + "attn_k.weight", embd, kv, embd**-0.5)
        matrix(p + "attn_v.weight", embd, kv, embd**-0.5)
        norm(p + "attn_q_norm.weight", head_dim)
        norm(p + "attn_k_norm.weight", head_dim)
        matrix(p + "attn_output.weight", q, embd, q**-0.5)
        norm(p + "ffn_norm.weight", embd)
        matrix(p + "ffn_gate.weight", embd, ff, embd**-0.5)
        matrix(p + "ffn_up.weight", embd, ff, embd**-0.5)
        matrix(p + "ffn_down.weight", ff, embd, ff**-0.5)
    norm("output_norm.weight", embd)
    matrix("output.weight", embd, vocab, embd**-0.5)
    return str(rng.integers(0, vocab, 14).tolist()) + "\n"


def main() -> None:
    kind, out = sys.argv[1], Path(sys.argv[2])
    w = GGUFWriter(out, "qwen3" if kind == "qwen3" else "evil")
    check = []
    tokens = ""
    if kind == "qwen3":
        tokens = qwen3(w, sys.argv[3:])
    elif kind == "evil-strings":
        for i, s in enumerate(EVIL):
            w.add_tensor(s, np.zeros(1, dtype=np.float32))
            check.append(char_array(f"want{i}", s))
            check.append(
                f"static_assert(model::tensors[{i}].name == std::string_view(want{i}, {len(s.encode())}));"
            )
        w.add_string("evil.text", EVIL[0] + EVIL[2])
    elif kind == "bad-key":
        w.add_uint32("evil.b-c", 1)
    elif kind == "digit-key":
        w.add_uint32("evil.1a", 1)
    elif kind == "nonfinite":
        w.add_float32("evil.scale", math.nan)
    elif kind == "zero-tensors":
        w.add_uint32("evil.block_count", 0)
    else:
        sys.exit(f"unknown kind {kind}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    if kind == "qwen3" and "sparse" in sys.argv[3:]:
        w.write_ti_data_to_file()
        f, pad = w.fout[0], GGUFWriter.ggml_pad
        f.truncate(pad(f.tell(), w.data_alignment) +
                   sum(pad(t.nbytes, w.data_alignment) for t in w.tensors[0].values()))
    else:
        w.write_tensors_to_file()
    w.close()
    if check:
        Path(f"{out}.check.hpp").write_text("#include <string_view>\n" + "\n".join(check) + "\n")
    if tokens:
        Path(f"{out}.tokens.txt").write_text(tokens)
    print(out)


if __name__ == "__main__":
    main()
