"""Writes hostile or edge-case GGUF files for driving gguf2hpp.

Usage: python -I craft_gguf.py <kind> <out.gguf>
Kinds: evil-strings, bad-key, digit-key, nonfinite, zero-tensors.

evil-strings also writes <out>.check.hpp: static_asserts that the generated header holds
exactly these bytes. The expected bytes are spelled as char arrays, not as escaped literals,
so the check does not reuse the emitter's escaping.
"""

import math
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

EVIL = ['q"uote', "back\\slash", "new\nline", "end*/comment", "tri??=graph", "nul\0byte", "\\x41B"]


def char_array(name: str, s: str) -> str:
    data = s.encode()
    body = ", ".join(str(b if b < 128 else b - 256) for b in data) or "0"
    return f"inline constexpr char {name}[] = {{{body}}};"


def main() -> None:
    kind, out = sys.argv[1], Path(sys.argv[2])
    w = GGUFWriter(out, "evil")
    check = []
    if kind == "evil-strings":
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
    w.write_tensors_to_file()
    w.close()
    if check:
        Path(f"{out}.check.hpp").write_text("#include <string_view>\n" + "\n".join(check) + "\n")
    print(out)


if __name__ == "__main__":
    main()
