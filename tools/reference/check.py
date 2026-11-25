"""Adds sha256 to a ref-dump manifest, checks every file, and compares it with the committed one.

usage: python3 -I check.py <dump-dir> <committed-manifest.json> [--update]
Exit 0 when the dump matches the committed manifest (or --update wrote it), 1 otherwise.
"""
import array
import hashlib
import json
import math
import sys
from pathlib import Path


def main() -> int:
    dump, committed = Path(sys.argv[1]), Path(sys.argv[2])
    update = "--update" in sys.argv[3:]
    manifest = json.loads((dump / "manifest.json").read_text())

    bad = 0
    for t in manifest["tensors"]:
        data = (dump / t["file"]).read_bytes()
        t["sha256"] = hashlib.sha256(data).hexdigest()
        n = math.prod(t["shape"])
        floats = array.array("f", data)  # little-endian host, as ref-dump asserts
        if len(data) != 4 * n:
            print(f"size mismatch: {t['file']} has {len(data)} bytes, shape {t['shape']} needs {4 * n}")
            bad += 1
        elif not all(math.isfinite(v) for v in floats):
            print(f"non-finite value in {t['file']}")
            bad += 1
    if bad:
        return 1

    # Same layout as ref-dump writes: header fields one per line, then one tensor per line.
    head = {k: v for k, v in manifest.items() if k != "tensors"}
    lines = ["{"] + [f"{json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()] + ['"tensors": [']
    lines += [json.dumps(t) + ("," if i + 1 < len(manifest["tensors"]) else "") for i, t in enumerate(manifest["tensors"])]
    text = "\n".join(lines + ["]", "}"]) + "\n"
    (dump / "manifest.json").write_text(text)
    print(f"tensors: {len(manifest['tensors'])}, all sizes match their shapes, all values finite")

    if update or not committed.exists():
        committed.write_text(text)
        print(f"wrote {committed}")
        return 0
    want = json.loads(committed.read_text())
    if want == manifest:
        print(f"checksums: all {len(manifest['tensors'])} match {committed}")
        return 0
    for k in sorted(set(head) | set(want) - {"tensors"}):
        if head.get(k) != want.get(k):
            print(f"differs: {k}: got {head.get(k)!r}, committed {want.get(k)!r}")
    got_t = {t["file"]: t for t in manifest["tensors"]}
    want_t = {t["file"]: t for t in want.get("tensors", [])}
    diff = sorted(f for f in got_t.keys() | want_t.keys() if got_t.get(f) != want_t.get(f))
    for f in diff[:20]:
        print(f"differs: {f}")
    print(f"checksums: {len(diff)} of {len(want_t)} tensors differ from {committed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
