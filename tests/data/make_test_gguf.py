from pathlib import Path

import numpy as np
from gguf import GGUFWriter

OUT = Path(__file__).parent / "test.gguf"


def main() -> None:
    # The second argument becomes the general.architecture key.
    writer = GGUFWriter(OUT, "test")

    # One key per value kind the reader has to handle early on.
    writer.add_uint32("toy.block_count", 2)
    writer.add_float32("toy.scale", 0.5)
    writer.add_string("toy.note", "hello")
    writer.add_bool("toy.flag", True)
    writer.add_array("toy.list", [10, 20, 30])

    # Two tensors with different shapes and dtypes, filled with 1, 2, 3...
    # numpy shape (2, 3) is stored in the file as dims [3, 2] (ggml order).
    writer.add_tensor("a", np.arange(1, 7, dtype=np.float32).reshape(2, 3))
    writer.add_tensor("b", np.arange(1, 5, dtype=np.float16))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
