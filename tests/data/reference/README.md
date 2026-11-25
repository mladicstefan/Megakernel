# llama.cpp reference fixtures

One directory per prompt. `tools/reference/run.sh <prompt-id>` regenerates the full dump in
`~/.cache/megakernel/reference/<tag>/<prompt-id>/` and checks it against these files.

- `prompt.txt`: the raw prompt, with no chat template and no trailing newline. Qwen3's
  `tokenizer.ggml.add_bos_token` is false, so there is no BOS either.
- `tokens.txt`: llama.cpp's tokenizer output for the prompt, from
  `llama-tokenize -m Qwen_Qwen3-0.6B-bf16.gguf -p "$(<prompt.txt)" --ids --log-disable`
  at llama.cpp `b12200`. `run.sh` reruns this command and compares.
- `manifest.json`: every dumped tensor's name, layer (-1 for `embd`, `result_norm` and
  `result_output`), position, shape in ggml order (ne0 first), dtype, file and sha256,
  plus the run settings (tag, threads, KV types, flash attention). It has no tensor data.
- `greedy.txt`: the 32 token IDs that `llama-simple -n 32` generates greedily after the
  prompt, re-tokenized from its text output.

`run.sh <prompt-id> --update` rewrites `tokens.txt`, `manifest.json` and `greedy.txt`.
