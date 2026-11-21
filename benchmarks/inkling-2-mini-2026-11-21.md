# Inkling 2 Mini on llama.cpp, 2026-11-21

Every number here was measured while `gemma-4-31b` was answering one or two
requests on the same GPU, which was in use and not mine to stop. So they
compare builds under the same load; none is the speed of an idle machine.
Weights: `Inkling-2-Mini-IQ4_NL.gguf`, all layers on the GPU, steps of 512
tokens, flash attention on.

`llama-bench`, tokens/s, mean ± deviation over three runs:

| Build | reads 512 | writes 96 | reads 512 behind 8k | writes 96 behind 8k |
|---|---:|---:|---:|---:|
| upstream llama.cpp (build 12616), ROCm | 329 ± 3 | 9.9 ± 3.6 | 168 ± 1 | 10.9 ± 0.9 |
| fused glue kernels, ROCm (the one in use) | 494 ± 1 | 19.3 ± 0.2 | 427 ± 13 | 13.3 ± 0.7 |
| fused build on Vulkan, with ggml-org/llama.cpp#32735 | 274 ± 265 | 13.9 ± 1.3 | 265 ± 1 | 14.9 ± 0.1 |

"Fused glue kernels" is realugbun's `inkling2-speed` branch merged into
upstream; `Containerfile.llama-cpp` pins both. Its 82 `INKLING2_*` cases of
`test-backend-ops` pass on this GPU against the CPU results.

Through the server, one 60,035-token prompt: 880 s before (68 tokens/s),
251 s after (239 tokens/s). Memory is unchanged, 26.5 GB at the peak. A
follow-up turn on that prompt reused 60,031 cached tokens and read the new 20
in 1.9 s.

Tried and left out:

- **DSpark speculative decoding** (ggml-org/llama.cpp#32218 with Unsloth's
  drafter file): 12–40% of the drafted tokens were accepted, at 3 or 7 drafted
  tokens, greedy or sampled. Generation was no faster and sometimes slower.
- **A q8_0 KV cache, 4 threads, four GPU queues**: nothing beyond the noise.
- **Flash attention off**: 2.5x slower; it stays on.
- **ggml-org/llama.cpp#32752 and #32740** (less working memory): both conflict
  with the fused kernels in the same graph code. Worth another look once
  either side is merged upstream.
