# Six model additions, 2026-11-16

Picked from third-party rankings only, read on 2026-11-16: Artificial
Analysis' [small (4–40B)](https://artificialanalysis.ai/models/open-source/small)
and [tiny (≤4B)](https://artificialanalysis.ai/models/open-source/tiny)
open-weights pages, plus Mistral Small 3.2 on request.

| Model | Source | Why |
|---|---|---|
| `nanbeige-4.3-33b` | Nanbeige4.3-33B, official INT4 | First on AA's small ranking (Progress Index 306, Qwen3.8 27B 303); Intelligence Index 20, the same as Qwen3.8 27B (xhigh) |
| `mistral-small-3.2-24b` | Mistral Small 3.2 24B Instruct 2506, Intel AutoRound INT4 | Asked for; dense 24B that sees pictures |
| `ministral-3-3b` | Ministral 3 3B Instruct 2512, cyankiwi INT4 | Mistral's edge model; sees pictures |
| `nemotron-3-nano-4b` | NVIDIA Nemotron 3 Nano 4B, BF16 | AA tiny ranking (176) |
| `granite-4.2-3b` | IBM Granite 4.2 3B, BF16 | AA tiny ranking (191) |
| `minicpm5-2b` | MiniCPM5 2B, cyankiwi INT4 | Second on AA's tiny ranking (221) |

Not added: K2 Horizon 3.7B (first on the tiny ranking, 242) needs a newer vLLM
than this image (no `K2HorizonForCausalLM`, no `k2_horizon` parsers). Nemotron
Nano 4B and Granite 3B have no trusted INT4 build, so they run BF16 and decode
at about a third of the speed an INT4 copy would allow.

## Results

Same suite as [all nine models](all-models-2026-11-14.md), one model at a
time, nothing else generating. Raw reports are in
`logs/small-models-2026-11-16/`. Tokens per second.

| Model | Decode, short chat | 4 at once | 16 at once | Read, 24k prompt | Decode behind 24k | Checks passed |
|---|---:|---:|---:|---:|---:|---|
| `minicpm5-2b` | 101.5 | 334.8 | 375.6 | 1,755 | 65.1 | 9/11; failed: reasoning, prefill |
| `ministral-3-3b` | 74.9 | 248.8 | 420.7 | 1,335 | 36.9 | 7/10; failed: chat, tool_none, deep_lookup |
| `granite-4.2-3b` | 32.1 | 125.4 | 422.1 | 778 | 24.7 | 10/11; failed: deep_lookup |
| `nemotron-3-nano-4b` | 30.9 | 110.1 | 321.9 | 3,542 | 29.0 | 11/11 |
| `mistral-small-3.2-24b` | 15.3 | 36.8 | 102.0 | 340 | 11.6 | 8/10; failed: prefill, deep_lookup |
| `nanbeige-4.3-33b` | 9.8 | 12.1 | 19.5 | 140 | 7.6 | 11/11 |

The remaining failures are the models' own answers, not serving faults:

- MiniCPM5 used its whole 1,024-token budget thinking about 17 × 23 − 5 and
  gave a wrong code in the 4k lookup.
- Ministral and its tool-free turn answer "Ready." when asked for exactly
  READY, and give a wrong code in the 24k lookup.
- Mistral Small finds the right code in the 4k lookup but wraps it in a
  sentence; it misses the 24k lookup, like most of the existing models.

Both Mistral models read a test picture correctly (the number 4711, a red
square on the left, a blue circle on the right). Ministral does that at about
74 tokens/s.

Nanbeige answers everything, but slowly in a batch. Alone it is near the
memory-bandwidth limit for 21 GB of weights. Its weights are asymmetric INT4
(they carry zero points), and the measured fast kernels in
`local_llm_dense.py` cover only symmetric weights in known shapes; that is the
place to look if Nanbeige should serve several requests at once.

## What broke on the way, and the protection each got

1. MiniCPM5's tool parser is `minicpm5` in this vLLM, not `minicpm5xml`.
   vLLM started and then crash-looped. `./llm up` now reads the image's
   parser names and refuses an unknown one.
2. vLLM 0.29 imports Transformers 4's Pixtral image rotation, which
   Transformers 5 changed. `vllm_plugins/pixtral-positions.patch` uses the
   new one with (row, column) positions; a build check proves the angles are
   identical to Transformers 4's. Image label `local-llm.pixtral-positions=1`.
3. With `tekken.json` present, Transformers 5 loads Mistral's own tokenizer,
   which treats `[IMG]` as text, and the server crashes on its first picture.
   Downloads skip the file (the standard tokenizer gives identical ids, checked
   against Mistral's own), and a leftover copy is refused before start.
4. Forcing a tool call (`required` or a named tool) on either Mistral model
   came back as JSON text. `local_llm_tools.py` now forces their own
   `[TOOL_CALLS]name[ARGS]{...}` with the tool's schema; 5/5 tool checks pass.
5. Nanbeige's weights fit its first share but its 32k KV cache did not.
   `./llm up` now adds the KV cache from the checkpoint config, plus 1.5 GB,
   and refuses up front. That margin admits every start that worked and
   refuses the one that failed.

Images: `localhost/local-llm:before-small-models-20261116` is the image before
this work. `small-models-20261116` added the Pixtral patch, and
`small-models-tools-20261116` (now `latest`) adds the forced-tool grammar.
Each was checked to carry otherwise identical plugin files.
