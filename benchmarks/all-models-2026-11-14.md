# All nine models on one suite, 2026-11-14

Until now each model had its own measurements, taken at different times with
different prompts. This puts all nine through the same cases in one night and
then follows what the numbers point at. Raw reports, startup logs and rejected
runs are in `logs/all-models-2026-11-14/`.

| Model | Decode, short chat | 4 at once | 16 at once | Read, 4k prompt | Read, 24k prompt | Decode behind 24k | Cached 24k first token | Checks passed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3-omni-30b | 64.4 | 146.0 | 341.9 | 1,298 | 701 | 34.9 | 0.44 s | 9/10; failed: deep_lookup |
| nemotron-3.5-lightning-30b-a3b | 64.7 | 143.8 | 332.0 | 1,536 | 1,439 | 61.5 | 0.72 s | 8/11; failed: reasoning, prefill, deep_lookup |
| gemma-4-26b | 40.9 | 130.5 | 366.2 | 1,168 | 982 | 24.1 | 0.17 s | 11/11 |
| gemma-4-12b | 24.6 | 74.2 | 141.9 | 540 | 476 | 15.3 | 0.25 s | 10/11; failed: reasoning |
| muse-glimmer-30b | 14.1 | 44.6 | 115.3 | 376 | 338 | 13.6 | 0.42 s | 9/11; failed: json, json_nonstream |
| granite-4.2-30b | 13.3 | 35.2 | 83.3 | 327 | 258 | 9.0 | 0.16 s | 11/11 |
| qwen3.8-27b | 12.8 | 43.9 | 96.6 | 395 | 318 | 11.2 | 1.84 s | 11/11 |
| olmo-3.1-32b-think | 11.3 | 40.2 | 102.5 | 270 | 210 | 9.8 | 0.36 s | 1/4; failed: json, prefill, deep_lookup |
| gemma-4-31b | 10.3 | 30.8 | 51.4 | 218 | 189 | 6.6 | 0.58 s | 11/11 |

Rates are tokens per second, medians of three runs after one warm-up. Decode
includes thinking tokens. "At once" is the total over all concurrent requests.
"Read" is the server's own prefill time for a prompt that is not in the cache.
"Behind 24k" generates 128 tokens after a 24,000-token prompt that is already
cached, which is how a turn in a long conversation looks to the model.

Every row is the configuration `llm` now starts. Omni's row is the instance
that has been running since 2026-11-10; a copy on the new image measured the
same (64.0, 154.5, 387.7, 1,299, 701, 35.1), so its next restart changes nothing.

The checks are the same for every model: an exact one-word answer, separated
reasoning with a number-only final answer, strict JSON (streamed and not),
five tool-call variants, and a lookup in a 4k and in a 24k prompt.
Omni has no thinking and skips the reasoning check; Olmo skips tools, and
ran only four of its checks this time (the rest are in the
[2026-11-12 report](added-models-2026-11-12.md), where they take minutes each).

## What the table says

**Long conversations are where the time goes.** Omni's own counters since
2026-11-10 (1,333 requests) show the real workload: 30,400 input tokens per
request on average, 85% of them answered from the cache, and 25.5 tokens/s
while generating. The short chat every earlier benchmark used says 64. Behind
24,000 tokens the suite measures 35, and reading drops from 1,298 to 701
tokens/s. Most models lose a third or more of their decode speed at that
depth. Nemotron does not (it has few attention layers): once the context is
long it is the fastest model here, both reading and generating.

**The dense models generate as fast as this memory can be read.** A dense
model reads all its weights for every token. `benchmarks/memory_bandwidth.py`
measures 239 GB/s for this GPU. Granite holds 15.2 GiB of weights and
generates 13.1 tokens/s: 214 GB/s. Gemma 31B, 19.6 GiB at 10.3: 216 GB/s.
No kernel makes one request faster than that. What is left for them is
several requests at once, and reading prompts; both improved below.

**Omni behind a long context is at the same limit.** At 24k it reads 2.4 GB
of attention cache per token on top of its weights. The extra 13 ms per token
is that read at about 180 GB/s. Three attempts to get around it failed (below).

**Some answers are wrong, and the same way in every run.** Omni and Nemotron
return a neighbouring item's code from the 24k inventory. Nemotron and Gemma
12B add words where only a number was asked. Olmo runs out of its 1,024
tokens while still thinking. Muse ignores a strict JSON schema: it wrote
`job_number` where the schema only allows `job`. Its reasoning parser never
hands a plain answer over to the grammar, so for Muse a schema is a request,
not a guarantee.

## What changed

**One GPU queue for every model.** On the image built on 2026-11-12 a loaded
model keeps the GPU at 100% without any request. It was put down to Granite
then; Gemma 12B and 26B did the same, and the suite found no quiet moment to
measure in. With the runtime's default of four queues an idle Gemma 26B held
the machine at about 36 W. With one it idles at 2% GPU and about 15 W, at the
same speed (40.9 tokens/s both ways; 131 and 131 for four at once). `llm` now
sets `GPU_MAX_HW_QUEUES=1` for all models; in 27 starts none idled above 7%. The
idle-GPU warning in `health.py` now names a model that was started without it.

**Reading long prompts, for Granite, Nemotron, Olmo and Muse.** Their
attention heads are 128 wide. The prefill tuning in `prefill-tiles.patch`
only covered 256-wide heads (Qwen), so these four read long prompts with
tiles made for generating. One changed line covers them:

| Reading, tokens/s | 4k before | 4k after | 24k before | 24k after |
|---|---:|---:|---:|---:|
| Granite | 276 | 327 | 119 | 258 |
| Nemotron | 1,353 | 1,536 | 961 | 1,439 |
| Olmo | 255 | 270 | 153 | 210 |
| Muse (on the Triton backend) | 356 | 376 | 262 | 338 |

Granite reads a 24k prompt in 93 s instead of 201 s. Generation is unchanged.
Olmo and Muse gain less: three of four of their layers look at a window, and
those are not covered. (Nemotron's and Olmo's 4k "before" are from the
2026-11-12 report, same inventory.)

**Nemotron: 52 to 63 tokens/s.** Its experts took over half of each step.
The single-token expert kernel that sped up Omni now has a variant for
Nemotron's layout (one projection squared after ReLU, six routes, group 64).
Per expert step it is 1.7x faster with warm caches and 1.35–1.5x after
eviction, on two layers and seeds, and closer to exact arithmetic than the
native path. In serving: short chat 52.0 → 63.0, behind 24k 49.7 → 61.4,
tool call 1.00 → 0.87 s. Several requests at once use
the native path as before (141 and 344 tokens/s both ways).

**Olmo: twice the throughput at 16 requests.** Its kernels were selected for
exactly 4, 128 and 512 tokens per step, the sizes the first benchmark used.
Any other number of concurrent requests ran native. The down projection is
now covered for 2–16 tokens (1.6–2.1x per call) and gate/up for 6–16
(2.5–3.5x). In serving: 16 at once 47.1 → 101.9 tokens/s, 4 at once
33.9 → 40.2, same answers.

**Gemma 31B: its kernels, measured in serving for the first time.** They were
on by default on the strength of component tests. With them: 4 at once 30.6
against 23.8 tokens/s, reading 216 against 179 (4k) and 188 against 166
(24k). The same small-step gate/up kernel as Olmo's was added (3.0x per
call): 16 at once 40.0 → 51.4 tokens/s.

**Muse on the Triton attention backend.** Muse named no backend, so vLLM put
it on its ROCm one. There its window layers keep it off the fast decode
kernel, and behind 24k tokens it generated at half speed. On Triton: 7.5 →
13.6 tokens/s behind 24k, and room for 653,000 instead of 351,000 cached
tokens. With the prefill tuning above a 4k prompt reads 9% slower than on
ROCm (376 against 413) and a 24k prompt 12% faster (338 against 301).

**Benchmarks no longer count as use.** `./llm usage` counted every benchmark
request. The suite now names the model it measures and the monitor leaves
that model out meanwhile. Everything counted for a model other than Omni
before 2026-11-14T22:17Z is benchmark traffic, as are 133 of Omni's requests
(260,528 input and 12,612 output tokens).

## Tried and not kept

| Tried | Result | Why not |
|---|---|---|
| Omni on the Triton attention backend | reads 737 (4k) and 217 (24k) against 1,253 and 687; decode equal | its ROCm backend is the right one |
| Omni with an 8-bit attention cache (needs the Triton backend) | holds 239k tokens instead of 101k; reads 418 and 108; behind 24k 30.1 against 34.7 | slower everywhere; only worth it for capacity |
| Omni with n-gram speculative decoding | short chat 30.0 against 64.1; repeating 20 lines of the prompt, which it should help, 29.4 against 53.7; behind 24k 12.8 against 34.7 | checking several guessed tokens costs an expert model far more than one step |
| Nemotron on the ROCm attention backend | reads 24k at 1,201, but behind 24k 24.9 against 49.7 | the widened tuning reads faster (1,439) without that loss |
| Granite on the ROCm attention backend | reads 330 and 249; decode equal | equal to the widened tuning (327 and 258), which needs no per-model setting |
| Granite small-step kernels | 0.34–1.0x of native at 2–16 tokens | its native group-128 kernels are already the better ones |
| Qwen 27B's own draft head (`--speculative-config`) | short chat 12.8 → 21.7, reasoning answer 8.9 → 4.4 s, tool call 4.0 → 2.3 s; 16 at once 96.6 → 81.7 | every follow-up turn resumes from the cache up to 3,200 tokens early: the first token behind 24k takes 12.8 s instead of 1.8 s. Good for one long answer, bad for a conversation or a tool loop; left as an option |

The first start of Qwen with its draft head failed three times in a row:
speculative decoding changes the scheduler's block size to 1600 tokens, and
the engine refuses `--prefix-cache-retention-interval 1568`, but only after
loading the weights. `./llm status` showed the reason; nothing else was
affected. `./llm up` now refuses that combination before it loads anything.
A schedule that switches the draft head off at five or more concurrent
requests, to avoid the loss at 16, failed the same way for another reason (an
assertion while the engine captures graphs for zero speculative tokens). It
is refused too, so that loss cannot be avoided in this runtime.

## Repeating it

    benchmarks/suite_one.sh gemma-4-31b default logs/suite            # a stopped model
    python3 benchmarks/added_models.py qwen3-omni-30b in-place \
      --case extended --keep-going --output logs/suite/omni.json      # a running one
    python3 benchmarks/suite_table.py logs/suite/*.json

    # one setting, one start
    LOCAL_LLM_EXTRA_FLAGS="--attention-backend ROCM_ATTN" \
      benchmarks/suite_one.sh granite-4.2-30b rocm logs/suite
    LOCAL_LLM_FAST_KERNELS=0 benchmarks/suite_one.sh nemotron-3.5-lightning-30b-a3b native logs/suite
    GPU_MAX_HW_QUEUES=4 ./llm up gemma-4-26b                          # the idle spin

    # on a copy of Omni, next to the one in use
    LOCAL_LLM_COPY=qwen3-omni-30b LOCAL_LLM_EXTRA_FLAGS="--kv-cache-dtype int8_per_token_head --attention-backend TRITON_ATTN" \
      benchmarks/suite_one.sh qwen3-omni-30b-copy int8 logs/suite --case extended --case copy

    # Qwen with its draft head
    LOCAL_LLM_EXTRA_FLAGS='--speculative-config {"method":"mtp","num_speculative_tokens":3} --prefix-cache-retention-interval 1600' \
      ./llm up qwen3.8-27b

Component tests run in the image without a model loaded, under the same lock
as the suite: `benchmarks/nemo_moe_compare.py --vector` (Nemotron's experts),
`benchmarks/dense_checkpoint_compare.py --block-m 16` (Olmo and Granite, real
weights), `benchmarks/dense_compare.py` (Gemma 31B), and
`benchmarks/memory_bandwidth.py`. The exact commands are in the `queue*.sh`
files next to the reports; each result has an `.audit.json`.

The image is `localhost/local-llm:all-models-20261114`,
`c5a02e31848e8577e60750138fd30fe7141354eb33afcd2825ada54961d3586f`, now also
`latest`. The one before is kept as `localhost/local-llm:before-all-models-20261114`.

## Limits

- One run per variant, three measured repetitions each. Differences of 2–3%
  are noise: Nemotron's 16-at-once total came out between 332 and 345.
- Omni was in use during the night. Trials another request overlapped were
  discarded and repeated; my runs will in turn have slowed those requests.
- The prompts are short and synthetic. The checks show whether an interface
  works and whether a lookup lands on the right line, not how good a model is.
- Idle power was sampled for ten seconds per start, and other work on the
  machine shows in some samples. The 36 W and 15 W are the clean ones.
- Loading nine models in turn filled the 8 GB of swap while 34 GB or more of
  RAM stayed available. `./llm status` shows it as a warning.
- Nothing here measures speed at 100k tokens and beyond, where Gemma, Qwen
  and Muse have room to go.
