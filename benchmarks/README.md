# Kernel experiments

[All nine models on one suite](all-models-2026-11-14.md) is the current
comparison: the same cases for every model, what a long conversation costs,
and the changes and rejected experiments that followed from it. The sections
below are the earlier, per-model measurements it builds on.

The [2026-11-12 model-addition report](added-models-2026-11-12.md) records pinned
Granite 4.2, Nemotron 3.5 Lightning and Olmo 3.1 Think checkpoints, compatibility
checks and completed measurements. Granite's retained profile uses a 0.23 share,
32,768-token context, one preprocessing thread and one GPU hardware queue; the
report includes its measured kernel gains, gateway API checks and near-limit
capacity result, plus Nemotron's raw-format limitations and rejected
routing-sensitive profiles. Nemotron retains native kernels with a 0.22 share
to save memory, with near-limit capacity verified and four gateway final-format
failures documented. The three added checkpoints and Omni's hot cache are on
NVMe; five verified legacy warm-model routes now use T7 and the complete cold
Hugging Face archive uses HDD. Verified duplicate pruning freed 198.15 GiB while
preserving the original running Omni cache/process. UUID boot-mount setup still
needs the administrator command recorded in the detailed report.
Olmo's greedy baseline is rejected for thought-only truncation. The final
feature-3 image has 82 passing reasoning/grammar CPU cases and measured selective
dense dispatch. Its retained 0.24-share profile saves 6.6 GiB of admission
reservation, delivers 255.08 uncached prefill tokens/s and 34.85 aggregate tokens/s
at concurrency four. Against a fresh native-process repeat, these are observed
gains of 25.7% and 22.7%; the report preserves the slower initial native run and
compilation/cache-budget differences. Capped text/batch outputs are thought only
with zero useful-answer throughput. Original 1,024-token quality failures remain;
separate 2,048-token JSON and near-limit capacity cases pass. The final gateway
API suite completed: 10/10 applicable requests are protocol-complete, but only
5/10 pass the strict answer checks; one simple Anthropic chat exhausts even
2,048 tokens in thought. Olmo supports no standard tool calls.

Measured on 2026-11-02: Radeon 8060S (`gfx1151`), vLLM `0.29.0+strix`,
PyTorch `2.11.0+rocm10.0.0`. Component experiments ran one at a time in
separate Python processes alongside idle serving processes, without loading
a full model copy. Their timings describe individual operations. The later
serving comparison measures the combined effect in the actual model servers.

The image provides three hooks through `local_llm_kernels.register`. Unknown
shapes and unsupported features use vLLM's implementation. A stopped model
can be started with `LOCAL_LLM_FAST_KERNELS=0 ./llm up NAME` to disable these
three hooks. This is a startup switch; existing graph captures do not change
when the environment changes. Established attention protections and expert
tuning profiles are independent of this switch.

`llm` enables these hooks by default for Omni. Gemma 26B defaults to profiles
only because its full-serving comparison below did not establish a gain.
Explicitly setting `LOCAL_LLM_FAST_KERNELS=1` enables its experimental vector
path when starting a stopped model. Gemma 31B's dense improvements remain
component results; that model was not restarted for this serving comparison.

## Full-model serving

The production Omni and Gemma 26B containers were restarted one at a time,
with profiles only (`LOCAL_LLM_FAST_KERNELS=0`) and with the hooks enabled
(`=1`). Their serving arguments were identical across modes: Omni retained
65,536 context / 0.30 GPU share / 100,816 KV tokens; Gemma retained 262,144
context / 0.25 share / 411,771 KV tokens. Both retained 512-token prefill
chunks. The Gemma comparison used the same image in both modes. Omni's
enabled image additionally contained compiler-safe first-use logging; the
kernel algorithms and tuning profiles were unchanged between those images.

Each case had one warmup and three measured trials; concurrency four yields
twelve measured requests. Temperature was zero. The harness checked both
backends' running/waiting gauges and completion counters before, during and
after each trial, catching short external requests between polls too. All
120 measured requests in the four suites were accepted without contamination.
Ordinary desktop applications continued running, so these are small
workload comparisons rather than isolated-hardware confidence intervals.

| API measurement | Omni profiles | Omni kernels | Gemma profiles | Gemma kernels |
|---|---:|---:|---:|---:|
| Single chat, generation (tokens/s) | 50.96 | 63.52 | 41.18 | 40.63 |
| Four chats, aggregate (tokens/s) | 147.07 | 150.27 | 132.34 | 127.36 |
| Fresh inventory, first token (s) | 3.013 | 3.008 | 3.476 | 3.494 |
| Fresh media, complete response (s) | 1.539 | 1.367 | 0.925 | 0.945 |

The text case generated 256 tokens per request; concurrency four used 128.
The inventory had about 3,700 input tokens. Media used a 2560×1440 synthetic
screenshot, plus a spoken question for Omni; Gemma received the image only.
Every media trial changed input bytes to avoid media-cache hits. File creation
and base64 encoding were outside HTTP timing. Inventory extraction, image/audio
understanding, arithmetic, exact JSON and named tool calls passed in both
modes. Tool parsers buffer tokens, so their client chunks cannot establish a
decode rate; only total latency is compared for tool calls. The older Omni
profile report retains a note marking its tool-rate estimate invalid.

Omni gained 24.6% in single-chat generation and saved 11.2% on this fresh
media response. Its batch-four and inventory-prefill changes were small.
Gemma's kernel-enabled text recheck reached 41.98 tokens/s; the variation
between its two runs does not establish a useful improvement over 41.18.
The retained startup modes are therefore Omni kernels enabled and Gemma
profiles only. These short checks do not measure long-conversation quality
or generation at the full configured context lengths.

The Gemma investigation also exercised a real vLLM factory-created MoE layer
and confirmed that the fast dispatch accepted the pipeline's tensors. That
probe used synthetic weights and is a dispatch check. With real checkpoint
weights and 256 MiB of cache eviction before each timed graph replay, native
experts took 0.350 ms versus 0.322 ms through the vector hook (8% less time),
much smaller than the warm-cache component result below. Eviction itself was
excluded from timing. This supports cache reuse as a contributor to the
microbenchmark/serving gap; it does not isolate every serving bottleneck.

Raw reports, answers, request-counter audits, images, arguments and startup
logs are in `logs/full-model-validation-2026-11-02/`. `serving_compare.py`
never restarts models. Its `mode` positional argument is a report label,
not a runtime switch:

    python3 benchmarks/serving_compare.py qwen3-omni-30b kernels \
      --output logs/omni-serving.json
    python3 benchmarks/serving_compare.py gemma-4-26b profiles \
      --output logs/gemma-serving.json

The default fixture directory is the retained synthetic fixture set under
`logs/performance-2026-11-02/fixtures/`. `--fixture-dir` can select another
set with `long.txt` (item 298 = 3881), `screen.png` (job 4711, errors 0) and
`question.wav` asking for the screenshot's job number and error count.
`--case text`, `batch4`, `prefill`, `media` or `quality` runs a subset.
`moe_vector_compare.py --evict-cache-mib 256` repeats component measurements
after eviction instead of reporting only warm-cache graph replay.

## Omni image and voice input

A second pass on 2026-11-02 prioritized input-to-first-response latency.
The baseline already included the retained Omni kernel hooks above. The only
serving change in this comparison was `OMP_NUM_THREADS=1`, now the default
for Omni in `llm`. Image, serving arguments, context/KV capacity and model
precision stayed identical. Gemma's startup settings were unchanged.

Median time to the first text response, five measured requests after a warmup:

| Input | Before | One preprocessing thread | Time saved |
|---|---:|---:|---:|
| Fresh screenshot | 1100 ms | 1012 ms | 8% |
| Fresh voice, PCM WAV, about 4 s | 201 ms | 166 ms | 17% |
| Fresh voice, microphone-style Opus, about 4 s | 209 ms | 169 ms | 19% |
| Fresh voice, Opus, about 15 s | 342 ms | 276 ms | 19% |
| Fresh screenshot + spoken question | 1124 ms | 1023 ms | 9% |
| Cached screenshot follow-up | 131 ms | 117 ms | 11% |
| Cached screenshot + voice follow-up | 106 ms | 83 ms | 21% |

The screenshot was 2560×1440 with the existing 1M-pixel processing budget;
it was not reduced further. Opus inputs used 48 kHz browser-style recordings.
The clips contained synthetic spoken job/error facts or a question about the
screenshot. PCM and PNG source hashes matched across modes. Fresh cases
required zero multimodal-cache hits; follow-ups required every media item to
hit cache after warmup. Both backends' counters also excluded overlapping
model requests. All 80 measured answers passed with no discarded trials.
Desktop work continued, so these short measurements do not establish tail
latency, noisy-speech accuracy or a guarantee for every image and clip.
Microphone capture, file creation and encoding were outside HTTP timing.

CPU-only extraction explains the choice: the actual Omni HF processor,
decode and resampling chain for a 10-second Opus clip took 171 ms with
32 Torch threads versus 34 ms with one; two seconds took 95 versus 11 ms.
The features were bitwise equal across 1/2/4/8/32 threads for each format.
The decoder worker already uses one thread; the launch setting also limits
the CPU preprocessing process. A separate text-generation check reached
65.5 tokens/s, with no measured regression against the preceding 63.5-token/s
kernel-enabled run. This is not an additional established decode speedup.

The kernel experiments in this pass are retained as experiments:

- `audio_compare.py` loaded only the real audio tower and compared
  `local_llm_audio.py` against native execution at chunk boundaries, an
  8-second attention boundary, longer inputs and mixed recordings. Outputs
  were bitwise equal. The host layout removes dynamic boolean packing and
  several scalar operations, but short-input timings were inconsistent.
  The adaptation duplicates part of the upstream forward method and checks
  its source fingerprints. It is **not packaged or registered for serving**.
- `omni_prefill_tune.py` tested 16 wider-tile configurations at 512 tokens
  using one real expert layer, independent-reference checks and paired
  warm/256-MiB-evicted timings. None established a retained improvement.
  The native expert profile remains unchanged. No context or prefill safety
  limit was raised.

Raw API trials, cache audits, fixture hashes, CPU/tower timings and rejected
tile measurements are under `logs/omni-inputs-2026-11-02/`. Repeat the input
comparison against a running Omni with:

    python3 benchmarks/omni_inputs.py LABEL --repeats 5 \
      --image-variant-offset 119 --audio-variant-offset 8 \
      --output logs/omni-inputs-LABEL.json

Use matching variant offsets across a restarted A/B pair, and new offsets
when repeating fresh cases against an existing cache. The harness checks
cache behavior rather than inferring freshness from a filename. Fixture
creation needs Pillow, `espeak-ng` and `ffmpeg`. An explicit
`OMP_NUM_THREADS=32 ./llm up qwen3-omni-30b` overrides the thread limit when
starting a stopped model; it does not alter a running server.

## Omni consistency and dictation

On 2026-11-03 a broader suite used `whisper-local`'s actual prompts,
temperature, duration-based token limit and non-streaming transcription
function. The client settings were read without recording keys or real
dictation contents. Serving settings stayed unchanged. The suite covered
17 text/image/audio/cache/concurrency cases: 94 measured HTTP requests and
21 warmups, with no overlapping external model requests or request errors.
Cached and fresh cases were checked against backend counters. Fresh media
used new UUIDs with identical pixels/audio, rather than subtly changing the
content; a safety pair shared its recording's UUID within that job.

These are **complete-result** times per trial, not recording time or time to
paste. Paired rows finish when both requests have completed:

| Workload | Trials | Median | Observed range |
|---|---:|---:|---:|
| Identical text prompt, 256 generated tokens | 10 | 4.03 s | 3.99–4.20 s |
| Fresh 2560×1440 screenshot, short answer | 5 | 1.26 s | 1.24–1.26 s |
| Fresh 3840×2160 screenshot, short answer | 5 | 1.31 s | 1.31–1.32 s |
| Fresh portrait screenshot, short answer | 5 | 1.24 s | 1.23–1.30 s |
| 60 s audio, short factual answer | 5 | 0.97 s | 0.93–0.98 s |
| 60 s audio, full streamed dictation | 5 | 3.15 s | 3.09–3.16 s |
| Same full dictation through the real non-streaming client | 5 | 3.20 s | 3.11–3.26 s |
| 60 s audio with much less speech, full dictation | 5 | 1.11 s | 1.11–1.12 s |
| Screenshot + voice, short factual answer | 5 | 1.50 s | 1.46–1.59 s |
| Two full dictations in parallel | 3 pairs | 6.42 s | 6.32–6.42 s |
| Client's screenshot/audio safety pair | 3 pairs | 7.67 s | 7.58–7.80 s |

The long full transcript emitted 149 tokens; the sparse recording emitted
23, despite both WAVs lasting 60 s. Its first streamed text arrived at about
0.76 s, versus 3.15 s for the complete transcript. That distinction explains
why first-token benchmarks do not describe the wait for pasted dictation.
Two concurrent long transcriptions had roughly twice the individual latency.
The local vector-expert path is restricted to single-token steps; concurrent
steps retain native batching. Other kernels and scheduling also contribute,
so this suite does not isolate the slowdown to that one hook.

Mixing an image request with the 256-token text request produced complete
results at about 1.79 s and 5.67 s respectively. The screenshot safety pair
preserves the audio-only transcript and allows only bounded spelling changes
from the image-conditioned result. It waits for both responses, so its cost
is substantial for long dictation. The audio-only app path uses one request;
the safety pair applies when screenshot capture supplies an image.

The real client's one-worker queue was tested without typing anything. A
short recording spent 3.23 s queued behind the long one, then 0.57 s in
transcription, making it ready after 3.80 s. A separate deterministic probe
used the real `Dictation` class with a fake clock, recorder and model: holding
the key for a second recording kept an already-completed first result waiting
60 s and blocked the next queued job. **Those 60 s are simulated client
waiting, not a measured server stall.** The modifier wait prevents pasting
with a shortcut key held. Nicolas subsequently confirmed that the slow test
used held-key recording, making this a strong match for the reported delay.
There are no per-job timing logs to prove how much of that particular wait
was in the client rather than the model.

Repeated identical text had one normalized output across ten runs. Fixed
image-fact and single-dictation fixtures also had stable normalized outputs;
all 30 factual image/audio answers matched the expected job/error numbers.
The two safety arms use different prompts, so their outputs are analyzed
separately; the final merged client text was stable across its three runs.
Literal word-error counts include numbers spelled out instead of digits.
For example, the long transcript repeatedly wrote “four thousand seven
hundred and eleven” rather than `4711`. That is a formatting failure against
the prompt, not evidence that the number was misheard. These synthetic cases
do not establish microphone or noisy-speech accuracy.

No minute-long model stall was observed in this suite. Its small observed
ranges cannot rule out rare pauses, timeouts or problems in unmeasured client
stages. In particular, microphone capture, local silence filtering, screenshot
waiting, modifier-release waiting and clipboard insertion are outside the
HTTP timings. The app currently has no successful-job stage logs, so past
delays cannot be retrospectively assigned to those stages. A useful client
timer should separate recording, queued, requesting, waiting-to-paste and
paste time, with usage/finish reason per safety arm and no content or keys.

`logs/omni-consistency-2026-11-03/` contains raw trials, per-arm/per-request
summaries, fixture durations, the fake-clock reproduction and health checks.
The server and dictation app were not restarted for these tests. The 72 host
checks passed. To repeat a subset against a running server:

    python3 benchmarks/omni_consistency.py --repeats 5 --parallel-repeats 3 \
      --output logs/omni-consistency.json
    python3 benchmarks/omni_consistency.py --only text_repeated --repeats 10 \
      --output logs/omni-identical-text.json
    python3 benchmarks/client_key_wait_probe.py

The harness imports the separate `whisper-local` checkout (override
`--client-project` if necessary), uses only synthetic fixtures, and never
records audio or inserts text into an application. Request errors are
distinct from expected generation caps; only text-throughput cases
intentionally reach their token limit. Reports retain full synthetic
answers, but never keys or real user recordings.

## Expert decoding

`local_llm_moe.py` replaces a single-token INT4 expert operation with two
vector kernels. The first reads selected experts and fuses gate/up projection
with activation. The second fuses down projection, routing weights and expert
reduction. This removes expert sorting and avoids padding one useful matrix
row to sixteen or more. The packed weights and their group32 scales are read
in place. Dequantized weights, intermediate projections and routed expert
outputs retain the original FP16/BF16 rounding points; reductions use FP32.

Dispatch is restricted to these measured single-token cases, 128 experts
and contiguous symmetric INT4 buffers on GPU 0:

| Model | Hidden / expert width | Precision | Activation | Routes / group |
|---|---:|---|---|---|
| Qwen Omni | 2048 / 768 | BF16 | SiLU, gated | 8 / 32 |
| Gemma 26B | 2816 / 704 | FP16 | GELU-tanh, gated | 8 / 32 |
| Nemotron 3.5 | 2688 / 1856 | BF16 | ReLU squared, one projection | 6 / 64 |

Nemotron's experts have no gate: one up projection, squared after ReLU. Its
kernel was added on 2026-11-14 (1.7x per expert step with warm caches, 1.35-1.5x
after eviction, two layers and seeds; 52 to 63 tokens/s in serving).

Zero points, activation scales, clamped activations, expert parallel maps,
LoRA, different layouts, other shapes and batched execution stay native.
The batch restriction matters: a two-token test sharing all selected experts
ran slower with vectors (Omni 0.195 ms native, 0.257 ms through the new hook).
Native matrix grouping reuses those weights more efficiently. Random routing
alone would have hidden this regression.

Real-weight single-token examples, measured through the expert apply hook:

| Model | Existing tuned kernel | Vector hook | Speedup |
|---|---:|---:|---:|
| Omni | 0.191 ms | 0.120 ms | 1.59x |
| Gemma 26B | 0.232 ms | 0.143 ms | 1.63x |

Other runs vary with memory layout and clock state. The reference independently
expands packed nibbles, applies varying per-group scales, rounds to the model
precision and computes the projections with PyTorch FP32 arithmetic. Tests
cover synthetic tail shapes and real first-layer weights, with synthetic
normalized activations and routing. FP16 comparisons use rtol/atol 0.005;
BF16 uses 0.03. Observed maximum real-weight reference errors were smaller for
the vector path than the native path; this is not a model-quality comparison.
Captured graphs were replayed after changing activations, routing weights
and expert IDs, checking outputs against a freshly computed reference.

`moe_vector_compare.py --registered --checkpoint SNAPSHOT` loads only one
expert layer from the cached checkpoint. `--shared-routing` stresses expert
reuse, and `--seed` changes inputs. `--rows` and `--warps` are experimental
tiling controls; serving uses only the validated two-row/four-warp variant.
Running experimental batched vectors does not enable them in serving.

## Dense matrices

`local_llm_dense.py` expands several scale groups inside each K tile, allowing
group32 weights to use K128 tiles instead of a K32 clamp. The measured variant
also uses wider output tiles. It preserves the ExLlama-shuffled weight buffer.
Split-K variants were tested, but the winning serving path needs one partition,
one kernel launch and no partial-sum workspace.

These BF16/group32 shapes are enabled (Granite's group128 shapes at 128 and
512 tokens are listed in the 2026-11-12 report):

| Model | Projection (K, N) | Tokens in the step | Tile M / N / K |
|---|---|---|---|
| Gemma 31B | Down (21504, 5376) | Every integer 2–16 | 16 / 64 / 128 |
| Gemma 31B | Gate/up (5376, 43008) | Every integer 6–16 | 16 / 64 / 128 |
| Gemma 31B | Both | 128 or 512 | 64 / 64 / 128 |
| Olmo | Down (27648, 5120) | Every integer 2–16 | 16 / 64 / 128 |
| Olmo | Gate/up (5120, 55296) | Every integer 6–16 | 16 / 64 / 128 |
| Olmo | Both | 128 or 512 | 64 / 64 / 128 |

Steps of 2–16 tokens are several requests generating at once. Up to five
tokens the native path sends gate/up to its HIP kernel, which the hook never
sees. The 6–16 rows for gate/up and Olmo's 2–16 row were added on 2026-11-14.

All use four warps and one pipeline stage. A second seed with reversed timing
order confirmed the prefill improvements:

| Projection / step | Native | Wider tiles | Speedup |
|---|---:|---:|---:|
| Down / 128 | 6.396 ms | 1.457 ms | 4.39x |
| Down / 512 | 11.490 ms | 4.974 ms | 2.31x |
| Gate/up / 128 | 5.274 ms | 2.876 ms | 1.83x |
| Gate/up / 512 | 14.183 ms | 10.339 ms | 1.37x |

The real registered `torch.ops.vllm.rdna_hybrid_w4a16_apply` route was also
tested with bias: a four-token down projection went from 1.097 to 0.442 ms.
HIP single-token dispatch and an unmeasured batch remained bitwise identical;
the fake implementation and graph replay remained usable. The hook changes
the Triton function looked up by the native operation, preserving its bias
addition and HIP eligibility checks.

Correctness checks independently unshuffle nibbles and vary scales across
groups, test odd tails, require finite outputs, compare tightly to native
outputs and retain independent-reference error metrics. Both native and new
BF16 kernels differ from FP32 arithmetic by about 0.33% in relative L2; a
fixed absolute tolerance at cancellation points would reject the native
implementation too. The independent checks bound relative L2 to 0.6% for
BF16 and 0.1% for FP16, plus a precision-scaled maximum error. These are not
changes to the stored quantization or output precision.

Qwen's group128 projections, Gemma 12B and Gemma 26B's dense shared MLP did
not benefit from the tested variants. Their native paths remain selected.
`dense_compare.py` exposes shape, batch, partition and tile controls to repeat
those negative results as well as the wins.

## Omni encoder scheduling

`local_llm_encoder.py` prepares the scalar maximum attention length on the CPU
once before the eager encoder stack. The attention wrappers otherwise read
the same GPU scalar back in every layer, serializing queued work. CPU tensor
metadata preserves the existing operation schema. Compilation, graph capture
and backends that do not need this bound retain the native behavior.

`encoder_compare.py` repeats a synthetic transformer block at Omni's actual
encoder dimensions and depths, using real vLLM attention wrappers, dense
projections, normalization, GELU and residuals. It excludes audio convolution,
patch embedding, rotary positions and merger work. Outputs were bitwise equal:

| Stack | Native | Host bound | Time saved |
|---|---:|---:|---:|
| Audio, 64 tokens, 32 layers | 17.79 ms | 12.85 ms | 28% |
| Audio, 312 tokens, 32 layers | 26.80 ms | 19.58 ms | 27% |
| Vision, 512 patches, 27 layers | 26.47 ms | 23.21 ms | 12% |
| Vision, 3960 patches, 27 layers | 283.39 ms | 272.74 ms | 4% |

Instrumentation counted 32/27 GPU scalar reads before and zero after; the same
number of CPU scalar reads remain. Registered real Omni methods also passed
metadata and actual CUDA-capture fallback checks in `kernel_integration.py`.
Current [upstream Omni](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_omni_moe_thinker.py)
still has the scalar-bound pattern. Porting the encoder compilation support
used by [Qwen2.5-VL](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_5_vl.py)
is a separate future experiment; changing the runtime flag alone does not
compile Omni's custom encoder classes in this installed version.

## Reproduction

The scripts run inside the GPU image with this repository's `vllm_plugins`
and `benchmarks` on `PYTHONPATH`. Copying them into `/tmp` in an existing
model container and executing a separate Python process is sufficient; this
does not change the serving process or reset its cache. Serialize GPU timing
runs, and avoid loading a duplicate model or running compiler jobs alongside
the timing experiments.

Raw measurements are retained under `logs/kernels-2026-11-02/` (expert paths
and operation integration), `logs/kernel-experiments/dense/`, and
`logs/performance-2026-11-02/kernels/` (encoders). The tests and image smoke
check verify native fallback forwarding, exact dispatch boundaries, metadata
handling, plugin discovery and the startup disable switch. Serving reports
are retained under `logs/full-model-validation-2026-11-02/`. The small serving
checks cover arithmetic, structured output, tool calls, inventory extraction
and media understanding; they do not establish long-conversation quality.
