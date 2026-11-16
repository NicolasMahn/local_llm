# Three model additions, 2026-11-12

This is a working measurement report. All three pinned checkpoints have finished
downloading. Granite has a completed native serving baseline with all 12 synthetic
cases passing, plus two accepted real-weight component comparisons. Its full-model
kernel candidate has also passed all 12 cases and improved fresh prefill. A
lower-memory profile is now the Granite default and has passed four repeated
performance/reasoning cases, 15 gateway API cases and near-limit allocation.
Nemotron has a measured native baseline, documented raw-format failures and a
rejected native-profile candidate. Its retained native path uses a smaller memory
share and passes near-limit capacity; 11 of 15 gateway checks pass, with four
exact-final-format failures documented below. Olmo has loaded and been profiled;
its initial greedy baseline is rejected for truncated thought-only outputs.
Author-recommended sampling exposed additional parser/grammar boundary defects,
now covered by a third-version correctness image and real-tokenizer CPU checks.
The final same-image comparison retains selected dense kernels and a 0.24 memory
share: conservative observed gains are 25.7% fresh prefill and 22.7% four-request
total generation against the fresh native repeat. Thought-only outputs and
capped quality failures remain explicit. Final gateway validation completed:
all ten applicable Olmo requests completed their API protocol, but only five
passed the unchanged strict answer checks. The intermediate sampled run remains
diagnostic evidence only.
Registration, a complete download, a component speedup, and successful startup
are separate results.

Raw reports, request bodies, synthetic answers, backend-counter audits, build logs
and failure traces are in `logs/added-models-2026-11-12/`. Nothing in this report
establishes broad model quality or a gain against the original BF16 checkpoint.

## Machine and runtime

- AMD Ryzen AI Max+ 395, Radeon 8060S, Strix Halo `gfx1151`; 16 CPU cores and
  32 hardware threads. This is one integrated AMD GPU, with shared system RAM.
- Linux `6.19.11-200.fc43.x86_64`; `/proc/meminfo` reports 130,997,700 KiB
  MemTotal, approximately 124.93 GiB. The GPU has a 536,870,912-byte dedicated
  VRAM aperture and borrows RAM. vLLM reports 118,111,600,640 bytes, exactly
  110 GiB, as its GPU memory total. Those are different views of the same
  physical memory; adding them would double-count RAM.
- Boot parameters include `ttm.pages_limit=28835840` and
  `ttm.page_pool_size=28835840`. The existing memory-admission checks must
  remain active; a model's nominal share alone does not prove RAM is available.
- Installed runtime: vLLM `0.29.0+strix`, Torch `2.11.0+rocm10.0.0`, Triton
  `3.8.0+git4cff872c.rocm10.0.0`, Transformers `5.17.0`, compressed-tensors
  `0.17.0`. The Containerfile starts from
  `docker.io/kyuz0/vllm-therock-gfx1151:latest` and adds this repository's plugins
  and matching AOTriton `0.13.50tp` images. A future `latest` can change; use the
  captured local image IDs when reproducing these measurements.
- Patched baseline image `localhost/local-llm:models-20261112`:
  `50c66ddd6c7e177a1928213d9de2465ba2278c8f569f220e4e1661e52214c518`.
  Experimental plugin image `localhost/local-llm:models-kernels-20261112`:
  `8f67a8c5973efaeef819b0b7549329ce03f20932d02ff2e95485dc2b8832a3cb`.
  Building an experimental image is not evidence that its kernels were retained.
- Olmo reasoning/grammar-correctness image
  `localhost/local-llm:models-reasoning-v3-20261112`:
  `ebf7e4083126013fd27d6e6f84c206f465d5ba4cb771b5b8dd63f5138a46c376`.
  The final native/candidate comparison must use this same image. Earlier
  correctness images remain historical diagnostics: feature 1,
  `models-reasoning-20261112`,
  `2b68da5a2c6e2a6842ed569f09fcd7281e4088850e60ae9c8d5a014bd14b2f94`;
  feature 2, `models-reasoning-v2-20261112`,
  `27452bf8eb3795778268fb0c3c7e21a5d39d6f6441c864e37ef1a268ba725f0f`.
  These retain the validated Granite/Nemotron kernel code.
- `localhost/local-llm:latest` was first promoted to the compatibility baseline
  `50c66ddd...`, then to the measured kernel image `8f67a8c5973e...` after Granite's
  complete candidate suite passed, then through Olmo parser feature 1/2 to
  `ebf7e4083126...` after feature 3's CPU checks and image build passed.
  Ordinary lifecycle commands therefore
  include both compatibility fixes and measured dispatch. The former image was preserved as
  `localhost/local-llm:before-models-20261112`, ID
  `90d4e05322200fc097a1b27ea3a21f38ea7c347878e934cbc04db5720099a331`.
  The existing Qwen Omni container still uses that original image and has not
  restarted since `2026-11-10T09:45:10.347765159+01:00`. Retagging the default
  image did not replace its process or serving runtime.

Qwen Omni was already running and was preserved. Timing experiments run serially
and audit every reachable configured backend. Serving reports retain counters,
peak active requests, runtime image, launch arguments and relevant environment
variables. The default GPU-idle threshold is 20%. Several retained Granite
diagnostic/baseline reports explicitly used `--max-idle-gpu-busy 100`; their model
request counters can exclude competing inference, but that override cannot exclude
desktop GPU work. Treat those earlier measurements as preliminary, with that
limitation; the completed `native-queue1-clean` suite uses the normal 20% threshold.

## Checkpoints and precision

The source models are exactly [IBM Granite 4.2 30B](https://huggingface.co/ibm-granite/granite-4.2-30b),
[NVIDIA Nemotron 3.5 Lightning 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
and [Ai2 Olmo 3.1 32B Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think).
These additions do not substitute Granite 4.0, Nemotron 3 Nano, or Olmo Instruct.

| Serving name | Deployment repository and immutable revision | Stored weights | Weight payload |
|---|---|---|---:|
| `granite-4.2-30b` | [`derikn/Granite-4.2-30B-GPTQ-INT4`](https://huggingface.co/derikn/Granite-4.2-30B-GPTQ-INT4/tree/fc0411d102bae7bd3446b0d477ecaca9f8a49961), `fc0411d102bae7bd3446b0d477ecaca9f8a49961` | AutoRound 0.15.1, GPTQ INT4, symmetric, group 128, `desc_act=false`; six safetensors shards | 16,442,945,032 bytes |
| `nemotron-3.5-lightning-30b-a3b` | [`useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16`](https://huggingface.co/useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16/tree/cc977f002c7a396da87dc34c7bb4e5cdba710e24), `cc977f002c7a396da87dc34c7bb4e5cdba710e24` | compressed-tensors GPTQ INT4, symmetric, group 64, static activation ordering; one safetensors shard | 17,826,201,400 bytes |
| `olmo-3.1-32b-think` | [`cyankiwi/Olmo-3.1-32B-Think-AWQ-4bit`](https://huggingface.co/cyankiwi/Olmo-3.1-32B-Think-AWQ-4bit/tree/0aebbdfaa8ba57bada66e87eb1c21c1a30ac2faa), `0aebbdfaa8ba57bada66e87eb1c21c1a30ac2faa` | compressed-tensors INT4, symmetric, group 32, no activation ordering; four safetensors shards | 19,609,154,304 bytes |

Bytes above sum only the downloaded `*.safetensors` files. Repository naming is
not a runtime quantization format: Olmo's AWQ-named repository stores
compressed-tensors, not AutoAWQ GEMM buffers. All three launch with BF16
activations and the native automatic KV dtype; these are weight-only
quantizations. Granite's stored FP16 scale tensors convert to BF16 in serving.

The [Nemotron conversion record](https://huggingface.co/useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16/blob/cc977f002c7a396da87dc34c7bb4e5cdba710e24/quantization.md)
pins its original BF16 source to `63a200063804e06fdb41d6717e43bc92f67859d2`.
It omits the MTP head and sets `num_nextn_predict_layers=0`; speculative MTP
decoding is not part of this deployment. Its inherited chat template is unchanged
by that conversion. Group 64 divides the 1,856-wide expert projection exactly.
Static ordering does not require runtime `g_idx` reordering; the downloaded
checkpoint contains no `g_idx` tensors. Inspection confirms the experts themselves
are packed INT4, including up-projection `[1856, 336]` and down-projection
`[2688, 232]` INT32 storage for a representative expert.

INT4 cuts weight storage and memory traffic substantially compared with BF16,
leaving room for KV cache and the desktop. It can lose accuracy, particularly on
reasoning, code and unusual inputs. A smaller quantization group stores more
scales and permits finer local approximation; it does not guarantee better task
quality between different quantizers. The three formats were chosen for the
available RDNA W4A16 paths and exact model dimensions. No matched BF16-versus-INT4
quality evaluation has been run here, so successful synthetic checks do not prove
quantization parity. NVIDIA's recommended NVFP4/CUDA deployment recipes are not
portable launch recipes for this AMD machine.

Downloads initially completed on the T7 under
`/run/media/nicolas/T7/local_llm/models/<name>/<revision>`. All three were then
copied and verified on native NVMe at
`/home/nicolas/.local/share/local-llm/models/<name>/<revision>`, which is now
selected by `~/.config/local-llm/model-dir`. Verification succeeded before the
duplicate T7 sources were removed, and offline config/tokenizer checks pass for
all three revisions. The three checkpoint directories occupy about 50.2 GiB on
native storage. These models no longer require the T7 mount to start.
Downloads remain resumable and revision-pinned. Disk admission reserves 25 GiB
beyond missing files, and storage health refuses a missing configured removable
mount instead of writing into its empty mount point or silently changing stores.

The relocation record is `logs/storage-2026-11-12/new-models-relocation.json`;
it records verified publication at `2026-11-12T23:03:34+00:00` and source removal
at `2026-11-12T23:03:40+00:00`. A separate verified native hot-cache copy for
Qwen Omni is recorded in `omni-hot-cache.json`; its original serving container
and process remain unchanged. Verified relocation of the Horscht project to
the archive HDD freed 331.4 GiB on native storage. The separate cold Hugging
Face archive completed full checksum verification at
`2026-11-13T00:05:38+00:00` (`huggingface-migration.json`, return code 0).
Five legacy current-ref snapshots (Qwen 27B, Gemma 26B/12B/31B and Muse),
92,520,298,171 bytes or about 86.17 GiB, completed their warm T7 copy at
`2026-11-13T00:11:58+00:00`. Full SHA1 checks, source-freeze checks and CPU-only
offline config/tokenizer reads passed for all five. Because exFAT has no native
symlinks, the destination stores regular current snapshots and `refs/main`,
without duplicate blob storage. Routes were published at
`2026-11-13T00:13:59+00:00`, retaining Omni's native hot-cache route.
`t7-warm-e9121d3925674a319b0e36ee358e6297.json` is the completed copy proof;
its separate `-routes.json` records publication. The earlier plan's
`routes_published=false` is historical, not current status. These are offline
compatibility checks; none of the five legacy models was GPU-started on the new
image for this work.

At `2026-11-13T00:14:22+00:00`, the default Hugging Face cache became a symlink
to the verified HDD archive. `huggingface-cutover.json` proves that the original
Omni container `92a85776603c8657cae39230097356774bb4bacc1f093b114bc80d65171590a8`
retained its original cache root (device 66309, inode 9577246) and could still
access its checkpoint before and after cutover. Its entire active repository
and original root remain until a natural model stop; neither process nor
container was restarted. A separate approximately 26-GiB native hot copy is also
retained safely for its future startup.

After checksum/usage auditing, 21 inactive source replicas were removed between
`2026-11-13T00:19:16+00:00` and `2026-11-13T00:19:28+00:00`, freeing
212,762,525,696 bytes (198.15054 GiB). `retired-hf-prune-result.json` records
preserved active-root/repository identities and zero broken/external symlink
targets in the retained tree. These removals prune verified duplicates, not
unique checkpoints. The first prune attempt refused an uninspectable user
systemd manager; the subsequent narrow exception authenticates its D-Bus
identity, exact argv, cgroup, UID, PPID and start time. The pruning suite has
24 passing fixture tests, the warm-copy suite 17 and route publisher 14;
the full host run has 225 passes, three skips and 50 passing subtests.

The resulting layout puts the three additions and Omni on native NVMe, five
legacy warm models on T7, and Horscht plus the full cold Hugging Face archive
on HDD. Available-space snapshots after the copies/prune are approximately
467 GiB native, 920 GiB T7 and 3,096.1 GiB (3.02 TiB) HDD, versus about 15 GiB
native at the start. The HDD's apparent 71.34-GiB accounting difference is
reserved XFS free extents, not model-file waste: metadata-only enumeration and
statfs reconciliation are in `archive-allocation-audit.json` and
`archive-allocation-result.md`. No reservation or filesystem change was made
because of that accounting result.

Both removable disks currently use desktop mounts. One administrator step is
still pending because sudo authentication was unavailable; this reviewed command
adds UUID-based boot mounts and the archive alias/guards, without formatting or
moving data:

```sh
sudo python3 /home/nicolas/Projects/local_llm/logs/storage-2026-11-12/prepare-archive.py
```

The storage copies, verification and pruning finished before the V3 native Olmo
performance run began at `2026-11-13T00:19:55+00:00`. That run uses the same
V3 image/configuration intended for its candidate, with profiling inactive;
its results and native-process repeats are recorded below. All storage operations
have manifests under
`logs/storage-2026-11-12/`.

## Capabilities and baseline memory limits

| Model | Architecture and checkpoint context | Local limit and reservation | Parser and capability contract |
|---|---|---|---|
| Granite | Dense `GraniteForCausalLM`, 64 layers, hidden 4096, MLP 32768; native 131,072 tokens | port 8007; retained share 0.23 = 25.3 GiB, down from baseline 0.28 = 30.8 GiB; 32,768 context; 16 active sequences | `qwen3` reasoning and `qwen3_coder` tools; thinking toggle; text, schema output and tools |
| Nemotron | Hybrid `NemotronHForCausalLM`, Mamba-2/MoE/attention; 128 routed experts, top 6, non-gated ReLU², hidden 2688, expert width 1856; checkpoint context 262,144 | port 8008; retained share 0.22 = 24.2 GiB, down from baseline 0.30 = 33 GiB; 32,768 context; 16 active sequences | `nemotron_v3` reasoning and `qwen3_coder` tools; thinking toggle; text, schema output and tools; no MTP |
| Olmo | `Olmo3ForCausalLM`, 64 layers, hidden 5120, MLP 27648; 4096-token sliding attention in three layers followed by a full-attention layer; checkpoint context 65,536 | port 8009; retained share 0.24 = 26.4 GiB, baseline 0.30 = 33 GiB; 32,768 context; 16 active sequences | `olmo3` reasoning; always thinks; text and schema output; standard tool calls unsupported |

These are capability contracts to validate, not a claim that every row has passed
serving checks. All three are text-only. Granite's [current model card and
template](https://huggingface.co/ibm-granite/granite-4.2-30b/tree/9e668ce1c538387ef24d3644e9b0606647762636)
use `<think>` reasoning and Qwen-style XML tool calls. The installed old `granite`
reasoning parser expects prose delimiters, while `granite4` tools expect JSON
inside tool tags; those names would select incompatible formats here.

NVIDIA documents `nemotron_v3` and `qwen3_coder` in its [BF16 serving
recipe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).
Its headline long-context claims exceed the pinned deployment config's native
262,144 limit and the conservative local 32,768 limit. No million-token claim is
made for this setup. [Ai2's template](https://huggingface.co/allenai/Olmo-3.1-32B-Think/blob/97ade57a9cc55ed48bcd76be4b7a3b6fee61b80d/chat_template.jinja)
contains its own `functions`/`function_calls` message convention, not standard
OpenAI `tools`/`tool_calls`; no standard tool adapter is advertised for Olmo.

Installed vLLM has native registry entries for Granite and Nemotron. Olmo uses
the Transformers modeling backend; the installed backend supports quantized
linear replacement and interleaved sliding attention. A CPU-only meta probe
constructed Olmo's actual Transformers model with its quantization config and
confirmed attention-backend support. That probe proves architecture construction;
the later native GPU launch and profile below establish loading and dispatch,
but do not by themselves prove quality. See also the [vLLM
supported-model documentation](https://docs.vllm.ai/en/latest/models/supported_models/).

All baseline launches use `--max-num-batched-tokens 512`,
`--max-num-seqs 16`, `--dtype bfloat16`, `--attention-backend TRITON_ATTN`,
and prompt-cache usage reporting. The 512-token prefill ceiling preserves the
project's desktop/GPU-watchdog protection. Context and sequence limits must be
tuned together: 16 admitted sequences do not mean 16 simultaneous full contexts.
Granite's patched native startup reported 15.24 GiB for model loading and
14.83 GiB available KV cache, holding 60,752 tokens: an estimated 1.85 full
32,768-token requests. These are startup capacity estimates, not a tested
long-context correctness or latency result. At the retained 0.23 share, Granite
startup reports 9.35 GiB available KV cache, 38,304 tokens and 1.17 estimated
full-length requests. A 32,512-token synthetic request has now completed at this
share; that capacity check does not establish full-context accuracy or latency
for real conversations. Its exact boundary, elapsed time and memory readings
are recorded below.
Nemotron's startup cache estimate and passing near-limit allocation are recorded
below; long-context accuracy/concurrency remain untested. Olmo's initial startup
reports 12.96 GiB KV cache and 144,446 nominal cache tokens, an estimated 4.41x
32k requests at share 0.30. At retained share 0.24, actual KV is 7.16 GiB/79,781
tokens/2.43x nominal contexts. Batch-four short generation and a fresh 32,512-token
near-limit capacity request pass; long-context accuracy/concurrency are untested.
Cold/warm compilation affects measured startup cache budgeting, as detailed below.

## Measurement contract and preliminary evidence

`benchmarks/added_models.py` sends only synthetic fixtures. Default runs take one
warmup and three measured repetitions. The cases are fixed across variants:

- Single text generation: 256 generated tokens; batch four: 128 per request.
  Thinking is disabled for models that support the toggle. Olmo's thinking stays
  on, so its total generation and useful answer throughput must remain separate.
- Fresh prefill: a 300-item inventory, returning item 298's code `3881`.
  A unique system prefix and per-request `cache_salt` prevent hidden template
  prefixes from hitting cache. Both returned cache usage and backend counters
  are retained. Prefill rate uses uncached tokens divided by the server's
  request-prefill-time counter delta, not prompt tokens divided by TTFT.
- Chat: exact `READY`; reasoning: `17 * 23 - 5 = 386` with separated reasoning
  and final answer; schema output: exactly integer fields `job=4711, errors=0`.
- Where supported, tools are checked with `auto`, `required`, a named tool,
  non-streaming named tool calls, and `none`; one `report_job` call with the
  expected integer arguments is required. JSON is checked streaming and
  non-streaming. Hidden reasoning tags leaking into final content fail checks.

The harness records TTFT, time to first final-answer content, complete latency,
post-first-token generation rate, reasoning-token counts, and useful-answer
tokens divided by complete latency. Tool argument throughput is left unset.
The field called useful-answer throughput is a usage-derived residual:
`(completion_tokens - reasoning_tokens) / complete_latency`, after forcing
observed empty final answers to zero and marking invalid token splits unknown.
It does not retokenize the plain final text. Native reasoning counters exclude
their framing markers, so the residual can include closing delimiters and
formatting as well as answer text. That overhead matters greatly for a
one-number answer; time to first final content and complete latency are more
direct measures of useful responsiveness. Raw usage remains preserved in the
reports, and rates below use this convention rather than claiming exact
plain-answer token counts.
Token-limit completion is intentional for throughput cases; truncation fails
quality cases. Clean trials require stable backend membership and exact request
counter increments, with no peer model traffic. The shared timing lock also
serializes `benchmarks/guarded_command.py` component runs.

| Model / variant | Single text TTFT / complete | Single decode | Batch-four aggregate | Fresh prefill | Status |
|---|---|---:|---:|---:|---|
| Granite patched native, queue 1 | 0.199 s / 19.729 s | 13.05 tokens/s | 32.93 output tokens/s | 230.33 uncached tokens/s | Completed all 12 cases; normal 20% GPU-idle gate |
| Granite kernel candidate, queue 1, share 0.28 | 0.199 s / 19.607 s | 13.14 tokens/s | 33.33 output tokens/s | 279.42 uncached tokens/s | All 12 cases pass; fresh prefill improves |
| Granite retained profile, queue 1, share 0.23, OMP 1 | 0.190 s / 19.340 s | 13.31 tokens/s | 35.33 output tokens/s | 280.36 uncached tokens/s | Four repeated cases, 15 gateway API cases and near-limit allocation pass |
| Nemotron native baseline, share 0.30 | 0.139 s / 4.872 s | 53.90 tokens/s | 147.35 output tokens/s | 1,387.67 uncached tokens/s, separate JSON inventory fixture | Raw inventory/reasoning format checks fail; JSON/tools pass |
| Nemotron retained native, share 0.22, NVMe | 0.142 s / 4.995 s | 52.54 tokens/s | 139.06 output tokens/s | 1,352.80 uncached tokens/s, separate JSON inventory fixture | 8.8 GiB less reservation; near-limit capacity passes; gateway 11/15 strict checks pass, four final-format failures |
| Olmo V3 native, share 0.30 | 0.205 s / 23.263 s | 11.06 total tokens/s; useful 0 | 20.19 total tokens/s; fresh native repeat 28.41 | 188.90 uncached tokens/s; repeat 202.93, separate ACK fixture | Capped text/batch thought only; original quality failures retained |
| Olmo retained V3 kernels, share 0.24 | 0.123 s / 22.617 s | 11.34 total tokens/s; useful 0 | 34.85 total tokens/s | 255.08 uncached tokens/s, separate ACK fixture | 6.6 GiB less reservation; sufficient-budget JSON and near-limit capacity pass; 5/10 strict gateway cases pass, 10/10 protocol-complete |

The completed source is `granite-native-queue1-clean.json`, captured from
`2026-11-12T21:39:45+00:00` to `2026-11-12T21:46:58+00:00`. Each of 12 cases
has one accepted warmup and three accepted measured trial batches: 48 accepted
batches, 60 HTTP requests including batch-four concurrency, and 45 measured HTTP
requests. All accepted correctness checks passed. The report also retains 154
preflight attempts rejected because GPU load exceeded 20%; these are excluded
from timings, not counted as successful requests or bad model answers. Backend
membership and request-counter audits are clean for the accepted trials.

| Granite native case | Median TTFT | First final answer | Complete latency | Total decode | Usage-derived answer-side tokens / complete latency |
|---|---:|---:|---:|---:|---:|
| Single text, 256 tokens | 0.199 s | 0.199 s | 19.729 s | 13.055 tokens/s | 12.976 tokens/s |
| Batch four, 128 tokens each | 0.740 s | 0.740 s | 15.545 s per request | 8.578 tokens/s per request | 8.234 tokens/s per request |
| Fresh inventory prefill | 16.842 s | 16.842 s | 17.002 s | 12.495 tokens/s | 0.176 tokens/s |
| Exact chat | 0.126 s | 0.126 s | 0.202 s | 13.233 tokens/s | 9.903 tokens/s |
| Reasoning | 0.199 s | 18.138 s | 18.215 s | 13.043 tokens/s | 0.220 tokens/s |
| Streaming JSON schema | 0.202 s | 0.202 s | 1.647 s | 13.148 tokens/s | 12.144 tokens/s |
| Non-streaming JSON schema | — | — | 1.642 s | — | 12.179 tokens/s |
| Automatic tool | 0.909 s | — | 3.415 s | — | — |
| Required tool | 0.825 s | — | 3.285 s | — | — |
| Named tool | 0.825 s | — | 3.206 s | — | — |
| Non-streaming named tool | — | — | 3.208 s | — | — |
| Tool choice none | 0.199 s | 0.199 s | 0.275 s | 13.228 tokens/s | 7.280 tokens/s |

Batch-four aggregate output is 32.929 tokens/s; the per-request decode rate
excludes first-token latency and cannot simply be multiplied by four to obtain
aggregate completed-work throughput. The three fresh inventory measurements each
report zero cached tokens and zero backend cache-hit delta, with median
server-observed uncached prefill throughput 230.333 tokens/s. All three reasoning
answers report 232 reasoning tokens and four remaining completion tokens,
including any delimiter/formatting overhead. Thus 13.043 total decode tokens/s
coexists with a usage-derived answer-side rate of 0.220 tokens/s. The direct
median wait for final content is 18.138 s.
These short deterministic fixtures establish API behavior, not broad accuracy.
The older `granite-native-v2-baseline.json` and `granite-quality-only.json` are
retained diagnostics and are superseded by this clean baseline.

Earlier fresh-prefill runs were rejected because a template prefix still produced
16 cached tokens, despite varying user text. The salted rerun is the replacement;
do not quote the rejected reports' approximately 234-token/s provisional estimate
as a clean fresh-prefill result. The completed salted queue-one suite replaces
that estimate. The matched kernel-candidate results follow below.

`granite-kernels-queue1.json` completes the same 12 cases, warmups, measured
repetitions, prompts, output limits and cache checks as the native baseline. It
contains 48 accepted trial batches, 60 accepted HTTP requests including warmups,
and no accepted correctness failures. Another 166 attempts were rejected by the
preflight GPU-idle gate and excluded. The image changes to the experimental
`8f67a8c5973e...` build and `LOCAL_LLM_FAST_KERNELS=1`; share 0.28, context 32,768,
512 batched tokens, 16 sequences, default CPU thread settings, queue count one
and inactive profiler configuration remain the same.

| Granite serving measurement | Native | Kernel candidate | Observed change |
|---|---:|---:|---:|
| Fresh server-observed uncached prefill | 230.333 tokens/s | 279.416 tokens/s | +21.31% |
| Fresh inventory TTFT | 16.842 s | 13.885 s | 17.56% less time |
| Fresh inventory complete latency | 17.002 s | 14.045 s | 17.39% less time |
| Single text decode | 13.055 tokens/s | 13.138 tokens/s | +0.64%; no established decode gain |
| Batch-four aggregate output | 32.929 tokens/s | 33.334 tokens/s | +1.23%; no established concurrency gain |
| Reasoning first final answer | 18.138 s | 18.969 s | 4.58% slower |
| Reasoning usage-derived answer-side rate | 0.220 tokens/s | 0.210 tokens/s | 4.36% lower |

Both variants report 232 reasoning tokens and the correct final answer `386`
in every measured trial, with four completion tokens remaining after reported
reasoning, including possible framing. The candidate's complete reasoning
times are 18.040, 19.046 and 19.614 s, versus native median 18.215 s. The slower
candidate median is retained here; correct answers and faster prefill do not
erase a latency regression. With only three repetitions and this observed
spread, its cause and persistence remain unresolved. The subsequent lower-memory,
single-CPU-thread result below does not reproduce that slower reasoning median.
The much smaller text-decode and batch-four changes are treated as variation,
not new speed claims. JSON and every supported tool-choice mode still pass
the same correctness checks, including non-streaming and tool choice none.

The runtime's default four hardware queues caused persistent idle GPU activity
in the initial Granite process. An explicit `GPU_MAX_HW_QUEUES=1` removed that
observed model-induced idle activity and permitted a clean 0%-idle profile window.
This is a resource-use fix, not an established speed gain: earlier preliminary
single decode was 13.08 tokens/s and the clean queue-one result is 13.05. Transient
desktop load can still exceed the idle threshold, as the 154 rejected preflight
attempts show. CPU preprocessing thread count remains at the default 32 in this
native-versus-kernel comparison; the retained lower-memory profile below changes
that setting to one. The single-thread component harness setting is separate.

## Granite retained memory and thread profile

`granite-low-memory.json` completes the same text, batch-four, fresh inventory
and reasoning cases with one warmup and three measured repetitions. It has
16 accepted trial batches, 28 accepted HTTP requests including warmups and
21 measured HTTP requests, with no accepted correctness failures. Sixty
preflight attempts above the GPU-idle threshold were rejected and excluded.
The explicit changes from the kernel-candidate profile are share 0.28 → 0.23
and `OMP_NUM_THREADS=1`, with the inactive profiler configuration removed.
The pinned checkpoint, kernel image, BF16 dtype, context 32,768, 512 batched
tokens, 16 sequences, one GPU hardware queue, prompts and output limits stay
the same. The reasoning and fresh inventory answers remain correct.

| Measurement | Native, share 0.28 | Kernels, share 0.28 | Retained kernels, share 0.23 and OMP 1 |
|---|---:|---:|---:|
| Nominal GPU memory reservation | 30.8 GiB | 30.8 GiB | 25.3 GiB |
| Single text TTFT / complete | 0.199 / 19.729 s | 0.199 / 19.607 s | 0.190 / 19.340 s |
| Single text decode | 13.055 tokens/s | 13.138 tokens/s | 13.315 tokens/s |
| Batch-four aggregate output | 32.929 tokens/s | 33.334 tokens/s | 35.326 tokens/s |
| Batch-four median complete latency | 15.545 s | 15.347 s | 14.483 s |
| Fresh uncached prefill | 230.333 tokens/s | 279.416 tokens/s | 280.362 tokens/s |
| Fresh inventory TTFT / complete | 16.842 / 17.002 s | 13.885 / 14.045 s | 13.833 / 13.990 s |
| Reasoning first final answer / complete | 18.138 / 18.215 s | 18.969 / 19.046 s | 17.761 / 17.838 s |
| Reasoning usage-derived answer-side rate | 0.220 tokens/s | 0.210 tokens/s | 0.224 tokens/s |

The smaller reservation saves 5.5 GiB and preserves the measured fresh-prefill
gain. Startup reports 9.35 GiB of KV cache and 38,304 cache tokens rather than
14.83 GiB/60,752. The tradeoff is less space for long concurrent conversations:
the startup estimate drops from 1.85 to 1.17 simultaneous full 32k requests.
This configuration is now Granite's default in `model_catalog.py` and `llm`,
with fast kernels, one GPU hardware queue and one preprocessing thread enabled.
The following capacity check verifies one near-limit allocation; the startup
concurrency estimate and long-context accuracy remain untested.

The earlier slower reasoning median does not recur in this profile. Its three
complete reasoning times are 17.833, 17.838 and 17.839 s, each reporting 232
reasoning tokens and four residual completion tokens with final answer `386`.
Both the earlier slower run and
this stable recheck remain in the report. The memory share and CPU thread
setting changed together, and this is a short experiment, so improvements
cannot be assigned separately to one thread, a smaller cache or another
runtime effect. The rationale for retaining this bundle is the demonstrated
memory saving with preserved prefill and passing repeated performance/reasoning
fixtures, rather than an isolated CPU-thread speed claim. The preceding
share-0.28 kernel build passed the full 12-case serving suite. The smaller-share
profile additionally passes the 15 gateway API cases below.

`granite-context.json` checks the actual retained launch. A raw 32,769-token
prompt plus one requested output token is rejected with HTTP 400 in 0.005 s,
with a context-limit error and no completed-engine request. A separate raw
32,512-token prompt plus one output token succeeds with HTTP 200 in 338.185 s.
Both response usage and backend counters confirm zero prefix-cache hits for
the near-limit request. All reachable-model counter audits are clean, including
the unchanged Qwen Omni counters. Host `MemAvailable` is 48,463,040,512 bytes
(45.135 GiB) before and 48,304,103,424 bytes (44.987 GiB) after. This proves that
the configured memory envelope can serve this one synthetic near-limit request.
It does not prove meaningful generation from that repeated-token input,
long-document retrieval, multiple simultaneous 32k requests or acceptable
interactive latency: its fresh prefill takes more than five minutes.

`granite-api-smoke.json` passes all 15 required cases through the gateway at
`http://127.0.0.1:8400`. Both OpenAI and Anthropic endpoints pass streaming and
non-streaming chat, separated reasoning/final answer, and named tool calls with
correct integer argument types. OpenAI also passes streaming/non-streaming
JSON schema and automatic streamed tool choice. All final accepted case audits
are clean. The file retains 72 attempts: preflight load checks rejected most
extra attempts, and one executed automatic-tool attempt overlapping peer work
was discarded and rerun successfully. It is API correctness evidence, not a
matched latency benchmark. Anthropic thinking is controlled using the installed
vLLM extension `chat_template_kwargs.enable_thinking`; this does not establish
support for Anthropic's top-level `thinking`/`budget_tokens` request schema.
Anthropic structured output was not tested.

`granite-idle-queue1.json` records a separate 10-second idle observation: no
requests, engine CPU time equivalent to 0.20% of one core, and every sample
sleeping in `futex_do_wait`. GPU busy reads 2% and package power 13.8–14.3 W.
These readings show that the retained process sleeps when idle. They are not a
controlled before/after power comparison or a model-specific energy measurement.

After these checks the experiment-owned Granite container was stopped to make
room for the next model. The original Qwen Omni container remained running with
its original image and process. Granite's tested defaults remain available
through the usual lifecycle controls; stopping that experiment did not change
the retained configuration.

## Experiments and decisions

| Experiment | Evidence | Decision |
|---|---|---|
| Existing Gemma/Qwen expert hook on Nemotron | Different hidden/expert dimensions, top-6 routing and non-gated ReLU²; existing hook requires measured gated top-8 shapes | Keep native expert dispatch; no shape-based assumption or speedup claim |
| Existing dense Gemma tiles copied to new shapes | Synthetic shape scan in `dense-candidates.jsonl`; mixed wins/regressions and single-token numerical rejections | No blanket dispatch extension |
| Granite synthetic down, 512 tokens, K128/M64/N64, four warps | 8.858 ms native vs 6.378 ms candidate; Granite gate at 512 regressed 9.888 → 10.658 ms | Synthetic lead; later real-weight and serving results determine retained dispatch |
| Olmo synthetic down, 128 tokens, same tile | 7.684 ms native vs 1.623 ms candidate | Synthetic lead; later real-weight checks and V3 serving/profile results validate selected dense shapes |
| Olmo selected group-32 dense paths and share 0.24 | Two-seed real weights, 2,880 verified serving-profile dispatches, fresh native-process repeat and final retained suite | Retain selected down/gate shapes and 6.6-GiB smaller reservation; M4 gate and unknown shapes remain native |
| Real checkpoint dense projections | Two clean audited runs, seeds 42 and 137 with reversed timing/projection order; all 28 timed native/candidate rows pass independent-reference and changed-activation replay checks | Retained Granite dispatch: down at 512 and gate/up at 128; full-suite prefill gain measured |
| Granite smaller reservation and one CPU preprocessing thread | Four repeated cases, 15 gateway cases and one near-limit allocation pass at share 0.23 and OMP 1; 5.5 GiB less reservation; prefill gain preserved | Retain the measured bundle; no isolated CPU-thread speed or long-context accuracy claim |
| GPU hardware queues | Default four queues produced persistent model-induced idle activity; explicit one queue removed it; clean native suite completed | Resource-use fix; no established decode speed gain |

Synthetic scan audits found no competing model requests. A candidate failing
numerical checks is not timed or enabled. The real-weight probe independently
decodes stored nibbles before comparing native and candidate operations. It tests
native-reference error, finite outputs and changed-activation graph replay;
optional 256-MiB eviction precedes timing events and is excluded from kernel
time. A second seed and reversed order are needed before retaining a gain.
Neither synthetic nor real projection probes modify checkpoints or register a
serving hook. A component gain must survive a matched serving comparison.

The real-weight component runs are `granite-real-dense.jsonl` and
`granite-real-dense-seed137.jsonl`; both corresponding `.audit.json` files have
`accepted: true`, return code zero and no competing-request reasons. The first
uses seed 42, native before candidate, and down before gate/up; the second uses
seed 137, candidate before native, and gate/up before down. The candidate uses
one K partition, tiles M64/N64/K128, four warps and one pipeline stage.

| Projection / step / seed | Native warm | Candidate warm | Native after 256-MiB eviction | Candidate after eviction |
|---|---:|---:|---:|---:|
| Down / 512 / 42 | 9.539 ms | 6.685 ms | 11.653 ms | 7.903 ms |
| Down / 512 / 137, reversed | 8.217 ms | 6.429 ms | 10.292 ms | 7.632 ms |
| Gate/up / 128 / 42 | 3.636 ms | 2.949 ms | 4.479 ms | 3.451 ms |
| Gate/up / 128 / 137, reversed | 3.581 ms | 2.924 ms | 4.435 ms | 3.626 ms |

These selected gains persist after eviction and timing-order reversal. Not every
case benefits: down at four tokens regresses 1.043 → 1.195 ms warm; gate/up at
four tokens regresses 0.589 → 1.712 ms. Gate/up at 256 and 512 tokens is slower
in both seeds. Down at 128 has inconsistent eviction results, including
4.409 → 5.113 ms in the second seed, so its warm-cache gain is insufficient to
justify selecting it. Numerically accepted does not mean selected for serving;
faster component rows alone do not establish a full-model speedup. The serving
candidate selects only Granite down at 512 and gate/up at 128; down at 256 is
not enabled by that dispatch whitelist despite its component gain.

Granite's original GPTQ scales include negative values. They are valid signed
scale factors and must be preserved during source-nibble reconstruction. The
probe checks finite, nonzero scales, converts stored FP16 to serving BF16, and
requires bitwise agreement between independent unpacking and the installed
native packing conversion before matrix comparisons. Across these real-weight
comparisons the independent-reference relative L2 error is approximately
0.22–0.33%, with changed-activation graph replay also checked. That error is
relative to arithmetic on the existing quantized weights, not a BF16 source-model
quality comparison.

`granite-real-dense-invalid-unreplayed.jsonl` and its audit are invalid harness
artifacts. The old helper compared captured output before the graph's first
replay, when capture had only recorded its operations. `graph_for` now replays
the graph before returning output to assertions. Those apparent failures are
not evidence of incorrect kernels; use the two corrected reports above.

The completed full candidate serving run uses image `8f67a8c5973e...` with
performance hooks enabled. It keeps the baseline's share 0.28, context 32,768,
prefill ceiling 512, 16 sequences, default CPU threads and one GPU hardware queue.
The profiler is configured but inactive in both compared processes. Fresh
prefill gains are measured above; the retained smaller-reservation comparison
and reasoning-latency recheck follow below.

`LOCAL_LLM_FAST_KERNELS=0` keeps measured performance hooks disabled while retaining
required compatibility fixes. Unsupported shapes, layouts, activation ordering,
precision and batches must stay on native dispatch. There is no safe fallback to
the pre-fix symmetric-GPTQ bug.

## Nemotron CPU preflight before weight loading

The selected Nemotron config has 2,991 ignore entries, including parent expert
paths such as `model.layers.1.mixer.experts.0`, while checkpoint tensors start
with `backbone.layers...`. This does not turn its expert projections into
unquantized layers in the installed runtime. Native `NemotronHForCausalLM`
maps `backbone` to `model`; applying its rename mapper leaves those already-mapped
ignore entries unchanged. Compressed-tensors ignore matching uses exact names
unless an entry explicitly begins `re:`, not automatic parent-prefix matching.

A CPU-only probe used the real installed `CompressedTensorsConfig` and an empty
`RoutedExperts` object with no weight allocation. The parent expert name is
ignored, but its `.up_proj`, `.down_proj` and `.gate_proj` names all select the
same non-null symmetric INT4, group-64, static-order, `pack-quantized` scheme.
The native MoE factory queries those projection names and therefore enters its
WNA16 branch rather than `UnquantizedFusedMoEMethod`. Shared-expert projection
quantization is also selected, while router and norm ignore entries match as
intended. No GPU kernel, inference or full model was started by this probe.

The bounded component harness `benchmarks/nemo_moe_compare.py` also supports
this checkpoint's single `model.safetensors` file without requiring a shard
index, caching one CPU reader handle across tensor fetches. Its config
validator treats empty `transform_config={}` as equivalent to no transform,
while still refusing nonempty unsupported transforms. These were harness
compatibility corrections, not failed serving kernels or changes to weights.
`test_nemo_moe_compare.py` passes all 15 CPU tests in the serving environment
with GPU visibility explicitly disabled; output is saved in
`nemo-moe-cpu-tests.txt`. The tests cover the exact model/layout contract,
non-gated ReLU² and group-64 reference arithmetic, no-index and indexed readers,
cached-handle closure, unsafe paths, missing shards and transform rejection.
No startup blocker was found in these checks. They do not prove GPU weight
loading, serving or performance; the subsequent native launch is recorded below.

## Nemotron native startup and measured bottleneck

Nemotron has now loaded the pinned INT4 checkpoint in the native serving path
with share 0.30, 32,768 context, 16 sequences, BF16 activations, one GPU hardware
queue and performance hooks disabled. Startup reports 16.78 GiB for weights and
14.46 GiB available KV cache, with 1,645,226 nominal KV tokens and a 50.21x
32k-request cache estimate. That is a cache-sizing estimate, not verified
concurrency: `max-num-seqs` is 16, Mamba state also has its own constraints,
and no near-limit/concurrency stress is established by startup. The native
Nemotron config preserves FP32 SSM state and uses the aligned Mamba prefix-cache
mode; the large KV estimate does not change the configured 32k context limit.

The accepted native trace is
`/run/media/nicolas/T7/local_llm/profiles/nemotron/dp0_pp0_tp0_dcp0_ep0_rank0.1794522089216148369.pt.trace.json.gz`.
`nemotron-3.5-lightning-30b-a3b-native-profile.json` has `audit.reasons: []`,
two completed target requests and unchanged Qwen Omni counters. Its diagnostic
pair generates 256 text tokens from 44 fresh prompt tokens, then nine tokens
from the 4,772-token fresh inventory. Analysis and trace SHA256 are recorded
in `nemotron-native-profile-analysis.json`.

The inventory response is `3881 is the answer.`. It extracts the correct code
but fails the existing fixture's exact-output requirement of `3881`. This is
an instruction-format limitation observed with the selected model/quantized
checkpoint, not evidence that runtime prefill is unsupported. Without a matched
BF16 comparison it also cannot be assigned specifically to quantization. The
raw failed report is preserved; no all-quality-pass claim is made. A separate
strict `prefill_json` inventory workload now has its own native baseline below;
its before/after comparison remains separate from the raw-text case.

The capture contains first-shape compilation and is not a warmed baseline.
Its first text token arrives at 2.274 s, total text latency is 7.124 s, and
post-first-token decode is 52.58 tokens/s. Inventory TTFT is 3.833 s and complete
latency 3.984 s, with zero cached tokens in both responses. These single profiled
requests do not establish reproducible prefill, answer throughput or quality.

| Raw GPU category | Whole capture | Share of kernel-duration sum |
|---|---:|---:|
| Native MoE WNA16 GEMMs | 3,994.525 ms, 12,742 calls | 49.09% |
| Dense INT4 GEMMs | 2,036.201 ms | 25.02% |
| Unquantized HIP/BLAS GEMMs, not fully attributed | 1,132.434 ms | 13.92% |
| Routing, activation, normalization and other | 412.411 ms | 5.07% |
| Attention | 391.529 ms | 4.81% |
| Mamba SSU decode state update | 81.755 ms | 1.00% |
| Mamba SSD prefill scan/state operations | 63.994 ms | 0.79% |
| Mamba causal convolution | 24.672 ms | 0.30% |

The total is 8,137.520 ms across 207,146 raw kernel events on one reported
stream; interval-union coverage agrees at the reported precision. Nested CPU
operations and GPU annotations are excluded from that sum. Kernels are assigned
to decode or prefill only if wholly contained in disjoint GPU execution ranges.
The phase totals therefore exclude 962.405 ms outside those ranges, mostly
unquantized/BLAS GEMMs, rather than forcing an uncertain attribution.

| Contained GPU work | Decode | Prefill |
|---|---:|---:|
| Total raw kernel time in phase ranges | 3,797.845 ms | 3,377.270 ms |
| MoE WNA16 GEMMs | 2,046.256 ms, 53.88% | 1,865.410 ms, 55.23% |
| Dense INT4 GEMMs | 1,087.510 ms, 28.63% | 924.560 ms, 27.38% |
| Attention | 14.700 ms, 0.39% | 376.829 ms, 11.16% |
| Mamba SSU state update | 81.755 ms, 2.15% | — |
| Mamba SSD prefill operations | — | 63.994 ms, 1.90% |
| Mamba convolution | 17.166 ms, 0.45% | 7.505 ms, 0.22% |

Execution ranges identify 264 single-token generation steps and 13 prefill
steps: nine at 512 tokens, two at 48, one at 68 and one at 44. Those prefill
counts sum to the two prompts' 4,816 tokens. The actual MoE calls have hidden
width 2,688, 128 router scores, six selected experts and non-gated ReLU²; the
native two-GEMM path accounts for 12,742 launches across 23 MoE layers. The
expert intermediate width is 1,856, group size 64, with shared expert width
3,712. Actual dense input/output shapes visible in the CPU trace include Mamba
input projection `2688 → 10304`, Mamba output `4096 → 2688`, shared expert
`2688 → 3712 → 2688`, and attention QKV `2688 → 4608`. The Mamba custom op
records `[512,10304] → [512,4096]` in full chunks. Checkpoint/config geometry
sets 64 Mamba heads, head dimension 64 and state size 128, with FP32 cache;
these dimensions must not be replaced with Gemma/Qwen expert assumptions.

CPU interval accounting assigns each covered segment once to its innermost
active scope, including 11 crossing compiler intervals that a simple nested
sum would mishandle. `hipEventSynchronize` accounts for 6,442.437 ms of CPU
self time, waiting for GPU completion. First-shape CPU scopes include
`moe_forward_shared` at 660.405 ms self, attention wrappers at 519.076 ms,
Mamba wrappers at 185.298 ms, and substantial time in dense launch annotations.
Two whole-frame compilation scopes total 567.571 ms inclusively; that nested
figure is not added to other CPU times. These cold compiler/cache costs make
the capture unsuitable for assigning a steady-state CPU-thread bottleneck.
Graph-launch runtime calls total 59.810 ms. No memory-bandwidth or occupancy
counters were captured, so the trace does not diagnose a specific hardware
bandwidth limit inside the MoE GEMMs.

Startup warns that both native tuning files are absent:

- `E=128,N=1856,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json` for MoE.
- `headdim=64,dstate=128,device_name=AMD_Radeon_8060S,cache_dtype=float32.json`
  for Mamba SSU.

The measured priority is a bounded native MoE tile/profile comparison: MoE is
over half of the contained GPU work in both phases, while SSU and SSD each
consume only about 2% of their respective phases. Missing configuration files
alone would not justify tuning the smaller component first. Use
`nemo_moe_compare.py` with the exact group-64, top-six, non-gated layout,
independent BF16-boundary reference checks, changed activations/expert IDs,
warm/evicted timings, reversed order and another seed. Test single-token and
representative concurrent/prefill steps before installing a native profile.
Keep the source FP32 SSM cache and safe native fallback. The existing custom
top-eight gated-expert vector hook does not match this model and remains out
of scope. No Nemotron tuning profile or end-to-end gain is retained from this
diagnostic capture; subsequent native measurements and the scoped experiment
are recorded next.

## Nemotron native serving and correctness results

`nemotron-native.json` has three measured text requests and three measured
batch-four groups after warmup. It stops at the original raw-text inventory
format failure; its two completed throughput summaries remain valid, not a
claim that the full suite passed. `nemotron-native-quality.json` separately
passes exact chat, then fails the original reasoning-final-format check: its
final answer includes an explanation and `386`, instead of only `386`, despite
correct arithmetic and 461 separately reported reasoning tokens. The parser
has separated thought from content; verbose final content is not a reasoning
tag leak or unsupported reasoning runtime. Both original fixtures and their
strict checkers are unchanged.

`nemotron-native-json-tools.json` completes all seven requested cases with
one warmup and three measured requests each: streaming/non-streaming JSON,
automatic/required/named tools, non-streaming named tools, and tool choice none.
All accepted answers and typed integer tool arguments pass; no failure is
recorded in that completed report.

`prefill_json` and `reasoning_json` are explicit additional workloads; neither
is included by `--case all`, and neither replaces an earlier failed test.
The inventory variant requires exactly `{"code":3881}` under a strict schema;
the reasoning variant requires exactly `{"remaining":386}` after separated
thinking. `nemotron-native-schema.json` passes both with one warmup and three
measured requests each. The final answer is longer in JSON tokens, so useful
answer-token rates cannot be treated as the same metric as a bare integer.

| Native Nemotron case | TTFT | First final answer | Complete latency | Throughput / result |
|---|---:|---:|---:|---|
| Single text, 256 tokens | 0.139 s | 0.139 s | 4.872 s | 53.904 decode tokens/s; 52.545 usage-derived answer-side tokens/s |
| Four concurrent texts | 0.308 s | 0.308 s | 3.465 s per request | 147.346 aggregate output tokens/s |
| Exact chat | 0.092 s | 0.092 s | 0.129 s | Pass |
| Original raw inventory | — | — | — | Fails exact formatting: extra words after correct code |
| Original raw reasoning | — | — | — | Fails exact formatting; correct 386 and 461 separated thought tokens |
| JSON inventory, new workload | 3.465 s | 3.465 s | 3.630 s | 1,387.674 uncached prefill tokens/s; pass |
| JSON reasoning, new workload | 0.165 s | 8.622 s | 8.771 s | 453 reasoning tokens; 1.140 usage-derived answer-side tokens/s; pass |
| Streaming / non-streaming JSON | 0.107 s / — | 0.107 s / — | 0.458 / 0.457 s | Both pass |
| Automatic / required tool | 0.422 / 0.409 s | — | 0.960 / 0.946 s | Both pass |
| Named streaming / non-streaming tool | 0.404 s / — | — | 0.924 / 0.929 s | Both pass |
| Tool choice none | 0.108 s | 0.108 s | 0.144 s | Exact READY; no tool call |

The schema-prefill number belongs to a different prompt/output contract from
Granite's raw inventory measurement. Do not use those two table cells as a
matched cross-model comparison or report the failed raw case as a JSON pass.
The failed raw results describe this selected checkpoint's instruction adherence;
they do not establish a BF16 model-versus-quantization difference.

The JSON-format cache report `nemotron-native-cache.json` checks a fresh
inventory question followed by a question
about a neighboring item using the same prefix. Three measured pairs pass after
one warmup pair. Every first request reports zero cached tokens; every followup
reports 4,192 cached tokens, supported by backend cache-hit counters. Median
complete latency is 3.600 s fresh and 0.754 s for the followup, with TTFT
3.432 and 0.589 s respectively. The correct codes are 3881 then 3868. Native
Mamba `align` mode and no explicit retention interval already provide this
behavior. This verifies existing prefix-cache reuse, not a gain from a newly
installed optimization or a prefill rate inferred from TTFT.

## Nemotron native-profile experiment

The original six-tile real-weight sweep in `nemotron-moe-seed42.jsonl` has an
audit marked `accepted: false` because Qwen Omni completed overlapping work.
Its timings are retained for diagnosis and excluded as evidence of a gain.
The focused rechecks `nemotron-moe-winner42.jsonl`,
`nemotron-moe-winner137.jsonl` and `nemotron-moe-shared237.jsonl` all have
accepted audits and no contamination reasons. They load one real MoE layer,
use BF16 activations/group-64 weights, independently check output arithmetic,
change activations and routing for graph replay, and time both warm and
256-MiB-evicted operation graphs. Native comparison tolerances are 0.03
relative/absolute with a 1% independent-reference relative-L2 limit.

The candidate native tile is M32/N32/K128, group-M one, one K partition,
four warps and one pipeline stage. Clean component results:

| Routing / step / seed | Native warm | Candidate warm | Native evicted | Candidate evicted |
|---|---:|---:|---:|---:|
| Random / 128 / 42 | 9.122 ms | 7.530 ms | 8.691 ms | 7.028 ms |
| Random / 512 / 42 | 9.648 ms | 7.999 ms | 9.337 ms | 7.848 ms |
| Random / 128 / 137, reversed | 7.909 ms | 6.474 ms | 7.624 ms | 6.309 ms |
| Random / 512 / 137, reversed | 8.211 ms | 7.031 ms | 7.995 ms | 6.915 ms |
| Shared expert IDs / 128 / 237 | 0.773 ms | 1.185 ms | 0.806 ms | 1.197 ms |
| Shared expert IDs / 512 / 237 | 3.114 ms | 4.862 ms | 2.903 ms | 4.562 ms |

Random-routing speed ratios are approximately 1.16–1.24x at 128/512, but the
shared-routing ratios fall to 0.64–0.67x: substantial regressions when tokens
reuse the same experts. Single-token eviction results also slightly regress.
Passing numerical checks and a random-routing component gain are therefore
insufficient to promote this profile. The actual model's routing must be
measured under the same serving workloads before choosing it.

The experimental file remains under
`logs/added-models-2026-11-12/nemo-profile-candidate/`, not installed in the base
image or selected by default. It contains 513 numeric step entries, covers all
steps 1–512, preserves native defaults outside exactly steps 128 and 512, and
includes a guard entry at 513. Only those two steps select the candidate tile.
This prevents vLLM's nearest-step lookup from silently applying it to unmeasured
decode batches. `LOCAL_LLM_MOE_PROFILE_DIR` is scoped to Nemotron and mounts that
folder at `/moe-profiles` with `VLLM_TUNED_CONFIG_FOLDER` set only in its container.
The comparison keeps share 0.30, context 32k, 512 batched tokens, 16 sequences,
default CPU threads, one GPU hardware queue, FP32 SSM cache and native kernel
code. The before/after commands are captured in the reports. Completed candidate
measurements reject this profile, as recorded next; it is not a default.

A prototype trace helper named `profile.py` collided with Python's standard
library `profile` module before valid candidate GPU timing. It was renamed
`capture_model_trace.py`; the affected attempt was deliberately aborted, and
no model/GPU crash or reset occurred. Separately, a missing experimental tuning
file could otherwise have caused vLLM to serve with native defaults under a
misleading candidate label. CLI admission now validates the exact expected
filename, every step 1–512 and supported positive tile settings before loading
weights. `health.py` uses the same validation when a profile folder is configured.
An empty-folder attempt is refused in `nemotron-missing-profile-refusal.log`.
The profile-admission and storage-health suites pass 39 CPU tests. These guards
make absent or incomplete profile experiments visible instead of accepting
unproven tuning results; they do not convert the candidate into a retained
serving optimization.

## Nemotron rejected profiles and retained memory envelope

`nemotron-profile-candidate.json` completes attempts for 14 cases. Twelve cases
have passing repeated summaries, while the required original `prefill` and
`reasoning` fixtures fail the same formatting constraints as native. The report
therefore correctly has `passed: false`. Crucially, the random-routing component
win does not survive actual fresh-schema inventory serving:

| Native-profile comparison, share 0.30 | Native | M32/N32/K128 at steps 128/512 |
|---|---:|---:|
| Single text decode | 53.904 tokens/s | 52.879 tokens/s |
| Batch-four aggregate output | 147.346 tokens/s | 143.740 tokens/s |
| JSON inventory uncached prefill | 1,387.674 tokens/s | 1,324.414 tokens/s |
| JSON inventory TTFT / complete | 3.465 / 3.630 s | 3.631 / 3.798 s |

Schema-prefill throughput falls 4.56%, so the profile is rejected. Its shorter
JSON-reasoning latency is not a demonstrated kernel gain: it emits 412 reasoning
tokens versus native's 453, changing the generated work. The unchanged schema
checks still validate the correct answer, but token-count variation must remain
visible when comparing complete-result latency. The retained path sets no
`VLLM_TUNED_CONFIG_FOLDER` and leaves the native MoE selection in place.

Two further M64/N32 variants with K64 and K128 extend the bounded component
grid from six to eight candidates. `nemotron-moe-wide64.jsonl` and
`nemotron-moe-wide64-shared.jsonl` have accepted audits and passing arithmetic/
graph checks, but every measured 128/512 case is slower than native: speed
ratios span approximately 0.28–0.92x across random/shared routing and warm/
evicted timings. Those wider variants are also rejected. No custom gated
expert hook, reduced SSM precision or unmeasured tuning profile is enabled.

`nemotron-low-memory.json` tests native dispatch at share 0.22 with OMP 1.
It completes 12 passing case summaries and retains the same two required
raw-format failures. A subsequent `nemotron-native-final.json` rechecks text,
batch four, JSON prefill and JSON reasoning after the verified NVMe relocation,
using native frontend thread settings rather than forcing OMP 1. Those four
cases pass with one warmup and three measured repetitions each.

| Native Nemotron setting | Original share 0.30 | Share 0.22, OMP 1 | Retained share 0.22, native frontend, NVMe |
|---|---:|---:|---:|
| Nominal reservation | 33.0 GiB | 24.2 GiB | 24.2 GiB |
| Single text TTFT / complete | 0.139 / 4.872 s | 0.140 / 4.954 s | 0.142 / 4.995 s |
| Single text decode | 53.904 tokens/s | 52.976 tokens/s | 52.544 tokens/s |
| Batch-four aggregate output | 147.346 tokens/s | 141.746 tokens/s | 139.056 tokens/s |
| Batch-four median complete | 3.465 s | 3.610 s | 3.675 s |
| JSON uncached prefill | 1,387.674 tokens/s | 1,365.047 tokens/s | 1,352.802 tokens/s |
| JSON inventory TTFT / complete | 3.465 / 3.630 s | 3.523 / 3.693 s | 3.555 / 3.723 s |
| JSON reasoning first final / complete | 8.622 / 8.771 s | 8.014 / 8.167 s | 8.124 / 8.278 s |
| JSON reasoning tokens | 453 | 412 | 412 |

The retained choice saves 8.8 GiB of reservation, with throughput a few percent
lower in these observations rather than an established speed gain. Against the
original native run, the final observed single decode is 2.52% lower, aggregate
batch throughput 5.63% lower and JSON-prefill throughput 2.51% lower. This is a
memory tradeoff with short-run variation; storage, launch history and thread
settings were not held fixed across every column. OMP 1 does not establish a
separate improvement and is not retained for Nemotron. The frontend therefore
keeps its native thread setting (no OMP override; normally 32 on this runtime),
while vLLM's serving worker still limits Torch execution to one thread.

Actual startup at share 0.22 has 6.26 GiB available KV cache and 711,338 nominal
KV tokens, an estimated 21.71x 32k requests rather than the original
14.46 GiB/1,645,226 tokens/50.21x. These cache estimates exceed the configured
16-sequence admission limit and do not prove Mamba state capacity or real
concurrent full-context behavior. Capacity and gateway results for this
retained profile follow below. The defaults are native MoE, fast hooks
off, one GPU hardware queue, BF16 activations, source FP32 SSM cache, 32,768
context, 16 sequences and the 512-token prefill ceiling. Larger prefill chunks
were not tested: the existing desktop/watchdog safety bound remains in place.

After relocation, native EXT4/NVMe startup logs report `Loading weights took
9.71 seconds` and total model loading of 13.572 s. The initial T7 launch reports
21.48 s in the same weight-loader timer and 25.184 s total model loading; a later
T7 restart reports 21.16 s and 24.963 s. The 9.71-s loader number must not be
compared with a 25-s total-model number as if they measured the same phase.
These are uncontrolled observations after copying/verification and different
cache/warmup history, not an isolated storage benchmark or an inference speedup.

## Nemotron retained context, gateway and cache checks

`nemotron-context.json` verifies the retained 0.22-share native launch from the
NVMe checkpoint. A raw prompt of 32,769 tokens plus one requested output token
gets HTTP 400 with a context-limit error in 0.005 s, without completing an
engine request. A 32,512-token prompt plus one output token gets HTTP 200 in
31.211 s. Response usage reports zero cached tokens and backend prefix-hit
delta is zero; both audits have no rejection reasons and peer counters remain
unchanged. Host `MemAvailable` is 40,840,908,800 bytes (38.036 GiB) before and
40,905,330,688 bytes (38.096 GiB) after. This proves one near-limit allocation
and completion within the configured memory envelope, not long-context
retrieval quality or 16 simultaneous full-length conversations.

`nemotron-api-smoke.json` checks both gateway protocols at the retained settings.
Eleven of 15 strict cases pass: streaming/non-streaming chat on both OpenAI and
Anthropic, OpenAI streaming/non-streaming strict JSON, named typed tools on both
protocols in streaming/non-streaming mode, and OpenAI automatic streamed tool
choice. Four required cases fail: reasoning in streaming and non-streaming mode
on each protocol. The report correctly retains `passed: false`.

All four failed reasoning rows complete the protocol normally with finish
reason `stop`, nonempty separately returned thought, no tool call and the same
verbose final content:

> The box initially contains 17 rows of 23 bolts.
> Total bolts = 17 × 23 = 391.
> After removing 5 bolts: 391 − 5 = 386.
>
> 386

The required final answer is only `386`; the extra explanation is the failure.
Inspection of the recorded rows and a CPU-only checker probe confirm that
changing only the in-memory final-answer string to `386` makes each row pass
every remaining criterion. The stored outputs, prompts and strict checkers are
unchanged. This distinguishes output-format adherence from arithmetic accuracy,
reasoning separation or protocol support. These gateway rows report 527/529
reasoning tokens in OpenAI usage; Anthropic reports 589 total output tokens
without that split field. The older native benchmark's 461-thought-token result
is a different request and is not substituted for the gateway counts.

The explicit gateway `reasoning_json` probe in `nemotron-gateway-schema.json`
returns exactly `{"remaining": 386}` with 412 separately reported reasoning
tokens. Its one accepted trial completes in 8.225 s, after a preflight GPU-load
rejection, with a clean accepted audit. This confirms that constrained output
and separated reasoning work through the gateway for that synthetic case.
One diagnostic request is not a throughput A/B comparison and does not turn the
four raw final-format failures into passes. Anthropic thinking uses the same
vLLM `chat_template_kwargs.enable_thinking` extension described for Granite;
Anthropic structured output remains untested.

`nemotron-final-cache.json` repeats the JSON inventory/followup check on the
retained 0.22-share native frontend configuration. All three measured pairs pass:
first requests report zero cached tokens and every followup reports 4,192 hits,
supported by the backend counters. Median complete latency is 3.708 s fresh and
0.779 s for the followup, with TTFT 3.541 and 0.610 s respectively. Default Mamba
alignment and native prefix-cache settings still work after the memory reduction
and NVMe move. This is existing cache reuse, not a newly claimed tuning gain.

Nemotron's catalog entry now records measured native experts, FP32 Mamba cache
and share 0.22. No custom MoE profile is enabled; fast performance hooks remain
off, frontend OMP is unset, the worker retains its native one-thread limit, and
the one-queue/512-token-prefill safety settings remain. Olmo's independent
validation and retained dense-kernel results follow.

## Olmo initial native startup and measured bottleneck

Olmo loads the pinned group-32 INT4 checkpoint through vLLM's Transformers
modeling backend. Native startup reports 18.4 GiB for loaded weights and
12.96 GiB available KV cache, 144,446 nominal cache tokens and an estimated
4.41x 32k requests at share 0.30. The captured launch uses 32,768 context,
16 sequences, 512-token prefill chunks, BF16 activations, one GPU hardware queue,
default frontend threads, native one-thread worker execution, fast hooks off
and image `8f67a8c5973e...`. These are startup estimates, not verified near-limit
or multi-request memory limits. Olmo always thinks; standard tools remain
unsupported in the catalog.

The accepted native trace is
`/run/media/nicolas/T7/local_llm/profiles/olmo/rank0.1794525556135972882.pt.trace.json.gz`.
`olmo-3.1-32b-think-native-profile.json` has a clean audit with no peer-request
rejection. The trace hash, nonduplicated timing categories, operation shapes
and CPU analysis are retained in `olmo-native-profile-analysis.json`.

Both captured requests reach their generation cap without a final answer:

| Profile request | Fresh input / generated tokens | First generated thought | Complete latency | Post-first-token total decode | Useful final output |
|---|---:|---:|---:|---:|---|
| Compiler explanation | 84 / 256 | 2.822 s | 25.900 s | 11.049 tokens/s | Empty; truncated |
| Inventory | 3,584 / 1,024 | 20.806 s | 120.294 s | 10.283 tokens/s | Empty; truncated |

The reasoning fields contain 1,145 and 3,876 characters respectively, while
the provider reports `reasoning_tokens=0` in usage for both. That split is
inconsistent with the returned content. Subtracting zero from completion-token
usage would wrongly count thinking as useful final output: observed useful
answer throughput is zero in these two requests. TTFT here is time to thought,
and first-final-answer time is absent. The text case can measure capped total
generation, but neither request is a successful useful-answer or quality case.
The native full suite must retain this distinction, with sufficient generation
budgets tested explicitly rather than silently relabeling a cap as success.

The initial common harness forced temperature zero, overriding Olmo's
[recommended sampling](https://huggingface.co/allenai/Olmo-3.1-32B-Think):
temperature 0.6 and top-p 0.95, also present in the downloaded generation config.
The greedy suite looped in reasoning and was stopped with its raw output
preserved as a rejected serving baseline; its remaining quality cases were not
completed. Total capped-generation throughput remains an observation, but
useful output is zero where the final answer is empty. A new native baseline
will use the recommended settings, fixed seed 42 and controlled visible prompt
identity, while fresh cache salts continue to prevent prefix reuse. That
sampling correction comes before hardware tuning. Its effects must not be
presented as kernel gains, and subsequent native/candidate comparisons must
use the same sampling and unchanged correctness checks. This trace still
establishes actual matrix shapes and timings for the captured workload; it
does not establish recommended-sampling quality or user-visible performance.

| Raw GPU category | Whole capture | Share of kernel-duration sum |
|---|---:|---:|
| Native HIP INT4 GEMMs | 106,163.598 ms, 327,168 calls | 74.64% |
| Native Triton INT4 GEMMs | 17,686.623 ms, 2,048 calls | 12.43% |
| Attention | 10,641.779 ms | 7.48% |
| Unquantized HIP/BLAS GEMMs | 5,768.984 ms | 4.06% |
| Remaining kernels | 1,976.380 ms | 1.39% |

Only raw kernel events are summed: 1,015,522 launches on one reported stream,
142,237.363 ms in total and 142,237.360 ms of interval-union coverage. Nested
GPU annotations and CPU calls are excluded. Kernels wholly inside disjoint
execution ranges give 114,504.498 ms of decode work and 21,637.497 ms of prefill
work. A further 6,095.369 ms outside those ranges remains unassigned, including
most unquantized output-projection work.

Within decode ranges, native HIP INT4 GEMMs take 106,163.598 ms, 92.72% of
contained kernel time, and attention 6,617.895 ms, 5.78%. Within prefill ranges,
Triton INT4 GEMMs take 17,389.357 ms, 80.37%, and attention 4,023.884 ms, 18.60%.
The capture has 1,278 single-token decode steps, seven 512-token prefill steps
and one 84-token step, matching the generated tokens and both prompt lengths.
This makes dense matrix operations the measured priority. It does not establish
a memory-bandwidth cause without bandwidth/occupancy counters.

The Transformers backend actually fuses gate and up projections. The native
operation records these exact BF16/group-32 layouts at M=512:

| Projection | K / N | Native INT8 packed buffer | BF16 scales | INT32 hook view |
|---|---:|---|---|---|
| Fused gate/up | 5120 / 55296 | `[55296,2560]` | `[55296,160]` | `[55296,640]` |
| Down | 27648 / 5120 | `[5120,13824]` | `[5120,864]` | `[5120,3456]` |

These are the same ExLlama-shuffled bytes viewed at different packing widths,
not duplicate or dequantized weight storage. Optional zero-point and bias
arguments are absent. QKV is also fused to N=7168. The 512-token down ranges
account for 8,952.048 ms across 448 kernels and fused gate/up for 6,159.473 ms
across 448, compared with 811.381 ms for QKV and 562.971 ms for attention output.
These projection subdivisions are part of the raw dense total and must not
be added to it again.

The geometry matches the existing dense-kernel experiment interface, but the
profile exercises only M=1 decode and M=84/512 prefill. It does not prove M=4
concurrent decoding or M=128 dispatch. It motivated the later real-weight
4/128/512 comparisons, independent stored-nibble conversion, BF16 error checks
and changed-input graph replay recorded below. Exact dtype, group, device,
contiguity, scale shape and no-zero-point guards remain necessary, as does the
native single-token HIP fallback. This initial trace alone did not establish a
retained dispatch or any single-token decode gain.

CPU self-time uses the same per-thread interval sweep as the other analyses,
assigning each covered segment once rather than adding nested labels.
`hipEventSynchronize` accounts for 140,342.686 ms, waiting for GPU completion.
Generation-execution annotations contribute 1,179.032 ms self, attention
wrappers 1,080.635 ms and graph launches 478.434 ms. No whole-frame compiler
interval is recorded; first-shape runtime/JIT overhead is not isolated by this
initial pair. The capture supports GPU matrix work as the dominant cost, not
an established steady-state CPU-thread bottleneck or an answer-quality gain.

## Olmo real-weight dense components

`olmo-real-dense42.jsonl` and `olmo-real-dense137-reverse.jsonl` both have
accepted audits with no competing-request reasons. They load actual layer-zero
group-32 checkpoint weights, use synthetic BF16 activations, independently
expand the original nibbles and check known packing, per-group scales and
selected output columns against FP32 arithmetic on the quantized weights.
Captured graphs are replayed after changing activations. Seed 137 reverses
native/candidate timing order; 256-MiB cache eviction is outside measured
kernel time. Candidate tiles remain M64/N64/K128, four warps, one pipeline
stage and one K partition.

| Projection / step / seed | Native warm | Candidate warm | Native evicted | Candidate evicted |
|---|---:|---:|---:|---:|
| Down / 4 / 42 | 1.180 ms | 0.744 ms | 1.350 ms | 0.846 ms |
| Down / 128 / 42 | 11.155 ms | 2.042 ms | 11.115 ms | 2.128 ms |
| Down / 512 / 42 | 14.802 ms | 6.031 ms | 14.596 ms | 5.726 ms |
| Gate/up / 128 / 42 | 6.311 ms | 2.903 ms | 6.389 ms | 2.796 ms |
| Gate/up / 512 / 42 | 13.529 ms | 11.347 ms | 13.270 ms | 11.335 ms |
| Down / 4 / 137, reversed | 0.949 ms | 0.755 ms | 1.187 ms | 0.835 ms |
| Down / 128 / 137, reversed | 8.151 ms | 1.637 ms | 8.177 ms | 1.765 ms |
| Down / 512 / 137, reversed | 16.547 ms | 5.671 ms | 16.497 ms | 5.785 ms |
| Gate/up / 128 / 137, reversed | 6.136 ms | 2.948 ms | 6.006 ms | 2.746 ms |
| Gate/up / 512 / 137, reversed | 13.013 ms | 10.862 ms | 13.028 ms | 10.744 ms |

All tabled variants pass the established native comparison and independent
reference, including changed-input graph replay. Independent-reference relative
L2 error is approximately 0.31–0.34% for these accepted rows. This validates
arithmetic relative to the stored quantized weights, not parity with original
BF16 model quality or real serving activations.

Gate/up at four tokens fails native equivalence in both seeds, before candidate
timing. About 0.4% of elements exceed the unchanged 1% relative/0.005 absolute
comparison, with maximum absolute difference 0.0234375. It remains native;
the tolerance is not loosened to enable a faster path. Thus the eligible
component leads are down at 4/128/512 and gate/up at 128/512 only. Unknown
batch sizes, layouts, precisions and optional zero points retain native
dispatch, as does single-token HIP decoding. These component gains alone do not establish a full-model serving gain; the
later same-image serving/profile comparison is recorded below.

The first seed-137 command used the nonexistent flag `--reverse-order` and
was refused by argument parsing before valid GPU timing. Its
`olmo-real-dense137.jsonl` and audit remain preserved as rejected harness
artifacts. The corrected command uses `--reverse` and a separate output name;
only `olmo-real-dense137-reverse.jsonl` supplies the second-seed timings above.

## Olmo reasoning correctness protection

The initial native parser inherited an unimplemented token-count method that
returned zero for nonempty reasoning. Separately, a non-streaming response
capped before `</think>` could expose its unfinished thought as final content,
although streaming kept it in reasoning. The Olmo-only compatibility plugin
keeps unfinished thought separate and replaces only the inherited counter stub;
a future implemented native counter remains native. Complete extraction also
remains native. Registration is idempotent and unrelated parsers are unchanged.

The first feature version fixed the original capped sample but missed closing
markers whose final bracket was merged with a newline by the real tokenizer.
In an actual sampled request, the final answer was `READY` while all 479
completion tokens were reported as reasoning. `olmo-parser-v1-rejection.json`
and `olmo-sampled-parser-v1-diagnostic.json` preserve this failure. Its invalid
split is unknown answer throughput, not zero or a claimed useful-answer rate.
No report is corrected retrospectively. The old inventory extraction fixture
also failed its 1,024-token limit with thought only under recommended sampling;
that remains a quality/length limitation even after abandoning greedy sampling.

Feature 2 replaces fixed token-pattern matching with incremental decoded token
spans across the ordinary `</think>` marker. It handles BPE closing variants,
prompt-supplied opening thought, adjacent empty frames and partial markers.
Completed framing markers are excluded from reasoning; incomplete markers remain
thought. Streaming feeds only new spans rather than decoding the full growing
response repeatedly. A token containing thought and a boundary still counts
as reasoning, so completion-minus-reasoning remains a usage residual rather than
an exact retokenization of final text.

Feature 3 protects structured output at a second boundary: real token 89963
decodes to `>{"`, finishing the thought marker and starting JSON in one token.
Native `StructuredOutputManager.trim_reasoning_for_advance` dropped that whole
token, so the grammar lost its JSON prefix. The Olmo-only wrapper advances the
grammar through a retokenized non-whitespace suffix after `</think>` and then
the remaining content IDs. Generated IDs and user-visible text are unchanged.
Whitespace-only suffixes keep native behavior, and a future native trim that
already supplies the suffix is preserved. This is a grammar-advance correction,
not a change to model sampling or a relaxed JSON checker.

The complete CPU suite now passes 82 cases, including real tokenizer/parser/
worker-manager cases and all tested BPE closers. The V3 image build independently
passes 39 cases and skips 43 cases requiring mounted real fixtures; those skips
are not counted as passes. Its separate RDNA build suite passes seven and skips
two device-import cases. `reasoning-v3-image-build.log` retains the exact build
checks. Earlier V1's 51-case run did not cover the newly discovered boundaries.

Image `ebf7e4083126013fd27d6e6f84c206f465d5ba4cb771b5b8dd63f5138a46c376`
is tagged `models-reasoning-v3-20261112` and `latest`. It advertises
`local-llm.olmo-reasoning=3`; CLI admission and health refuse older/missing
feature versions before Olmo startup. The compile-cache namespace is
`runtime-e701f4a75449`; earlier feature namespaces remain separate. This
compatibility/usage fix is active with `LOCAL_LLM_FAST_KERNELS=0` and is not a
performance kernel selection. Granite/Nemotron code is unchanged from their
validated image, and the original running Qwen Omni process/image is unchanged.

## Olmo sampled diagnostics before the final comparison

`olmo-v2-native.json` completed at `2026-11-13T00:03:41+00:00` on feature 2 with
native kernels, share 0.30, 32,768 context, 16 sequences, 512-token prefill and
one GPU hardware queue. These are diagnostic measurements, not the final native
baseline. The final native and candidate runs must both use feature 3, identical
temperature 0.6/top-p 0.95/seed 42, prompt bodies, limits and quality checks.
The V3 native performance run began at `2026-11-13T00:19:55+00:00`, after
storage copying/checks/pruning completed; its completed same-image results and
native-process repeats are recorded next.

| V2 diagnostic case | Median first token / first final (s) | Median complete latency (s) | Total output rate | Answer observation |
|---|---:|---:|---:|---|
| Text, three requests | 0.146 / none | 22.761 | 11.276 tokens/s decode | 256 thought tokens; empty final, zero answer rate |
| Four concurrent texts, twelve requests | 0.468 / none | 17.245 per request | 29.518 tokens/s aggregate | 128 thought tokens each; empty final, zero answer rate |
| Explicit `prefill_ack`, three requests | 18.104 / 65.709 | 65.899 | 197.852 uncached prefill tokens/s | All finish exactly `READY`; 463/635/503 thought tokens |

The acknowledgement fixture uses the same long inventory but asks only for
exactly `READY`. It is a separate explicit-only `prefill_ack` workload, like
`prefill_json`/`reasoning_json`; `--case all` does not include these extras.
The original inventory still requires exactly `3881` and is unchanged. V2's
acknowledgement usage-derived residual is 0.075874 tokens/s; first-final latency
is the direct observation, and that residual includes framing/formatting rather
than measuring the one-word answer in isolation. Completed generation timings
for thought-only text are valid total-generation measurements, not successful
useful-answer throughput. The interrupted V2 quality artifacts do not establish
a completed quality suite or final kernel gain.

For seeded A/B identity, Olmo's visible nonce now uses the accepted trial index
(`benchmark-{case}-{accepted}`), while each fresh prefill attempt receives a
unique cache salt. Rejected idle attempts therefore cannot alter the next
accepted prompt body. Native/candidate bodies must match apart from cache salt;
changing this harness behavior is not itself a hardware speedup.

CPU policy was inspected and left unchanged: `amd-pstate-epp`, governor
`powersave`, active AMD pstate and EPP `balance_performance`.
`powerprofilesctl` is not installed. A V2 diagnostic sample observed GPU
2,763 MHz, 81 degrees C and 129.6 W package RAPL; this is neither a controlled
power comparison nor GPU-only energy. These intermediate V2 diagnostics do not
supply the final native baseline.

## Olmo final serving comparison and retained settings

The accepted final serving reports are `olmo-v3-native.json`,
`olmo-v3-fast.json`, `olmo-v3-native-repeat.json`,
`olmo-v3-native-repeat-prefill.json` and `olmo-v3-final.json`. They use the same
feature-3 image, recommended temperature 0.6/top-p 0.95/seed 42, accepted-index
prompt identities, fresh UUID cache salts for prefill, 32,768 context,
512-token prefill chunks, 16 sequences and one GPU hardware queue. Profiling
was configured but inactive during these runs. Main suites have one warmup and
three measured repetitions; the fresh native repeat has two batch repetitions
and one separate no-warmup prefill repetition. Each accepted trial audits peer
requests and GPU-idle conditions. Exact commands, usage and rejected attempts
remain in the raw reports.

| V3 serving variant | Single total decode (tokens/s) | Four-request aggregate (tokens/s) | Fresh ACK prefill (tokens/s) | ACK first token / first final / complete (s) |
|---|---:|---:|---:|---:|
| Initial native, share 0.30 | 11.0595 | 20.1921 | 188.8990 | 18.9666 / 58.2127 / 58.4057 |
| Fast hooks, share 0.30 | 11.3226 | 34.7727 | 254.2474 | 14.0971 / 52.4800 / 52.6691 |
| Fresh native process repeat, share 0.30 | Not repeated | 28.4050 | 202.9269 | 17.6491 / 59.1971 / 59.3861 |
| Retained fast hooks, share 0.24 | 11.3361 | 34.8546 | 255.0804 | 14.0535 / 56.0270 / 56.2161 |

The fresh native repeat establishes substantial native-run variation: its batch
rate is much higher than the first native run. The earlier V2 diagnostic also
reached 29.518 aggregate tokens/s, but uses another correctness image and is not
a final A/B arm. Against the fresh V3 native repeat, the retained profile is
22.7% faster in four-request total generation and 25.7% faster in fresh prefill.
These are the conservative observed comparisons; the first native/candidate
pair's larger batch difference is not a general speedup claim. Native prefill
was 188.899–202.927 tokens/s and fast prefill 254.247–255.080 in these small
runs. Different cache budgets and cold/warm compilation state are recorded
below; the experiments do not claim every aspect of allocated memory was matched.

Single-text median TTFT/complete latency changes from 0.2055/23.2626 s native
to 0.1230/22.6173 s retained. All text/batch outputs are deliberately capped
thought-only generations (256/128 tokens), with zero observed useful-answer
throughput. The rates above describe total generation. All measured ACK cases
finish exactly `READY`; native thought counts are 426/407/379, fast-share-0.30
432/407/355, and retained 621/445/379. Their first-final differences therefore
mix runtime speed and different sampled reasoning lengths. They do not isolate
a kernel gain in time to a useful answer. The retained ACK usage-derived residual
is 0.088942 tokens/s, with the framing caveat defined above.

The retained default is share 0.24, fast hooks enabled, one GPU hardware queue,
native frontend threading (32 threads, no OMP override), one-thread native
worker, BF16 activations and automatic native KV dtype. It reserves 26.4 GiB
rather than the baseline 33 GiB, saving 6.6 GiB of admission reservation. Startup
reports 18.96 GiB consumed weights/non-Torch memory, 0.28 GiB peak activation,
0.18 GiB graph memory and 7.16 GiB actual KV cache: 79,781 nominal tokens,
2.43 estimated full 32k requests. This estimate is not a guarantee of two
simultaneous full-context conversations. Batch-four short generation and the
single near-limit allocation below were tested.

`olmo-3.1-32b-think-startup-memory-journal.json` preserves startup messages even
for removed experiment containers. The initial cold native share-0.30 process
had 12.96 GiB KV/144,446 tokens/4.41x context and 1.07 GiB peak activation.
The warm fast process had 13.76 GiB/153,327 tokens/4.68x and 0.28 GiB peak;
crucially, the fresh native `FAST=0` repeat also had that same 13.76-GiB pool and
0.28-GiB peak. The approximately 0.8-GiB cache difference is therefore not a
kernel memory-saving result. Cold/warm compilation changed the startup budget.

`olmo-cache-control-audit.json` and inspection of the cached compiled graph show
that RDNA matrix multiplication remains an opaque custom operation, without an
inlined split-K kernel/module. Graphs are captured freshly in each process and
`LOCAL_LLM_FAST_KERNELS=0` returns before the local hook. Sharing the feature-3
compile namespace therefore retains a real native fallback; the warm native
repeat also confirms native GPU dispatch. A logged inability to pickle the
Transformers AOT function is retained as a cache warning, not a model crash.
No speculative cache-isolation change was needed.

## Olmo final profile verifies the changed bottleneck

`olmo-v3-native-profile-analysis.json` and `olmo-v3-fast-profile-analysis.json`
record both trace hashes, raw-only kernel accounting, phase attribution and
seven passing comparison checks: image, launch contract, request bodies apart
from salt, distinct fresh salts, prompt/cache/output counts, clean model-counter
audits and equal GPU phase exposure. These checks compare the launch/request
contract, not allocated KV-pool sizes. The captured workload has a 128-token
single generation, a fresh 3,577-token prompt with one output token, and four
concurrent 32-token generations. All capped outputs are thought only; the
profile itself claims no quality pass.

| Profiled operation | Native mean (ms) | Fast mean (ms) | Attribution |
|---|---:|---:|---|
| M4 down, K27648/N5120 | 1.994916 | 0.734989 | 1,920 calls; model-order/HIP-template/dispatch-guard inference |
| M4 complete decode step, kernel-duration sum | 187.573448 | 105.462817 | 30 equal logical-M4 ranges |
| M512 down, K27648/N5120 | 16.365028 | 5.864547 | 384 calls, direct CPU/HIP correlation |
| M512 fused gate/up, K5120/N55296 | 13.418554 | 10.990996 | 384 calls, direct CPU/HIP correlation |

The M4 down reduction accounts for 98.2% of that phase's saved kernel duration;
its gate/up stays native because its real-weight correctness comparison failed.
The M512 means combine fresh-prefill and mixed ranges; the fresh-only subset
has 378 calls per projection. The candidate contains 2,880 actual local Triton
INT4 dispatches versus none native. Its M505 remainder retains native fallback.
One-token decoding remains the same native HIP path; its 2.47% phase-time
variation is not a custom-kernel gain. Unknown shapes/layouts and unsupported
precision/zero-point configurations also remain native.

Only raw GPU kernels are summed, without nested CPU calls or annotations.
Native kernel duration totals 37,105.785 ms (interval union 37,105.780 ms);
fast totals/union are 29,177.613 ms. Both use one reported stream. These are
profiled-work sums rather than extra wall-clock latency measurements; profiling,
clocks and cache state can perturb them. The unprofiled serving runs establish
the practical gains, while the traces prove the intended dispatch and bottleneck.

## Olmo correctness, capacity and client limits

`olmo-v3-native-quality.json` and `olmo-v3-fast-quality.json` preserve identical
pass/fail classifications with the original checkers and 1,024-token cap:

| Case | Native | Fast | Meaning |
|---|---|---|---|
| Exact chat `READY` | Pass | Pass | Separated thought and final |
| Explicit schema reasoning `remaining=386` | Pass | Pass | Correct arithmetic, schema and separated thought |
| Original raw inventory | Fail | Fail | Thought only at cap, no final |
| Explicit JSON inventory | Fail | Fail | Thought only at cap, no final |
| Job JSON, streaming and non-streaming | Fail | Fail | Thought only at cap, no final |
| Original plain reasoning | Fail | Fail | Correct 386 in verbose final; exact final-format check fails |

Separate `olmo-v3-native-json2048.json` and `olmo-v3-fast-json2048.json` rerun
only job JSON with a 2,048-token budget. Both modes pass streaming and
non-streaming with exact `job=4711, errors=0`; all four responses use 1,360
reasoning tokens. Native complete latencies are 126.329/126.327 s, fast
123.406/123.441 s. This establishes JSON compatibility with sufficient budget,
not a broadly useful JSON latency or isolated kernel speedup. It does not
replace or erase the failing original capped cases. The installed runtime's
generic thinking-budget mechanism was inspected only on CPU; no GPU trial or
new default was adopted, including for natural BPE-merged boundary cases.

At retained share 0.24, `olmo-v3-final-context.json` records HTTP 400 for 32,769
prompt tokens plus one requested output token, and HTTP 200 for 32,512 fresh
prompt tokens plus one completion token in 237.051 s. Cache hits are zero and
accepted counter audits have no reasons. Host available memory was
48,678,060,032 bytes before and 48,671,281,152 after (about 45.34 GiB);
SwapFree was about 0.50 GiB. This tests capacity and refusal, not long-context
accuracy, real-conversation latency or long-context concurrency.

The final `olmo-v3-final-api-smoke.json` completed on 2026-11-13. With the
retained 0.24 profile and a 2,048-token output budget, all ten applicable
requests completed the OpenAI/Anthropic protocol, with thinking kept separate
from final content. Five pass the original strict answer checks; five fail:

| Gateway case | Strict result | Complete latency | Evidence |
|---|---|---:|---|
| OpenAI chat, stream / nonstream | Pass / pass | 42.605 / 42.603 s | Exact READY after trimming whitespace |
| OpenAI reasoning, stream / nonstream | Fail / fail | 69.895 / 49.564 s | Correct 386, but verbose final violates exact-format check |
| OpenAI JSON, stream / nonstream | Pass / pass | 123.436 / 123.204 s | Exact job=4711, errors=0 |
| Anthropic chat, stream / nonstream | Fail / pass | 184.227 / 105.798 s | Stream reaches 2,048-token length cap with thoughts only; nonstream returns READY |
| Anthropic reasoning, stream / nonstream | Fail / fail | 56.698 / 93.565 s | Correct 386, but verbose final violates exact-format check |

The other five tool cases are explicitly not applicable, rather than passing.
All retained attempts have empty contamination reasons; earlier busy-GPU
preflights were discarded. The Anthropic API does not accept a seed, so these
latencies and differing chat outcomes are diagnostic correctness evidence,
not matched throughput comparisons. These are model adherence/overthinking
limits, not HTTP, parser or GPU crashes. No prompt, checker or reasoning
budget was relaxed to turn the failures into passes. A useful final answer
cannot be promised within 2,048 tokens, even for a simple READY request.

Gateway model discovery carries the three models' central capability metadata.
In the external GTK/Pi client, Olmo's `tools=false` suppresses unsupported tools
on model/mode/session changes and immediately before a request; selecting
Granite, Nemotron or a cloud model restores their supported tool set. The actual
installed Pi loader/provider CPU suite passes five cases
(`gtk-capability-tests.log`), including separate native reasoning deltas. A
regression test first caught the SDK turning the system message into an
unsupported `developer` role; explicit `supportsDeveloperRole=false` and
reasoning-effort compatibility metadata fix the actual provider request. New Pi
sessions need a natural reload to pick up the extension. Existing tool-history
messages remain intact, so Olmo needs a fresh conversation when the history
contains unsupported tool roles. No standard Olmo tool support is advertised.

The panel's real load discovers all nine models. Node 22 polling/discovery has
two passing checks. All five GNOME extensions were inactive while ScreenSaver
was active, without an extension-specific UI error; no desktop restart/unlock
was performed. A reload method was unsupported and one local disable/enable
left the panel preference enabled. Visible GNOME rendering is consequently not
claimed validated by the Node/loader checks. Health's swap warning now reports
actual available RAM rather than incorrectly inferring low RAM from allocated
swap; the warning threshold is unchanged.

## Two Granite failures now protected

1. `granite-unpatched-failure.log` records
   `zp shape mismatch: torch.Size([32, 768]) vs (6144, 32)` during the first profile
   pass. AutoGPTQ registered packed `qzeros` even for symmetric biased-unsigned
   weights whose RDNA configuration explicitly has `zero_points=False`.
   Native post-load conversion did not convert or clear that unused tensor, and
   apply forwarded it as expanded zeros. `local_llm_rdna.py` now clears only
   that unused optional parameter after native conversion. Native forward,
   asymmetric zero points, failure propagation and dtype/layout conversion remain
   intact. The image advertises `local-llm.rdna-symmetric-gptq=1`; CLI admission
   and storage/runtime health refuse an image missing the feature before weights
   load. Nine plugin tests passed with device visibility, including the real
   Granite AutoGPTQ QKV factory on meta tensors and real asymmetric conversion.
   Image builds ran seven CPU-only tests and skipped the two device-import tests.
2. `granite-stale-aot-failure.log` records the first patched launch loading its
   earlier AOT graph and raising `Expected tensors only, but got: NoneType`.
   That graph still expected the tensor removed by the compatibility fix.
   Runtime feature versions now determine a separate `VLLM_CACHE_ROOT` namespace,
   currently `/root/.cache/vllm/runtime-07bbd771b0d9`. Existing cached graphs were
   preserved. `test_postload_runtime_changes_get_a_separate_compile_cache` checks
   namespace changes when feature versions change. Generic crashed-engine health
   surfaces either failure through status, panel, chat and gateway diagnostics;
   the README records both root causes and protections.

These were engine-startup failures, not evidence of a GPU reset or a desktop
crash. Restarting alone would have left both causes in place. The patched
namespace launch loaded and served Granite; that is the relevant integration
check beyond the unit stand-ins.

The final host run (`final-host-tests.log`) passes 236 tests and 138 subtests,
with three skips. It runs `python3 -m pytest -q --ignore=logs
--ignore=vllm_plugins` with a temporary PATH alias for the installed Node 22,
so both real panel-method tests execute. `plugin-tests.txt` separately records
131 passing plugin checks; the final image's real-tokenizer reasoning/worker/
grammar suite passes 82 CPU cases. Five installed-Pi provider integration tests
pass in `gtk-capability-tests.log`. Earlier `checks.txt` and
`rdna-fix-tests.txt` remain historical evidence, not final aggregate counts.
The five migration-helper suites separately pass 76 tests in
`logs/storage-2026-11-12/final-migration-tests.log`.
Shell syntax and both repositories' whitespace checks pass. Final service
inspection (`final-service-state.json`) confirms only the original Omni model
remains running, with its original ID and start time. No kernel OOM, GPU reset,
AMDGPU fault/timeout, I/O error or XFS error was found in the experiment window.
These checks do not erase the live model-quality failures recorded above.

## Granite profile: measured bottleneck

The accepted Torch trace is
`/run/media/nicolas/T7/local_llm/profiles/granite/rank0.1794519435041388930.pt.trace.json.gz`.
The earlier trace whose timestamp begins `179451933615` was contaminated and is
excluded. `granite-4.2-30b-profile.json` audits the accepted capture: no peer
inference, no pending target requests before the capture, two completed target
requests, idle GPU 0%, and `reasons: []`. It records the patched native image,
`LOCAL_LLM_FAST_KERNELS=0`, and `GPU_MAX_HW_QUEUES=1`.

The captured pair comprises a 256-token text answer to a 43-token prompt with
32 cached tokens, followed by the fresh 3,868-token inventory with zero cached
tokens and the correct answer `3881`. Their complete latencies were 19.708 s
and 16.921 s. These are two profiled requests, not a stable throughput comparison.
The compact analysis, trace SHA256, event counts, phase grouping and CPU self-time
method are saved in `logs/added-models-2026-11-12/granite-profile-analysis.json`.

| GPU kernel category | Sum of raw kernel duration | Share of kernel-duration sum |
|---|---:|---:|
| Native HIP INT4 matrix multiplication | 18,055.187 ms | 49.76% |
| Native Triton INT4 matrix multiplication | 13,334.103 ms | 36.75% |
| Unified attention | 3,422.253 ms | 9.43% |
| Native HIP BF16 matrix multiplication | 952.935 ms | 2.63% |
| Fused SiLU/gate activation | 251.806 ms | 0.69% |
| Fused RMS normalization | 139.722 ms | 0.39% |
| Remaining kernels | 127.779 ms | 0.35% |

Only raw `kernel` events contribute to this table. GPU annotations, CPU calls
and synchronization are not added to it. All 192,497 recorded kernels are on
the trace's one reported GPU stream. The 36,283.784-ms sum and 36,283.783-ms
interval union differ by just 1.264 microseconds of timestamp overlap. This is
GPU work within the profiled pair, not an additional elapsed-time measurement,
and the percentages do not imply isolated GPU hardware utilization. Multiple
streams would require explicit union accounting rather than adding durations.

GPU execution annotations identify 258 one-token generation steps, seven
512-token prefill chunks, a 284-token remainder and the initial 11 uncached
text-prompt tokens. In kernels wholly inside the one-token ranges, native HIP
INT4 GEMMs take 97.88% of the kernel time. In the 512-token ranges, Triton INT4
GEMMs take 79.71% and attention 18.54%; in the later 284-token range attention
reaches 31.35%. A further 1,163.819 ms of kernels lies outside complete enclosing
execution ranges and remains explicitly unassigned, including BF16 GEMMs and
24 Triton INT4 launches. This avoids pretending annotations perfectly bracket
asynchronous work.

The separately identified prefill projection ranges show where matrix work goes:
at 512 tokens, the down projection `(K=32768, N=4096)` accounts for
5,381.353 ms across 448 kernels, and fused gate/up `(K=4096, N=65536)` for
5,116.852 ms across 448 kernels. QKV and attention-output projections account
for 984.514 and 634.218 ms respectively. These are subdivisions of the matrix
category, not additional time to add to the preceding table. Actual stored
projection dimensions and group-128 layout match the Granite checkpoint probe.

CPU self times subtract contained children per thread across CPU operations,
runtime calls and user annotations, so nested compiled-graph labels cannot
inflate totals. No crossing CPU intervals were found. The largest self time is
`hipEventSynchronize`, 35,805.294 ms over 267 calls: the CPU is waiting for GPU
completion. Other observed self times include 166.155 ms in generation execution
annotations, 72.235 ms in `hipGraphLaunch`, 46.061 ms in the attention wrapper,
33.652 ms in module-kernel launches, 31.121 ms in KV-update wrappers and
21.985 ms in tensor slicing. These small host costs do not justify claiming a
CPU-thread bottleneck. Synchronization is not useful CPU computation and must
not be added to GPU kernel time.

This profile supports prioritizing native INT4 matrix operations for decode
and the large MLP matrices for prefill; it does not support copying expert
kernels or merely raising prefill batch size. Attention is a secondary prefill
cost that grows later in the inventory. No bandwidth, occupancy or cache-miss
counters were captured, so the trace alone cannot distinguish a memory-bandwidth
limit from compute/layout inefficiency within those GEMMs. Retention still
requires real-weight correctness and matched end-to-end measurements. Queue
count, CPU threads and other launch choices need their own comparisons.

## Granite candidate profile: actual dispatch verified

The candidate trace is
`/run/media/nicolas/T7/local_llm/profiles/granite/rank0.1794520841168269724.pt.trace.json.gz`.
Its `granite-4.2-30b-kernels-profile.json` audit has no rejection reasons and
records the same text/inventory prompt and output token lengths and cache usage
as the accepted native profile. The inventory again returns `3881`. The compact
GPU-only comparison is `granite-kernels-profile-analysis.json`, with both trace
identities, SHA256 of the candidate trace, raw category counts and projection
subdivisions. CPU and nested GPU annotation durations are not added to raw kernels.

| Recorded GPU work | Native trace | Candidate trace |
|---|---:|---:|
| Native HIP INT4 GEMMs | 66,048 calls; 18,055.187 ms | 66,048 calls; 18,020.686 ms |
| Native Triton INT4 GEMMs | 2,304 calls; 13,334.103 ms | 1,856 calls; 7,296.925 ms |
| Local `_split_k_matmul` | 0 calls | 448 calls; 3,004.493 ms |
| Unified attention | 17,088 calls; 3,422.253 ms | 17,088 calls; 3,495.014 ms |
| All raw kernels, complete pair | 36,283.784 ms | 33,268.545 ms |

Every one of the 448 local calls is wholly inside a
`hybrid_triton_w4a16 512x4096x32768` GPU range, exactly seven 512-token chunks
times 64 layers. Their launch grid is `[8, 64, 1]`, matching M64/N64 tiles,
512 output rows, 4096 output columns and one K partition. Native down-projection
time for those ranges is 5,381.353 ms; the selected local path takes
3,004.493 ms in the candidate trace. The full native trace independently confirms
zero `_split_k_matmul` calls. Native Triton calls decrease by exactly 448 while
the HIP decode-call count stays unchanged. This proves the real serving process
uses the selected down-at-512 kernel; reused AOT compilation did not silently
bypass the hook.

The profile's other prefill row counts are 11 and 284, so it does not exercise
the gate/up-at-128 whitelist. Gate/up at 512 remains native: its recorded time
also changes, from 5,116.852 to 4,372.425 ms, despite unchanged kernel selection.
That variation is a reason not to assign every millisecond of the end-to-end
gain to the local hook. The trace verifies dispatch and locates the main reduced
projection cost; the repeated unprofiled suite measures the actual prefill gain.
There is one reported stream, and candidate raw-kernel interval union is
33,268.541 ms, within 4.049 microseconds of the duration sum. These are single
profiled pairs and can be perturbed by profiling, cache state and clocks.

## Reproduction

Run from `/home/nicolas/Projects/local_llm`. Do not stop a model owned by another
session. A launch command does not change an already-running container; only
restart models this experiment owns, after checking requests and free memory.

```bash
./download.sh --check granite-4.2-30b nemotron-3.5-lightning-30b-a3b olmo-3.1-32b-think
./download.sh granite-4.2-30b nemotron-3.5-lightning-30b-a3b olmo-3.1-32b-think
./llm config
./llm status
./llm up granite-4.2-30b

LOCAL_LLM_IMAGE=localhost/local-llm:models-20261112 LOCAL_LLM_FAST_KERNELS=0 GPU_MAX_HW_QUEUES=1 ./llm up granite-4.2-30b
python3 benchmarks/added_models.py granite-4.2-30b native-recheck --repeats 3 --warmups 1 --output logs/added-models-2026-11-12/granite-native-recheck.json
python3 benchmarks/added_models.py granite-4.2-30b gateway-recheck --case quality --url http://127.0.0.1:8400/v1 --output logs/added-models-2026-11-12/granite-gateway-recheck.json
python3 benchmarks/api_smoke.py granite-4.2-30b --output logs/added-models-2026-11-12/granite-gateway-api-recheck.json
python3 benchmarks/context_check.py granite-4.2-30b --output logs/added-models-2026-11-12/granite-context.json
```

The same serving suite accepts the other two catalog names. It omits unsupported
Olmo tool cases automatically. A report label only labels output; it never
changes server settings. Keep checkpoint, context, cache dtype, prompts, sampling,
token limits, warmups, repetitions and concurrency identical across a pair.
The plain launch now uses the retained 0.23-share profile. The explicit native
fallback launch also uses the current catalog share; it does not reconstruct
the historical 0.28 baseline merely by changing image or report label. The exact
historical share, launch command and environment are captured in each raw JSON.
Likewise, Nemotron's current share is 0.22 rather than its historical 0.30
baseline; new native/candidate comparisons must use the same share in both arms.
Capture `./llm config`, container image/arguments, health and raw reports for each
variant, and report observed system/GPU memory alongside cache capacity.
`context_check.py` separately requires HTTP 400 for `context+1` raw prompt tokens
and successful serving of `context-256` prompt tokens with one generated token.
It audits peer requests and records host available memory/swap. The near-limit
fixture proves allocation and serving only, not retrieval accuracy or practical
latency in a real 32k conversation. Run it serially, after checking available
capacity, and never combine its long prefill with component timing.

The Nemotron structured workloads and cache checks are explicit invocations:

```bash
env -u OMP_NUM_THREADS -u LOCAL_LLM_MOE_PROFILE_DIR GPU_MAX_HW_QUEUES=1 LOCAL_LLM_FAST_KERNELS=0 ./llm up nemotron-3.5-lightning-30b-a3b
python3 benchmarks/added_models.py nemotron-3.5-lightning-30b-a3b native-structured-recheck --case text --case batch4 --case prefill_json --case reasoning_json --output logs/added-models-2026-11-12/nemotron-native-structured-recheck.json
python3 benchmarks/added_models.py nemotron-3.5-lightning-30b-a3b raw-format-recheck --case prefill --case reasoning --keep-going --output logs/added-models-2026-11-12/nemotron-raw-format-recheck.json
python3 benchmarks/cache_followup.py nemotron-3.5-lightning-30b-a3b native-recheck --json --output logs/added-models-2026-11-12/nemotron-cache-recheck.json
python3 benchmarks/context_check.py nemotron-3.5-lightning-30b-a3b --output logs/added-models-2026-11-12/nemotron-context-recheck.json
python3 benchmarks/api_smoke.py nemotron-3.5-lightning-30b-a3b --output logs/added-models-2026-11-12/nemotron-api-recheck.json
python3 benchmarks/added_models.py nemotron-3.5-lightning-30b-a3b gateway-schema-recheck --case reasoning_json --url http://127.0.0.1:8400/v1 --repeats 1 --warmups 0 --output logs/added-models-2026-11-12/nemotron-gateway-schema-recheck.json
```

The raw-format recheck is expected to exit unsuccessfully if the recorded
instruction-adherence failures recur; `--keep-going` records both failures
without marking them as passed. The strict gateway smoke suite likewise retains
failed reasoning-final-format cases; the separate gateway schema request has
its own output contract. After recording the native baseline, an
experiment-owned stopped Nemotron can use the existing candidate folder:

```bash
LOCAL_LLM_MOE_PROFILE_DIR="$PWD/logs/added-models-2026-11-12/nemo-profile-candidate" GPU_MAX_HW_QUEUES=1 LOCAL_LLM_FAST_KERNELS=0 ./llm up nemotron-3.5-lightning-30b-a3b
python3 benchmarks/added_models.py nemotron-3.5-lightning-30b-a3b profile-structured-recheck --case text --case batch4 --case prefill_json --case reasoning_json --output logs/added-models-2026-11-12/nemotron-profile-structured-recheck.json
```

The folder is a rejected measurement candidate, not a recommended default. A running
container is unchanged by `llm up`; verify captured arguments and tuning-folder
environment before interpreting an A/B label. The candidate profile has a
shared-routing regression and its full-serving prefill regression are recorded
above. The retained Nemotron path uses no tuning-folder override.

Olmo's current catalog launch uses the retained share 0.24. These commands
reproduce a new native/fast pair at that same current share, rather than silently
reconstructing the historical 0.30 runs. Run the first launch/suite, stop only
the experiment-owned model after checking requests, then run the second. Set
no OMP override. Both must resolve the feature-3 image ID recorded above:

```bash
env -u OMP_NUM_THREADS LOCAL_LLM_IMAGE=localhost/local-llm:models-reasoning-v3-20261112 GPU_MAX_HW_QUEUES=1 LOCAL_LLM_FAST_KERNELS=0 ./llm up olmo-3.1-32b-think
python3 benchmarks/added_models.py olmo-3.1-32b-think native-recheck --case text --case batch4 --case prefill_ack --repeats 3 --warmups 1 --timeout 300 --output logs/added-models-2026-11-12/olmo-native-recheck.json

env -u OMP_NUM_THREADS LOCAL_LLM_IMAGE=localhost/local-llm:models-reasoning-v3-20261112 GPU_MAX_HW_QUEUES=1 LOCAL_LLM_FAST_KERNELS=1 ./llm up olmo-3.1-32b-think
python3 benchmarks/added_models.py olmo-3.1-32b-think fast-recheck --case text --case batch4 --case prefill_ack --repeats 3 --warmups 1 --timeout 300 --output logs/added-models-2026-11-12/olmo-fast-recheck.json
python3 benchmarks/added_models.py olmo-3.1-32b-think original-quality-recheck --case prefill --case chat --case reasoning --case json --case json_nonstream --case prefill_json --case reasoning_json --max-tokens 1024 --repeats 1 --warmups 0 --timeout 300 --keep-going --output logs/added-models-2026-11-12/olmo-original-quality-recheck.json
python3 benchmarks/added_models.py olmo-3.1-32b-think json-budget-recheck --case json --case json_nonstream --max-tokens 2048 --repeats 1 --warmups 0 --timeout 300 --output logs/added-models-2026-11-12/olmo-json2048-recheck.json
python3 benchmarks/context_check.py olmo-3.1-32b-think --output logs/added-models-2026-11-12/olmo-context-recheck.json
python3 benchmarks/api_smoke.py olmo-3.1-32b-think --max-tokens 2048 --timeout 300 --output logs/added-models-2026-11-12/olmo-gateway-api2048-recheck.json
```

Repeat the original-quality and 2,048-token JSON commands in each arm for a new
correctness comparison; the original-quality command is expected to fail if the
recorded cap/adherence limits recur. The harness applies Olmo's catalog sampling
and seed. Do not infer a useful-answer speedup from capped thought-only output,
or treat a report label as a launch override. Historical 0.30 arguments and
actual startup budgets remain in the raw reports/journal extraction.

CPU-only real-checkpoint packing inspection, inside a disposable container:

```bash
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined --security-opt label=disable -v "$PWD:/app:ro" -v /home/nicolas/.local/share/local-llm/models:/models:ro -w /app localhost/local-llm:models-20261112 python3 benchmarks/dense_checkpoint_compare.py --checkpoint /models/granite-4.2-30b/fc0411d102bae7bd3446b0d477ecaca9f8a49961 --model granite-4.2-30b --projection down --inspect-only
```

The guarded real-weight GPU comparison can be repeated one projection at a time:

```bash
python3 benchmarks/guarded_command.py granite-4.2-30b --output logs/added-models-2026-11-12/granite-down-recheck.jsonl -- podman run --rm --network none --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined --security-opt label=disable -v "$PWD:/workspace:ro" -v /home/nicolas/.local/share/local-llm/models/granite-4.2-30b/fc0411d102bae7bd3446b0d477ecaca9f8a49961:/checkpoint:ro -e OMP_NUM_THREADS=1 -e GPU_MAX_HW_QUEUES=1 localhost/local-llm:models-kernels-20261112 python3 /workspace/benchmarks/dense_checkpoint_compare.py --model granite-4.2-30b --checkpoint /checkpoint --projection down --batches 4 128 256 512 --seed 42 --evict-cache-mib 256 --keep-going
```

Repeat with `--seed 137 --reverse` and another output filename, then both
`--projection down` and `--projection gate`. Use the exact Olmo snapshot/model
name for its separate tests; Granite's results do not apply to that model.
`guarded_command.py` shares the serving timing lock and saves the full command,
return code and peer-request audit as a companion `.audit.json`. The exact
two-projection commands used for the accepted Granite experiments are also
preserved in their audit JSON. Do not run component kernels alongside serving
timings. CPU-only inspection does not require a clean GPU timing window.

The final Olmo 2,048-token gateway API report is complete: ten applicable
requests are protocol-complete, five pass strict answer checks, and five tools
are not applicable. Retained settings, before/after measurements, rejected
variants, cache/capacity limits and quality failures are recorded above.
No broad BF16 quality parity,
long-context accuracy or successful standard Olmo tools is claimed. The separate
administrator boot-mount command also remains pending; current desktop mounts
and runtime storage identity checks are active.
