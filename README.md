# local_llm

Local models on the Strix Halo box, served through one port that speaks both
the OpenAI and the Anthropic API.

    ./download.sh        fetch the weights (resumable, safe to re-run)
    ./llm install        run the monitor and the control service from boot on (once)
    ./llm up qwen3.8-27b start the gateway and these models
    ./llm up             start the models that ran last (what happens at boot)
    ./llm stop <name>    stop one of them
    ./llm status         which servers answer, and any problems
    ./llm logs <name>    follow one server
    ./llm key            the API key (made on first start)
    ./llm admin-key      the key that may also start and stop models
    ./llm owner-key      the key that may also see, make and block keys
    ./llm web <name>     serve the gateway to the internet over HTTPS
    ./llm down           stop everything
    ./llm gateway        restart the gateway (after editing gateway.py)

    KEY=$(./llm key)

    curl fedora.local:8400/v1/chat/completions -H "authorization: Bearer $KEY" \
      -H 'content-type: application/json' \
      -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "hi"}]}'

    curl fedora.local:8400/v1/messages -H "x-api-key: $KEY" \
      -H 'content-type: application/json' \
      -d '{"model": "gemma-4-26b", "max_tokens": 100,
           "messages": [{"role": "user", "content": "hi"}]}'

OpenAI clients use `http://fedora.local:8400/v1` as base URL, Anthropic
clients `http://fedora.local:8400`. Opening `http://fedora.local:8400` in a
browser on any device in the network gives a small test chat. It asks for the
key once and keeps it in that browser; `http://fedora.local:8400/#api=<key>`
saves it in one step (the panel's "Chat" row copies that link).

The page takes pictures and videos (PNG, JPEG, WebP, GIF, MP4, WebM, up to
50 MB) for the models that see, and sound for `qwen3-omni-30b` and `gemma-4-12b`; with another
model selected it says so instead. The desktop chat window, which also keeps
past chats, is its own project, local_llm_chat.

| Model | Role |
|---|---|
| `qwen3.8-27b` | the smart one: hard tasks, batch jobs. Slow per request (dense). ~13 tokens/s, ~97 in total at 16 requests; reads ~400 tokens/s. |
| `gemma-4-26b` | the fast one: agents and high-volume work; sees pictures and videos. ~41 tokens/s, reads ~1,150 tokens/s. |
| `gemma-4-12b` | the one that hears: speech plus a screenshot in one request, no transcription model in between. ~25 tokens/s, reads ~540 tokens/s. |
| `gemma-4-31b` | Gemma's largest, dense: for when the 26B is not good enough. ~10 tokens/s, reads ~215 tokens/s; 131k context. |
| `qwen3-omni-30b` | the fast one that sees and hears: a screenshot plus speech answered in ~2.5 s, transcribes almost word for word, audio clips longer than 30 s. ~64 tokens/s in a short chat, ~35 behind 24,000 tokens; reads ~1,300 tokens/s; 65k context. |
| `muse-glimmer-30b` | Meta's open agent model, dense; always thinks first, sees pictures. ~14 tokens/s, reads ~380 tokens/s; 131k context. Does not enforce a JSON schema. |
| `granite-4.2-30b` | IBM's dense 30B text model; optional thinking and XML tool calls. ~13 tokens/s, reads ~330 tokens/s. Pinned INT4 checkpoint. |
| `nemotron-3.5-lightning-30b-a3b` | NVIDIA's hybrid Mamba/attention MoE text model; optional thinking and tools. ~63 tokens/s and barely slower in a long conversation (~61 behind 24,000 tokens); reads ~1,500 tokens/s. Pinned INT4 checkpoint, without MTP draft heads. |
| `olmo-3.1-32b-think` | Ai2's dense text reasoning model. Always thinks, ~11 tokens/s including the thinking; its released template does not support standard OpenAI tool requests. Pinned INT4 checkpoint. |
| `nanbeige-4.3-33b` | Nanbeige's dense 33B, top of Artificial Analysis' 4–40B ranking; thinking and tools. Answers every check but slowly: ~10 tokens/s, ~20 in total at 16 requests; reads ~165 tokens/s. |
| `mistral-small-3.2-24b` | Mistral's dense 24B; sees pictures, tools, no thinking. ~15 tokens/s, reads ~340 tokens/s. |
| `ministral-3-3b` | Mistral's edge model; sees pictures, tools. ~75 tokens/s. |
| `nemotron-3-nano-4b` | NVIDIA's tiny hybrid model, optional thinking and tools; passed every check. ~31 tokens/s (BF16), reads ~3,500 tokens/s. |
| `granite-4.2-3b` | IBM's tiny dense model, optional thinking and tools. ~32 tokens/s (BF16). |
| `minicpm5-2b` | the fastest: ~100 tokens/s, ~375 in total at 16 requests; optional thinking and tools. Thinks long on arithmetic. |
| `inkling-2-mini` | Thinking Machines' looped MoE (26B, 3.4B active); sees pictures, tools, always thinks a little. Served by llama.cpp, since vLLM runs it only on NVIDIA Hopper or newer. 65k context. ~19 tokens/s, reads ~430 tokens/s behind 8k tokens and ~240 across a 60k prompt, all measured next to a busy `gemma-4-31b` (benchmarks/inkling-2-mini-2026-11-21.md). |

Not all of them fit at once (their memory shares are in `llm`); start the ones
you need from the website, the API, the panel or `./llm up`.

[All nine models on one suite](benchmarks/all-models-2026-11-14.md) compares
them with the same cases, including what a long conversation costs, and lists
what that changed. The speeds above are from there.

The [2026-11-12 additions report](benchmarks/added-models-2026-11-12.md)
records pinned checkpoints, before/after tests and rejected experiments for
the three newest models, as they were set up then. Granite
improved fresh prefill by 21.3%; Nemotron kept its native kernels at the time and reserves
8.8 GiB less memory. Olmo's conservative observed gains are 25.7% fresh prefill
and 22.7% generation at concurrency four, including thought tokens. Its simple
JSON takes about two minutes and some plain-answer checks still fail; it is
available for experimentation, with those limits recorded. All three have a
verified 32,768-token local context capacity; this is not a long-context
accuracy result. `./llm usage` now collects usage for future storage decisions.

The [2026-11-16 additions](benchmarks/small-models-2026-11-16.md) added the
six models after Olmo in the table: why each, their suite results, and the five
things that broke on the way, each with a protection.

### Sound

`qwen3-omni-30b` and `gemma-4-12b` take audio in OpenAI chat requests
(`/v1/chat/completions`), as a content part after the text:

    {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}

or `{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}`.
WAV, FLAC, MP3, M4A, OGG and WebM all work (tested). On `gemma-4-12b` each
clip may be at most 30 s: vLLM cuts longer ones without an error, so split
long recordings; `qwen3-omni-30b` heard a 50 s clip whole. The
Anthropic API (`/v1/messages`) has no audio block, so it refuses audio. Google's
prompt for transcribing works best; without it the model sometimes added a
translation nobody asked for:

    Transcribe the following speech segment in English into English text.
    Follow these specific instructions for formatting the answer: Only output
    the transcription, with no newlines. When transcribing numbers, write the
    digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.

On LibriSpeech clips it transcribed word for word apart from rare names
("Leighton" as "Leytens"), in 1–2 s. A picture and a spoken question about it
in one request were answered correctly. Larger Gemmas have no audio at all;
Gemma 4's E4B also hears, but is much weaker.

`qwen3-omni-30b` is the faster choice (2026-10-29, same Retina screenshot
plus a spoken question, a fresh image each time): first word after 1.5 s,
whole answer 2.4 s, against 1.2 s and 3.4 s on `gemma-4-12b`, and it
transcribed the LibriSpeech and German clips word for word. It reads images
near full size, which made that screenshot 5,000 tokens and 10 s, so `llm`
caps images at 1M pixels (~1,000 tokens); fine print can then be misread. A
request can ask for more, at the price of time (2M pixels: 5 s, small text
read right):

    "mm_processor_kwargs": {"size": {"longest_edge": 2007040, "shortest_edge": 401408}}

Small images are enlarged to 400k pixels: below that it read a large "4711"
as "411".

The gateway listens on the local network and wants a key for everything but
the chat page and a bare `/health`. Three keys, all made on first use:

- the API key (`~/.config/local-llm/api-key`, `./llm key`) reaches the models
  and the measurements. It is the one that goes into clients;
- the admin key (`~/.config/local-llm/admin-key`, `./llm admin-key`) also
  starts and stops models. Keep it to the website and your own scripts, so a
  key leaked from some client cannot turn models off. It works only over
  HTTPS or from this computer: the gateway refuses it over plain HTTP from the
  network, and the page does not send it there;
- the owner key (`~/.config/local-llm/owner-key`, `./llm owner-key`) also
  sees, makes and blocks all other keys (below). Keep it to yourself. Like
  the admin key, it works only over HTTPS or from this computer.

The traffic inside the network is plain HTTP, so anyone on the same network
could read prompts and the API key; use the HTTPS name where you can. The
model servers behind it (ports 8001–8009, and their engines' internal
sockets) have no key and listen on localhost only; keep it that way.
`test_gateway.py` checks that every route the gateway has turns away a request
without a key, so a new route is closed until it is made public on purpose.
The chat page runs only its own script (a fresh nonce each time), so a slip
in how it escapes model output cannot become a way to steal the keys.

A red team (2026-10-29: from this computer, from the network, and from a
throwaway Hetzner server) found no way past the key checks. It found the
engines' sockets on the network (fixed, see below), the admin key crossing
the network in plain HTTP and the page allowing any inline script (both
fixed as described here).

## Starting and stopping models from anywhere

The website (the chat page) and an API start, stop, restart and switch models,
with the admin key. `control.py`, a user service (`./llm install`), does the
work through `./llm`, so the same memory checks apply; the gateway, which runs
in a container and cannot start containers, passes requests on to it over a
Unix socket.

    A=$(./llm admin-key)
    curl -H "x-api-key: $A" https://<name>/admin/models
    curl -X POST -H "x-api-key: $A" https://<name>/admin/models/gemma-4-12b/start
    curl -X POST -H "x-api-key: $A" https://<name>/admin/models/gemma-4-31b/start?replace=true
    curl -X POST -H "x-api-key: $A" https://<name>/admin/models/gemma-4-26b/restart
    curl -X POST -H "x-api-key: $A" https://<name>/admin/models/qwen3.8-27b/stop

There is no standard for this (OpenAI's API has none; Ollama, LM Studio and
llama-swap each have their own), so it is plain REST. Starting takes minutes,
so the answer is `202 Accepted` right away and `GET /admin/models` shows each
model's state (`live`, `starting`, `failed` with the reason, `off`), what is
being done to it, and why the last attempt failed. A model that does not fit
next to the running ones gets `409` with the models that would have to stop;
asking again with `?replace=true` stops them first (the fewest, then the
smallest). Stop and restart interrupt a start that is still going.

The models that run are remembered (`~/.local/state/local-llm/wanted`), and
the control service starts them again at boot, together with the gateway and
HTTPS. A model that crashes is restarted up to three times by podman; after
that it stays down with its reason, and a restart from the website tries again.

## Handing out keys, and taking them back

Give each person their own key, so you can block one without locking out
everyone. On the chat page, put the owner key in either key box and a key
button appears (to the owner only): it makes keys, copies a key or a link
that signs in with it, and blocks and deletes them. The same, from anywhere
over HTTPS:

    O=$(./llm owner-key)
    curl -H "x-api-key: $O" https://<name>/owner/keys
    curl -X POST -H "x-api-key: $O" https://<name>/owner/keys \
      -H 'content-type: application/json' -d '{"name": "sam", "role": "admin"}'
    curl -X POST -H "x-api-key: $O" https://<name>/owner/keys/sam/block
    curl -X POST -H "x-api-key: $O" https://<name>/owner/keys/sam/unblock
    curl -X DELETE -H "x-api-key: $O" https://<name>/owner/keys/sam

A role is `api` (models only) or `admin` (also starts and stops them). The
list shows each key with its requests, when it was last used and from which
address. A block takes effect on the next request and lasts across restarts.
The shared keys from `./llm key` and `./llm admin-key` are in the list as
`api` and `admin`; they can be blocked, but not deleted. Blocking one of them
locks out everyone you gave it to. To replace one, delete its file
(`~/.config/local-llm/admin-key`, say) and run `./llm gateway`: the new key
starts unblocked.

The list is `~/.config/local-llm/keys/keys.json`. The gateway rewrites it, so
edit it only while the gateway is stopped. The owner key is not in it, so it
can never be blocked.

## Waking the computer over the network

Prepared: the wired connection (`eno1`) wakes on a magic packet
(`nmcli connection modify "Wired connection 1" 802-3-ethernet.wake-on-lan magic`;
`... wake-on-lan default` undoes it), and from boot on the control service
brings back the gateway, HTTPS and the models that ran last, without anyone
logging in (lingering is on).

Still to do, by hand:

1. In the GMKtec EVO-X2's BIOS, turn on waking from LAN / power on by PCIe
   (and turn off any deep-sleep / ErP setting that cuts the network card's
   power when off).
2. In the FRITZ!Box, under Internet → Permit Access → the port sharing for this
   computer, tick "start this computer automatically when accessed from the
   Internet". Opening the HTTPS address then boots it; the MyFRITZ!App and the
   FRITZ!Box's network page also have a wake button.
3. Then a shutdown from the website can follow, and with it the question
   whether suspend (seconds to wake, models stay loaded) works on this chip.

## From the internet

`./llm web <name>` puts Caddy in front of the gateway: HTTPS under that name,
with a certificate it gets and renews itself. `./llm web off` closes it again.
`./llm url` prints the address in use. In use:
`https://fedora.4cx12605olebbktj.myfritz.net`.

The name is this computer's MyFRITZ device name, not the router's. The
router's own name (`4cx12605olebbktj.myfritz.net`) resolves over IPv6 to the
FRITZ!Box itself, which answers with its login page; browsers try IPv6 first,
so at home they never reached the gateway. The device name resolves over IPv4
to the router and over IPv6 straight to this computer. It takes three things:

1. In the FRITZ!Box, under Internet → Permit Access → Port Sharing, this
   computer has one sharing: a MyFRITZ! sharing of type HTTPS server. It opens
   port 443 over IPv4 and IPv6; nothing else is open.
2. The IPv6 address the FRITZ!Box shares (interface ID `::40e1:6f85:bed:dc0b`)
   must stay this computer's, also when 1&1 changes the prefix, so
   NetworkManager sets it:
   `nmcli con mod "Wired connection 1" ipv6.addr-gen-mode eui64 ipv6.token ::40e1:6f85:bed:dc0b`.
3. Caddy listens on 8443, since only root may use 443; the firewall passes 443
   on, with a separate rule for IPv6. `./llm web` prints the commands when
   either is missing.

`./llm status` warns when the name points over IPv6 at another machine.

The key is all that stands between the internet and the models, and it has
no limits: do not hand it out. To replace it, delete
`~/.config/local-llm/api-key` and run `./llm gateway`.

`panel/` is a GNOME Shell extension, symlinked into
`~/.local/share/gnome-shell/extensions/local-llm@nicolas`. It adds a button
to the top bar that shows how many requests are in the models right now; its
menu has graphs of tokens per second, requests and GPU load, the GPU's
memory, power and temperature, a switch per model (with why it failed, if it
did), and the URLs and key to copy.

`monitor.py` measures power, energy and cost, GPU load and the models' load
every 2 s, for the panel and the chat page's graphs (the chart button, top
right). `./llm install` installs it once as a user service that runs from boot.
Energy and cost are kept per day in `~/.local/state/local-llm/energy.json`,
across reboots: the page shows the totals and a bar per day for the last 30
days, the panel today and the total. It is a plain JSON file rather than
OpenTelemetry, which would need a collector and a time-series database to keep
one running total; should a dashboard like Grafana ever want the numbers, the
monitor can expose them as Prometheus metrics.

- Energy comes from the chip's RAPL package counter (CPU, GPU, memory
  controller), made readable for users by a udev rule in
  `/etc/udev/rules.d/70-rapl-readable.rules`:
  `SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chmod o+r /sys%p/energy_uj"`.
  Linux locks these counters because reading them fast can leak secrets
  (PLATYPUS, CVE-2020-12912); delete the rule to undo. Without it the monitor
  samples the chip's power reading instead. Either way it is the chip's own
  estimate: the computer draws more at the wall, so energy and cost are a
  lower bound. Time while the monitor is stopped is not counted, and neither
  is the computer while it is off (or asleep), which draws a few watts.
- The chat page splits the power into GPU, CPU cores and the rest of the chip
  (memory controller, data links, I/O). GPU and cores come from the chip
  firmware's `gpu_metrics` table (format 3.0; other formats get no split); the
  core figure is the firmware's activity-based estimate. The rest is the RAPL
  total minus both, so the parts always add up to it.
- Memory is all installed memory (system RAM plus the VRAM carve-out), split
  into what the GPU holds (models and their caches) and everything else. What
  the GPU borrows from RAM (GTT) is counted once, under the GPU.
- The price is the German day-ahead market price for the current hour
  ([aWATTar](https://api.awattar.de/v1/marketdata); Esslingen is in the one
  German market zone) plus what a household in Esslingen pays per kWh on top,
  with 19 % VAT: the Netze BW grid fee
  ([2026 prices](https://assets.ctfassets.net/xytfb1vrn7of/7eQvxehZzn3ECbR9rALmyD/ecc795b9dcd666ce1f53d9d04362a321/netzentgelte-strom-netze-bw-gmbh-2026.pdf)),
  the concession fee for a town of Esslingen's size, electricity tax and the
  2026 levies ([overview](https://www.wattline.de/energiewissen/strom-umlagen/)).
  Monthly base fees are left out, since they do not grow with use. The
  numbers are in `SURCHARGES_CT` in `monitor.py`; check them each January.

## When things go wrong

Inkling 2 Mini runs on llama.cpp, from an image that `./llm up` builds on first
use (`Containerfile.llama-cpp`: upstream plus a branch that fuses the model's
many tiny GPU calls, both pinned by commit). llama.cpp takes what a model needs
and ignores the memory share. Started at its
262k context on 2026-11-21, it went past 44 GB where llama.cpp's own projection
said 32, ran the computer out of RAM, and the kernel's OOM killer ended it four
times in a row. Its context is now 65k, where it was measured (26.6 GB at the
peak of a 60k-token prompt). `./llm up` counts that measurement against the
share, and refuses a llama.cpp model started with a context, slot count or
batch size other than the measured ones. A container killed without a word in
its log now shows "killed (SIGKILL), most likely for lack of memory" in status,
the panel and the chat. For llama.cpp, status shows the first error of the last
start, which names the cause, rather than the last line that only says loading
failed. Before you raise the context, measure it: start the model by hand with
the new `--ctx-size`, watch `mem_info_gtt_used`, then update `context` and
`measured_gib` together. Upstream ggml-org/llama.cpp#32752 should cut its 6.5 GB
working memory. Thinking cannot be turned off: `chat_template_kwargs.reasoning_effort`
(0.0 to 0.99) only shortens it. llama.cpp keeps no count of finished requests,
so this model's use is not in `./llm usage`.

Granite's symmetric GPTQ checkpoint exposed a Strix runtime bug on 2026-11-12:
AutoGPTQ registered packed `qzeros` even though the RDNA kernel configuration
declared that no zero-point tensor was needed. The native post-load conversion
left that unused tensor untouched, then passed its packed `[32, 768]` layout to
an operation expecting expanded `[6144, 32]` zeros. `local_llm_rdna.py` clears
only that unused symmetric parameter after native conversion. It leaves native
forward dispatch and asymmetric zero points unchanged. Regression checks cover
the actual Granite QKV factory and asymmetric conversion on meta tensors.
Images advertise the tested fix as `local-llm.rdna-symmetric-gptq=1`; `./llm up`
refuses an image missing it before loading weights. Existing running models
keep their original image until restarted.

The first patched launch then exposed stale AOT compilation: vLLM loaded the
earlier graph, whose input list still expected the removed tensor, and failed
with `Expected tensors only, but got NoneType`. Runtime feature versions now
select a separate compile-cache namespace. Old cached graphs are preserved,
and a future compatibility-version change cannot replay them accidentally.
The existing crash checks expose both failures in status, the panel and chat.

The native Olmo reasoning parser inherited an unimplemented token counter and
reported zero reasoning tokens. Its nonstream parser also exposed an unfinished
thought as the final answer when generation reached its token cap. The Olmo-only
plugin counts its multi-token delimiters and keeps unfinished thoughts separate,
matching streaming. Real-tokenizer regression cases cover both closing patterns,
empty thoughts, incomplete markers and capped outputs. Images advertise
`local-llm.olmo-reasoning=3`; startup and health checks refuse an image without
it. Version 2 also handles a closing bracket merged with following newlines
by BPE tokenization; the native three-token check missed that actual chat
output, affecting counting and structured-output activation. Version 3 also
handles a closing bracket merged with a JSON prefix during grammar advancement,
without changing generated tokens. Olmo's benchmarks
use Ai2's recommended temperature 0.6 and top-p 0.95.
Earlier greedy runs that returned only thinking are retained as rejected runs,
with zero useful-answer throughput.

MiniCPM5 2B first named its tool parser `minicpm5xml`, a name from its own
documentation that this vLLM does not have. vLLM started anyway, then failed
its first config check and restarted in a loop. `./llm up` now reads the
parser names the image has (from vLLM's source, since importing vLLM needs a
GPU) and refuses a model whose parser is not among them.

Mistral Small 3.2 and Ministral 3 crashed on 2026-11-16 in two ways. First,
vLLM 0.29 imports Transformers 4's Pixtral image rotation, which Transformers 5
renamed and changed: positions are now (row, column) pairs instead of one flat
index. `pixtral-positions.patch` uses the new rotation with pairs; a build check
proves it rotates every image patch exactly as Transformers 4 did. Images
advertise it as `local-llm.pixtral-positions=1`, and `./llm up` refuses those
models on an image without it. Second, with `tekken.json` in the checkpoint,
Transformers 5 loads Mistral's own tokenizer, which reads `[IMG]` as text, so
the image processor finds no image token and the server crashes while
starting. The download skips `tekken.json`: the standard tokenizer then gives
the same ids as Mistral's own (checked on mixed text, code, numbers and
emoji). A copy that still has the file is refused before start, with its path.

Nanbeige 4.3 33B's weights fit its first share, but vLLM then refused to
start: one 32k-token request needs 6.5 GiB of KV cache and only 1.7 GiB were
left. `./llm up` now adds that cache, worked out from the checkpoint's config,
plus 1.5 GB (what every start that day used beyond weights and cache), and
refuses the share before loading. Hybrid and sliding-window models keep less
per token than that formula, so the check leaves them to the old 4 GB room.

A model on the image built on 2026-11-12 leaves the GPU at 100% activity
while it has no requests. It was first seen with Granite and put down to that
model; on 2026-11-14 Gemma 4 12B and 26B did the same, and the benchmark suite
found no quiet moment to measure in. An idle Gemma 4 26B held the machine at
about 36 W. With `GPU_MAX_HW_QUEUES=1` instead of the runtime's default of
four it idles at 2% GPU and about 15 W, and answers at the same speed
(40.9 tokens/s either way, all 16 suite cases pass). `llm` now starts every
model with one queue; all nine idled at 7% GPU or less with it. A model started
before that keeps its setting until it is restarted (the Omni started on
2026-11-10 runs the older image, which idles quietly either way); the
idle-GPU warning names such a model and says how to restart it. Inspect GPU
clients and logs first; the warning alone does not establish a GPU reset or
justify restarting the desktop.

Qwen3.8 can generate with its own draft head (`--speculative-config`, see
[the comparison](benchmarks/all-models-2026-11-14.md)). On 2026-11-15 two
ways of asking for it crashed the engine three times each, minutes into
loading: with the usual `--prefix-cache-retention-interval 1568` (speculative
decoding makes the scheduler's block 1600 tokens, and the interval must be a
multiple), and with a schedule that switches it off for some batch sizes (an
assertion while capturing graphs). `./llm up` now refuses both before it
loads anything, and says which; the crash itself showed in `./llm status`
with its reason.

`health.py` (run by the monitor) watches for the failures seen so far and
lists them in `./llm status`, the panel (top of the menu; the icon turns into a
warning) and the chat page (a dot on the chart button):

- the GPU hanging and being reset (kernel log), which kills models and desktop;
- a model that crashed, with the reason from its log;
- a model stopped from outside, e.g. by logging out: without
  `loginctl enable-linger` a logout ends every model;
- a model server listening on the network: with host networking vLLM put its
  engine's internal sockets (no key) on the network card's IPv6 link-local
  address, which the firewall's default zone lets every computer in the
  network reach (found by the red team, 2026-10-29). `llm` now sets
  `VLLM_HOST_IP=127.0.0.1` and `GLOO_SOCKET_IFNAME=lo`, and the check is an
  error naming the port;
- a container whose monitor process (conmon) lives inside whatever started it.
  podman leaves it in the caller's cgroup, so models started from the panel
  lived inside GNOME Shell and the gateway inside the terminal app that ran
  `./llm` (found 2026-10-29); when that ended, so did they, or podman lost
  track of them. `llm` now starts every container in a scope of its own
  (`systemd-run --scope`, in `local-llm.slice`), and the check names any that
  are still elsewhere;
- a container podman lost track of: `podman system reset`, even one pointed
  at other storage with `XDG_DATA_HOME`, deletes the runtime files of running
  containers (it happened at 2026-09-28T14:14). They keep serving, but a crash
  would not be restarted, and stopping them takes two tries, which `./llm`
  does on its own;
- a start taking far longer than that model's last one (kept in
  `~/.local/state/local-llm/starts.json`);
- a model answering for over an hour without a moment idle, which is how a
  request stuck in a loop looks from outside. On 2026-10-29 two requests
  without streaming and without `max_tokens` kept Gemma 4 12B generating for
  over 1.5 hours after their client (a Mac in the network) had given up:
  vLLM sends nothing until the answer is complete, and the gateway did not
  notice the client leaving before that. It does now, and stops the request;
  `./llm gateway` ends any such request still running. Clients should still
  set `max_tokens` (or stream), so a loop ends on its own;
- the GPU busy for minutes with nothing to do, naming a model that was
  started without the one-queue setting (above);
- the public name pointing over IPv6 at another machine: the router's MyFRITZ
  name sent browsers at home to the FRITZ!Box login (2026-11-09), see "From
  the internet";
- RAM or swap nearly full, the GPU allowed to borrow more than the RAM there
  is, and a hot chip.

The gateway answers requests for a model that is starting, crashed or off with
a 503 saying so (with `Retry-After` while it starts) instead of a failed
connection, and logs each refusal with the model, the client and the reason
(`./llm logs gateway | grep Refused`). Crashed models restart on their own up to three times. `./llm up`
also refuses a model whose memory is not free in RAM, and one that is not
fully downloaded: vLLM runs offline, so it would fail minutes in and restart
three times (it did, 2026-10-29). At boot, one model that cannot start does not
keep the others from starting.

Caddy serves HTTP/1.1 and HTTP/2 only. With HTTP/3 on, it told clients to
switch to QUIC over UDP, but only TCP 443 is passed on, so the UDP path half
worked: small requests got through, a dictation's megabytes from a Mac app
hung for 45 s and failed (2026-11-10). Browsers fall back to TCP; macOS's
URLSession does not. Caddy's log showed it as QUIC `timeout: no recent
network activity`.

`llm` runs everything from a `main` called on its last line: bash reads a
script while running it, so editing `llm` during a long `./llm up` used to
make the running copy misread the new file and fail.

The GPU hang behind these checks: the kernel resets any GPU queue whose job
waits longer than `amdgpu.lockup_timeout` (2 s by default on kernel 6.19).
A model's step is hundreds of short GPU calls, and the desktop gets its turn
between them, so a long step is only slow. What resets the GPU is a single
call that runs for seconds and keeps everything else off the GPU, such as
one attention call over 8192 prompt tokens (the old default step size) or
over a very long context. Prefill steps are therefore capped at 512 tokens.
A longer timeout (`sudo grubby --update-kernel=ALL
--args="amdgpu.lockup_timeout=10000"`) turns such a call into a frozen
screen instead of a reset, at the price of real hangs taking longer to
recover from.

vLLM's attention kernel on this GPU reads the whole context again for every
two prompt tokens, so its single calls grow with the context: with Gemma
4's 512-wide heads a step took 0.1 s longer per 1,000 tokens, and one call
passes 2 s somewhere beyond 64k. `vllm_plugins/` holds the fixes, built into
the image:

- a vLLM plugin that computes Gemma's prefill attention as chunked matrix
  multiplies instead, each call a few milliseconds (tested against vLLM's
  kernel by `vllm_plugins/test_attention.py`). Gemma has its full 262k: a
  255k-token prompt reads in about 12 minutes. Answers come at ~32 tokens/s
  at short context, ~11 at 128k and ~6 at 255k. If a vLLM update moves the
  function it replaces, the model refuses to start rather than running its
  long context on the slow kernel;
- a patch giving vLLM's Triton kernel larger tiles for Qwen's 256-wide heads:
  its attention calls stay near 0.5 s even at 255k. Qwen has 262k too, but
  reads slowly: ~3.5 s per step at 64k and ~9 s at 255k, so a 255k-token
  prompt takes the better part of an hour. The image build fails if the
  patch no longer applies.

Long contexts rely on the prefix cache: a chat sends its whole history every
turn, and only what is new should be read. Gemma's sliding-window layers
and Qwen's recurrent layers can only resume from a kept checkpoint, and by
default vLLM keeps one only where the previous request ended. A chat turn
diverges earlier, where the last answer began (thinking is left out of the
history), so every turn was read from scratch. `--prefix-cache-retention-interval`
keeps a checkpoint every sliding window (Gemma, 1024) or cache page (Qwen,
1568). With it, the second turn over a 20k-token document read 20,000 tokens
from cache on Gemma (3 s instead of 21) and 18,816 on Qwen. Kept
checkpoints are evicted like any cached block, so they reserve no memory;
the cache holds ~340k (Gemma) and ~385k (Qwen) tokens, so a second long
conversation pushes the first one out.

Benchmarks must not run vLLM's own attention kernel on contexts past 64k: a
single call on 255k ran long enough (2026-09-27T19:22) that the kernel reset
a VM's and the browser's GPU queues. Compare against it at short context, as
`vllm_plugins/test_attention.py` does, and time only the fixed paths beyond.

## Storage and observed model use

The 2026-11-12 storage migration keeps active weights on SSD and uses the
4 TB Toshiba HDD as an XFS archive. Horscht is now at
`~/.local/share/local-llm/archive/nicolas/Projects/horscht`; its original project
path is a symlink. Its files, Git state and virtual environment were checked
before removing the verified internal copy, freeing 331.4 GiB. The three new
pinned checkpoints occupy about 50.2 GiB under
`~/.local/share/local-llm/models` on the internal NVMe. Their temporary T7 copies
were removed only after full checksum verification and stopping our test models.
The T7's existing datasets and games are unchanged.

The original running Omni container has not been restarted. Its next natural
start will use a separately verified native SSD copy under
`~/.local/share/local-llm/hot-huggingface`, selected by
`~/.config/local-llm/model-caches.json`. Qwen 3.8, the three Gemma models and
Muse now use verified current snapshots in
`/run/media/nicolas/T7/local_llm/warm-huggingface` (86.17 GiB). Their snapshot
files are ordinary files because exFAT cannot store the Hub's symlinks. The
full original Hugging Face cache, including older revisions, is archived on
the HDD; `~/.cache/huggingface` points there. The archive migration manifests,
full checksum receipts and offline config/tokenizer checks are under
`logs/storage-2026-11-12/`.

After verification, only archived inactive native replicas were removed,
freeing another 198.15 GiB. The original Omni container still has its original
cache root inode and entire repository at
`~/.cache/.huggingface-active-original-2026-11-12`; keep that temporary copy
until the original container naturally stops. No unique checkpoint was deleted.

| Drive | Current role | Available after migration, 2026-11-13 |
|---|---|---|
| Internal NVMe | Three additions, Omni hot cache, runtime/kernel caches | about 467 GiB |
| T7 SSD | Five existing serving snapshots; existing datasets/games | about 920 GiB |
| Toshiba HDD, XFS | Horscht and full Hugging Face archive | about 3.0 TiB |

The HDD's 71.34 GiB difference between free extents and user-available space
is XFS metadata reservation, verified against allocation counters; it is not
missing project data. Both removable drives currently use desktop mounts.
Making them available before login still requires administrator rights:

    sudo python3 /home/nicolas/Projects/local_llm/logs/storage-2026-11-12/prepare-archive.py

That prepared finalizer only adds UUID-based boot mounts and updates archive
aliases/guards; it never formats a drive or moves model/project files. Until
then, mount the HDD before using Horscht or the cold cache, and the T7 before
starting its five models. The native additions and Omni hot cache are independent
of those mounts. Missing or wrong drives are refused before loading.

Keep frequently used models on NVMe, less frequent active models on the T7,
and inactive checkpoints on the HDD. Disk speed chiefly affects loading;
weights already resident in unified memory do not become faster merely by
moving their files to a faster disk. These runs did not isolate cold disk load
times. Allow roughly 20–35 GiB per future 27–32B INT4 checkpoint plus 25 GiB
free download space; BF16 originals and conversion experiments need much more.
The exact future Mistral/model releases still need checkpoint-specific sizing.
Use several weeks of ordinary usage before promoting or archiving models.

`./llm usage` (or `./llm usage --json`) reports observed requests, input/output
tokens, sampled active seconds and last use per model. The monitor keeps the
totals in `~/.local/state/local-llm/model-usage.json`, with UTC daily records.
It never stores prompts, answers, client identities or keys. Counting starts
with the first observed counter sample; old backend totals are not invented
as historical usage. Restarts preserve the ledger and establish a fresh
baseline when the backend changes. Active time is sampled wall time, not GPU
time, and long monitor gaps are not counted as continuous activity. These
totals include benchmark requests; use the retained benchmark reports when
judging ordinary use before deciding which models deserve SSD space.

The three additions in `model_catalog.py` pin exact checkpoint revisions.
`~/.config/local-llm/model-dir` selects their ordinary checkpoint directory,
separate from the existing Hugging Face cache. Inspect sources, revisions,
quantization and capabilities with `./llm config`. Download only the models
wanted, or check capacity first without transferring weights:

   ./download.sh --check granite-4.2-30b nemotron-3.5-lightning-30b-a3b olmo-3.1-32b-think
   ./download.sh granite-4.2-30b nemotron-3.5-lightning-30b-a3b olmo-3.1-32b-think

Downloads resume, require the remaining file sizes plus 25 GiB of free space,
and refuse a missing or wrong configured drive before touching its model
folders. The public additions download anonymously so an expired OAuth login
from another project cannot break them; existing gated checkpoints still use
the local Hugging Face login. `~/.config/local-llm/storage.json` records each
external storage path, its mount location and expected filesystem UUID.
The monitor reports missing/wrong drives and warns below 25 GiB free (an error
below 4 GiB). An off model's storage reason also reaches API requests as a
503, instead of suggesting that starting it would fix an absent drive.

## Measuring and improving speed

`benchmark.py` measures the time until the first generated token, total time,
generation speed and aggregate throughput through the actual streaming API.
It uses Python's standard library and reads the existing API key without
printing it. It never starts or stops a model:

    python3 benchmark.py qwen3-omni-30b
    python3 benchmark.py qwen3-omni-30b --concurrency 4
    python3 benchmark.py qwen3-omni-30b --image screen.png --audio question.wav \
      --prompt 'Answer the spoken question about this picture in one sentence.'
    python3 benchmark.py qwen3.8-27b --no-thinking

Use `--output results.json` to keep answers and measurements. One warmup is
reported separately, followed by three measured repetitions. These repeat
the same input: the prefix and media caches can make repetitions much faster
than a fresh prompt or screenshot. For a fresh-media measurement, use a new
file and `--warmups 0 --repeats 1`. `finish_reason: length` means the answer
hit the token limit (128 by default). Aggregate throughput counts output
tokens over the whole measured batch, including the wait for the first token;
it is different from one request's generation speed. Concurrent work and
other applications sharing the GPU can affect both. Timing starts at the HTTP
request; local file loading and recording time are outside the measurement.

`benchmarks/added_models.py` is the suite all nine models are compared on
(the name is from when it covered three). It checks answers as well as speed:
exact replies, separated reasoning, strict JSON, tool calls, lookups in a 4k
and a 24k prompt, and it measures what a long conversation costs: reading a
24,000-token prompt, and generating behind it. Like `benchmark.py` it never
starts or stops a model, waits for a quiet GPU, and discards a trial another
client overlapped. Its requests are left out of `./llm usage`.

    python3 benchmarks/added_models.py qwen3-omni-30b in-place --case extended \
      --keep-going --output logs/omni.json
    benchmarks/suite_one.sh gemma-4-31b default logs/suite
    python3 benchmarks/suite_table.py logs/suite/*.json

`suite_one.sh` is for a stopped model: it starts it, records what it draws
while idle, runs the suite and stops it again. Two variables try a setting
for one start without touching a default, also through `suite_one.sh`:

    LOCAL_LLM_EXTRA_FLAGS="--attention-backend TRITON_ATTN" ./llm up muse-glimmer-30b
    LOCAL_LLM_COPY=qwen3-omni-30b ./llm up qwen3-omni-30b-copy

The second starts a copy of one of the six original models ten ports up
(8016 for Omni), so a setting can be measured on Omni while the Omni in use
keeps running; `LOCAL_LLM_COPY=qwen3-omni-30b ./llm stop qwen3-omni-30b-copy`
ends it. [All nine models on one suite](benchmarks/all-models-2026-11-14.md)
has the comparison, what it changed and what it ruled out.

On 2026-11-02, Omni's startup log exposed a missed optimization: vLLM looked
for `AMD_Radeon_8060S` expert tuning, while its own shipped profile was named
`Radeon_8060S_Graphics`. `vllm_plugins/alias_moe_config.py`, run by the image
build, makes that profile reachable under the current name. A newer image's
native profile takes precedence; a missing legacy source fails the build
instead of silently discarding the optimization. The profile is the
[upstream 8060S Omni tuning](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/configs/E%3D128%2CN%3D768%2Cdevice_name%3DRadeon_8060S_Graphics%2Cdtype%3Dint4_w4a16.json).

`benchmarks/moe_compare.py` compares the generic settings with that profile
using Omni's actual expert dimensions, quantization and routing, synthetic
inputs, and a separate GPU graph for each candidate. The optimized outputs
matched the original within BF16 tolerance at batches 1, 2, 4, 8, 16, 32,
128 and 512. Median expert-layer times on this box:

| Tokens in the step | Generic | Tuned | Time saved |
|---|---:|---:|---:|
| 1 (one chat decoding) | 0.235 ms | 0.188 ms | 20% |
| 4 | 2.261 ms | 0.813 ms | 64% |
| 32 | 5.809 ms | 2.932 ms | 50% |
| 512 (prefill chunk) | 6.889 ms | 5.668 ms | 18% |

These are expert-layer timings, not whole-model speedups. The shipped file
only covers batches through 32; its nearest entry also improved 128- and
512-token steps in this comparison. The safety cap of 512 prompt tokens per
step stays in place. To repeat the kernel comparison inside a running model
container (about 0.4 GiB of temporary GPU allocations, without stopping it):

    podman cp benchmarks/moe_compare.py llm-qwen3-omni-30b:/tmp/moe_compare.py
    podman exec llm-qwen3-omni-30b python3 /tmp/moe_compare.py

A controlled full-server comparison then toggled only the tuning profile on
the same temporary Omni container: 0.25 GPU share, 8192-token context limit,
32 concurrent sequences, 512-token prefill chunks, same synthetic prompts.
The production model containers kept running. Median results (three single
requests, eight requests at concurrency four, after warmup):

| API measurement | Generic | Tuned |
|---|---:|---:|
| One chat, generation | 43.1 tokens/s | 51.5 tokens/s (+19%) |
| One chat, first token | 199 ms | 90 ms |
| Four concurrent requests, aggregate output | 59.6 tokens/s | 135.3 tokens/s (2.27x) |
| Cached screenshot + spoken question, total | 0.598 s | 0.484 s |

The image/audio answer was correct with both settings. A fresh 3732-token
inventory prompt took about 3.3 s to its first token with either setting;
there is no established improvement for fresh-media processing or long
prompts. Media warmups include first-use imports and kernel work, so they
are not comparable to cached repetitions. These small benchmarks establish
the short-context gain, not a guarantee across workloads or the full 65k
context. Raw reports from this run are in `logs/performance-2026-11-02/`;
`tuned-*` versus `untuned-control-*` is the controlled comparison, while
`current-*` measures production with different capacity settings.
Loading the temporary duplicate also pushed inactive pages into zram (7.7 GiB
logical, about 2.6 GiB compressed RAM at cleanup). Removing it restored about
41 GiB available RAM and zero current memory pressure, but swap-use warnings
can remain while those pages stay compressed. For future full-server A/B
tests, prefer an idle maintenance window with one model instance: a GPU share
that fits does not account for all checkpoint-loading buffers. The API-only
benchmark does not load a duplicate.

Rebuild with `podman build -t localhost/local-llm:latest .`; running model
containers pick up the tuning when next restarted from the website or panel.
A rebuild alone does not change their already captured GPU graphs.

Checking the other configured models found one more match: `gemma-4-26b`
uses the same Triton INT4 expert kernel, but needs an entirely new profile
for its 704-wide experts. Its FP16 activations, GELU-tanh activation and
2816-wide hidden state differ from Omni. The four dense models do not read
these expert profiles:

| Model | Does this optimization apply? |
|---|---|
| `gemma-4-26b` | Yes: local 8060S profile, installed by the image build. |
| `qwen3.8-27b` | No: dense matrix kernels, not expert kernels. |
| `gemma-4-12b`, `gemma-4-31b` | No: dense matrix kernels. |
| `muse-glimmer-30b` | No: dense matrix kernels. |

The initial test transferred Omni's tile settings to Gemma's own shapes:
small steps improved, but the 512-token prefill step became 13% slower.
The local profile in `vllm_plugins/moe_configs/` therefore uses those settings
only through 40 tokens. Adjacent keys at 40 and 41 prevent vLLM's nearest-key
selection from spilling the decode settings into larger steps. Every step
from 41 onward uses the same generic parameters as before. Tests cover that
boundary and preserve a newer image's native Gemma profile if one exists.

On 2026-11-02, synthetic Gemma expert-layer tests checked finite outputs and
FP16 agreement (rtol/atol 0.005) at steps 1, 2, 3, 4, 8, 12, 16, 24, 32,
40, 41, 48, 64, 128, 256 and 512. A second seed with reversed timing order
confirmed the small-step improvement. First-run median expert-layer times:

| Tokens in the step | Generic | Gemma profile | Time saved |
|---|---:|---:|---:|
| 1 | 0.255 ms | 0.224 ms | 12% |
| 3 | 0.877 ms | 0.566 ms | 36% |
| 4 | 1.251 ms | 0.950 ms | 24% |
| 16 | 3.821 ms | 2.816 ms | 26% |
| 40 | 4.818 ms | 3.541 ms | 26% |

These are kernel measurements, not measured gains in full-model tokens/s.
Those component tests used tensors alongside the running servers. The later
full-model validation below restarted Gemma with this profile installed.
Raw data and the model audit are in `logs/performance-2026-11-02/gemma/`.
After rebuilding and restarting Gemma, repeat the check with:

    podman cp benchmarks/moe_compare.py llm-gemma-4-26b:/tmp/moe_compare.py
    podman exec llm-gemma-4-26b python3 /tmp/moe_compare.py --model gemma-4-26b

Before restarting, copy the local profile into `/tmp/gemma-moe.json` in that
container and pass `--profile /tmp/gemma-moe.json`. `--seed 1729 --reverse`
checks another routing pattern and reverses candidate order.

The dense models select their RDNAHybridW4A16 matrix tiles directly by GPU
architecture and matrix dimensions, without a GPU-name JSON lookup. Renaming
or copying expert profiles cannot speed them up. Further kernel work now adds
a measured path for Gemma 31B's matrices; the other dense models keep their
native kernels after the tested variants regressed them.

The image also includes new kernel code in `vllm_plugins/`, beyond tuning
profiles: two fused vector kernels for single-token Omni/Gemma 26B experts,
wider quantization-group tiles for specific Gemma 31B matrices, and one host
attention-bound preparation per Omni media encoder stack. These paths have
strict shape/precision/feature guards and native fallbacks. See
[the kernel experiments](benchmarks/README.md) for algorithms, numerical
checks, measured wins and rejected cases. The full-model comparison below
measures their combined effect on Omni and Gemma 26B.

`LOCAL_LLM_FAST_KERNELS=0 ./llm up NAME` starts a stopped model without these
three new hooks; the existing attention protections and expert profiles still
apply. `llm` passes this startup setting into the container. Use it if a runtime
update makes a new hook incompatible; the monitor reports server failures
through the existing crash checks. Changing the environment does not alter
an already running model or its captured graphs.

A full-model comparison on 2026-11-02 restarted Omni and Gemma 26B one at a
time, first with tuning profiles only, then with the new hooks. Context
limits, prefill limits, GPU shares and KV cache capacities stayed the same.
Median single-chat generation rates (three measured requests after warmup):

| Model | Profiles only | New kernels | Retained default |
|---|---:|---:|---|
| Omni | 51.0 tokens/s | 63.5 tokens/s (+25%) | New kernels |
| Gemma 26B | 41.2 tokens/s | 40.6 tokens/s | Profiles only |

Omni's fresh screenshot plus spoken question finished in 1.37 s versus
1.54 s (11% less time). Four-request throughput and fresh long-prompt
processing showed no substantial additional gain. Both modes passed the
arithmetic, JSON, tool-call, inventory and media checks. The 120 measured
requests had no overlap with other model requests; ordinary desktop work
continued, so small timing differences remain noisy.

Gemma's text recheck reached 42.0 tokens/s, still too close to its baseline
to establish a serving improvement. Its expert-layer advantage shrank sharply
after evicting cache before each measurement. Repeating one layer with warm
weights overstates the benefit available while a full model cycles through
many layers. Gemma therefore defaults to profiles only on future starts;
`LOCAL_LLM_FAST_KERNELS=1 ./llm up gemma-4-26b` explicitly enables the
experimental hooks on a stopped model. `./llm config` reports the startup
choice as `fast_kernels` (0 or 1), including an environment override. See
[the serving comparison](benchmarks/README.md#full-model-serving) for the
complete measurements and limits.

The next input-focused pass kept the image/audio samples and serving capacity
unchanged and limited Omni's CPU preprocessing to one thread. On five fresh
requests per case, the wait for a first response fell from 1100 to 1012 ms
for a screenshot, 209 to 169 ms for a roughly four-second microphone-style
Opus clip, and 342 to 276 ms for a roughly fifteen-second clip. A screenshot
plus spoken question fell from 1124 to 1023 ms. This setting is now Omni's
startup default; it is separate from `LOCAL_LLM_FAST_KERNELS`. See
[the input measurements](benchmarks/README.md#omni-image-and-voice-input)
for cache checks, quality limits and the kernel experiments that were not
enabled after testing.

For dictation, measure the complete result as well as the first token.
`whisper-local` waits for the full response, processes jobs in order, and
waits for the dictation key to be released before pasting. A recording can
therefore delay an already-finished earlier result. A broader suite on
2026-11-03 checked these client waits and repeated text, differently sized
images, minute-long audio and simultaneous requests. See
[the consistency analysis](benchmarks/README.md#omni-consistency-and-dictation)
for observed ranges and the difference between model time and client waiting.

For further work, the promising directions are:

- Crop a screenshot to the relevant window or region before sending it. Keep
  readable text within the pixel budget. Reducing the whole image to 400k
  pixels also reduced first-token time in a synthetic screenshot test, but
  can lose fine print, so the 1M-pixel default stays.
- Reuse identical image/audio bytes for follow-up questions. In the synthetic
  screenshot test, the first 1M-pixel request took 4.26 s until the first word;
  repeating it after the cache was populated took 0.21 s. That combines
  media and prefix-cache savings, not faster fresh-image encoding.
  [vLLM also accepts media UUIDs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/#cached-inputs)
  to avoid resending cached payloads, but a client must retry with the full
  media on cache misses or after a model restart.
- Trim silence around recorded speech, keeping a margin so words are not
  cut off. This and capturing only the relevant screen region belong in the
  recording client; neither is implemented here.
- [N-gram speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/)
  was tried on a copy of Omni on 2026-11-14 and made it about half as fast,
  also when copying text from the prompt (29 against 54 tokens/s): a step
  that checks several guessed tokens costs this model far more than a step
  for one. It is not a direction for Omni.

## Notes

The gateway is only a router: vLLM itself serves both API styles, so requests
are forwarded untouched, by the `model` field.

The models call tools, which is all MCP needs: the MCP client (an agent,
Claude Code) turns an MCP server's tools into ordinary tool definitions.
Tested on both APIs, also streamed: single and parallel calls, picking the
right tool out of 41, tool results with pictures (Gemma) or 100k tokens, and
the Responses API. Tool definitions sit at the start of the prompt, so the
prefix cache keeps them between turns. Forcing a call (`tool_choice`
`"required"` or a named tool) works on Qwen through vLLM's grammar for its
call format; for Gemma, vLLM has none and silently ignored the force, so
`vllm_plugins/local_llm_tools.py` adds one. `muse-glimmer-30b` has the same gap
and no fix yet: its parser treats `"required"` and a named tool as `"auto"`,
so it may answer in text instead (a named tool it did call, tested 2026-10-29).
The two Mistral models had the force turned into a bare JSON array that their
parser does not read, so the call came back as text; the same plugin forces
their own `[TOOL_CALLS]name[ARGS]{...}` with the tool's schema (tested
2026-11-16, streamed and not).
With `tool_choice` `"none"` the models
do not see the tools at all: otherwise they still tried to call one, and the
answer came back empty. That turn misses the prefix cache.

`gpu-memory-utilization` per model is a share of *total* GPU memory, so the
shares in `llm` must add up to less than 1. `./llm up` refuses a model whose
weights (plus a few GB to work in) do not fit its share, or that would push
the running shares past 0.95, and says why.

First start of a model compiles kernels and can take the better part of an
hour. `~/.cache/vllm` and `~/.cache/llm-kernels` keep the result, so later
starts are faster.

Models start one after another, even when started from separate commands:
vLLM sizes its cache from the memory that is free while it starts, so two
models loading at once each see too little.

The GPU borrows system RAM via `ttm.pages_limit` on the kernel command line,
with the BIOS VRAM reservation kept at its minimum. Together they decide how
much memory all models share.

## Not here yet

* K2 Horizon (the faster, smarter pick) has no working vLLM implementation.
  Watch <https://github.com/vllm-project/vllm/pull/53806>; swapping it in is
  one line in `llm`.
