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
50 MB) for the models that see, and sound for `gemma-4-12b`; with another
model selected it says so instead. The desktop chat window, which also keeps
past chats, is its own project, local_llm_chat.

| Model | Role |
|---|---|
| `qwen3.8-27b` | the smart one: hard tasks, batch jobs. Slow per request (dense). |
| `gemma-4-26b` | the fast one: agents and high-volume work; sees pictures and videos. ~39 tokens/s. |
| `gemma-4-12b` | the one that hears: speech plus a screenshot in one request, no transcription model in between. ~25 tokens/s. |
| `gemma-4-31b` | Gemma's largest, dense: for when the 26B is not good enough. ~10 tokens/s, reads ~170 tokens/s; 131k context. |
| `muse-glimmer-30b` | Meta's open agent model, dense; always thinks first, sees pictures. ~14 tokens/s, reads ~300 tokens/s; 131k context. |

Not all of them fit at once (their memory shares are in `llm`); start the ones
you need from the website, the API, the panel or `./llm up`.

### Sound

`gemma-4-12b` takes audio in OpenAI chat requests (`/v1/chat/completions`),
as a content part after the text:

    {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}

or `{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}`.
WAV, FLAC, MP3, M4A, OGG and WebM all work (tested). Each clip may be at most
30 s: vLLM cuts longer ones without an error, so split long recordings. The
Anthropic API (`/v1/messages`) has no audio block, so it refuses audio. Google's
prompt for transcribing works best; without it the model sometimes added a
translation nobody asked for:

    Transcribe the following speech segment in English into English text.
    Follow these specific instructions for formatting the answer: Only output
    the transcription, with no newlines. When transcribing numbers, write the
    digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.

On LibriSpeech clips it transcribed word for word apart from rare names
("Leighton" as "Leytens"), in 1–2 s. A picture and a spoken question about it
in one request were answered correctly. It is the 12B and not the larger
Gemmas because those have no audio at all; Gemma 4's E4B also hears, but is
much weaker.

The gateway listens on the local network and wants a key for everything but
the chat page and a bare `/health`. Two keys, both made on first use:

- the API key (`~/.config/local-llm/api-key`, `./llm key`) reaches the models
  and the measurements. It is the one that goes into clients;
- the admin key (`~/.config/local-llm/admin-key`, `./llm admin-key`) also
  starts and stops models. Keep it to the website and your own scripts, so a
  key leaked from some client cannot turn models off. It works only over
  HTTPS or from this computer: the gateway refuses it over plain HTTP from the
  network, and the page does not send it there.

The traffic inside the network is plain HTTP, so anyone on the same network
could read prompts and the API key; use the HTTPS name where you can. The
model servers behind it (ports 8001–8005, and their engines' internal
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
with a certificate it gets and renews itself. The name must point at the
router (a MyFRITZ or other dynamic DNS name), and the router must forward its
port 443 to port 8443 of this computer; nothing else is forwarded.
`./llm web off` closes it again. `./llm url` prints the address in use.

Inside the network the name only partly works: it resolves to the FRITZ!Box's
own IPv6 address first, and the FRITZ!Box answers that itself (its 404 page),
so browsers, which prefer IPv6, do not reach the gateway; over IPv4 (`curl -4`)
it does. Until that is solved (a local DNS entry, or the FRITZ!Box's IPv6
sharing for this computer), use `http://fedora.local:8400` at home, where
models cannot be started.

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
- the GPU busy for minutes with nothing to do;
- RAM or swap nearly full, the GPU allowed to borrow more than the RAM there
  is, and a hot chip.

The gateway answers requests for a model that is starting, crashed or off with
a 503 saying so (with `Retry-After` while it starts) instead of a failed
connection. Crashed models restart on their own up to three times. `./llm up`
also refuses a model whose memory is not free in RAM, and one that is not
fully downloaded: vLLM runs offline, so it would fail minutes in and restart
three times (it did, 2026-10-29). At boot, one model that cannot start does not
keep the others from starting.

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
