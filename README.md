# local_llm

Local models on the Strix Halo box, served through one port that speaks both
the OpenAI and the Anthropic API.

    ./download.sh        fetch the weights (resumable, safe to re-run)
    ./llm up             start the gateway and all models
    ./llm up qwen3.8-27b start only some of them
    ./llm stop <name>    stop one of them
    ./llm status         which servers answer
    ./llm logs <name>    follow one server
    ./llm key            the API key (made on first start)
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
key once and keeps it in that browser; `http://fedora.local:8400/?api=<key>`
saves it in one step (the panel's "Chat" row copies that link).

The page takes pictures and videos (PNG, JPEG, WebP, GIF, MP4, WebM, up to
50 MB) for `gemma-4-26b`, the one model here that can see; with another model
selected it says so instead. The desktop chat window, which also keeps past
chats, is its own project, local_llm_chat.

| Model | Role |
|---|---|
| `qwen3.8-27b` | the smart one: hard tasks, batch jobs. Slow per request (dense). |
| `gemma-4-26b` | the fast one: agents and high-volume work; sees pictures and videos. |

The gateway listens on the local network and wants the key from
`~/.config/local-llm/api-key`. The traffic is plain HTTP, so anyone on the same
network could read prompts and the key. The model servers behind it (ports
8001–8002) have no key and listen on localhost only; keep it that way.

`panel/` is a GNOME Shell extension, symlinked into
`~/.local/share/gnome-shell/extensions/local-llm@nicolas`. It adds a button
to the top bar that shows how many requests are in the models right now; its
menu has graphs of tokens per second, requests and GPU load, the GPU's
memory, power and temperature, a switch per model (with why it failed, if it
did), and the URLs and key to copy.

`monitor.py` measures power, energy and cost since boot, GPU load and the
models' load every 2 s, for the panel and the chat page's graphs (the chart
button, top right). `./llm monitor` installs it once as a user service that
starts at login.

- Energy comes from the chip's RAPL package counter (CPU, GPU, memory
  controller), made readable for users by a udev rule in
  `/etc/udev/rules.d/70-rapl-readable.rules`:
  `SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chmod o+r /sys%p/energy_uj"`.
  Linux locks these counters because reading them fast can leak secrets
  (PLATYPUS, CVE-2020-12912); delete the rule to undo. Without it the monitor
  samples the chip's power reading instead. Either way it is the chip's own
  estimate: the computer draws more at the wall, so energy and cost are a
  lower bound. Time before login, or while the monitor is stopped, is not
  counted.
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
- a start taking far longer than that model's last one (kept in
  `~/.local/state/local-llm/starts.json`);
- the GPU busy for minutes with nothing to do;
- RAM or swap nearly full, the GPU allowed to borrow more than the RAM there
  is, and a hot chip.

The gateway answers requests for a model that is starting, crashed or off with
a 503 saying so (with `Retry-After` while it starts) instead of a failed
connection. Crashed models restart on their own up to three times. `./llm up`
also refuses a model whose memory is not free in RAM.

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
passes 2 s somewhere beyond 64k. `attention/` holds the fixes, built into
the image:

- a vLLM plugin that computes Gemma's prefill attention as chunked matrix
  multiplies instead, each call a few milliseconds (tested against vLLM's
  kernel by `attention/test_attention.py`). Gemma has its full 262k: a
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
`attention/test_attention.py` does, and time only the fixed paths beyond.

## Notes

The gateway is only a router: vLLM itself serves both API styles, so requests
are forwarded untouched, by the `model` field.

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
