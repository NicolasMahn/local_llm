"""Compare already-running model variants without changing their serving state.

Only synthetic fixtures are sent. A report label does not change launch options.
The text cases measure total generation, including thinking; quality cases also
require useful final answers. Fresh prefill retains cache evidence rather than
treating prompt length divided by first-token latency as measured GPU throughput.

The deep cases repeat the measurement behind a long conversation. Real use of
these servers is mostly long prompts answered from the prefix cache, where
both reading and generating are several times slower than in a short chat.
"""

import argparse
import concurrent.futures
import copy
import fcntl
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import answer_token_count, contamination_reasons, parse_backend_metrics, request, require_completion_count
from model_catalog import MODELS
from model_usage import benchmark

HERE = Path(__file__).resolve().parents[1]
EXPECTED_JOB = {"job": 4711, "errors": 0}
JOB_SCHEMA = {"type": "object", "properties": {"job": {"type": "integer"},
              "errors": {"type": "integer"}}, "required": ["job", "errors"],
              "additionalProperties": False}
TOOL = {"type": "function", "function": {"name": "report_job",
        "description": "Report a job number and its error count.", "parameters": JOB_SCHEMA}}
TEXT = ("Write a detailed twenty-paragraph explanation of how a compiler works, covering parsing, "
        "type checking, optimization and machine code. Use full paragraphs.")
INVENTORY = "\n".join(f"Item {i:03}: code {i * 13 + 7}; state ready." for i in range(300))
QUALITY = ("chat", "reasoning", "json", "json_nonstream", "tool_auto", "tool_required",
           "tool_named", "tool_named_nonstream", "tool_none")
CASES = ("text", "batch4", "prefill", *QUALITY)
DEEP_CASES = ("deep_prefill", "deep_decode", "deep_lookup")
EXTRA_CASES = ("prefill_json", "reasoning_json", "prefill_ack", "batch16", *DEEP_CASES, "copy")
# The suite every model runs for a cross-model comparison.
EXTENDED = (*CASES, "batch16", *DEEP_CASES)
PREFILL_CASES = ("prefill", "prefill_json", "prefill_ack", "deep_prefill")
CONCURRENCY = {"batch4": 4, "batch16": 16}
# Editing code is mostly repeating the prompt. Speculative decoding guesses
# such text ahead, so this case shows its gain where free text shows none.
COPIED = INVENTORY.splitlines()[100:120]
# Every model fits this depth in its context, so the deep cases compare like with like.
DEEP_TOKENS = 24000
# A fresh deep read takes minutes on the dense models; it is measured once.
SINGLE_PASS = ("deep_prefill", "deep_lookup")
# The six original models are launched from `llm`, outside the catalog.
# The suite only needs to know which cases each of them can answer.
LEGACY = {
    name: {"capabilities": {"input": ["text"], "reasoning": reasoning, "thinking_toggle": toggle,
                            "tools": True, "structured_output": True}}
    for name, reasoning, toggle in (
        ("qwen3.8-27b", True, True), ("gemma-4-26b", True, True), ("gemma-4-12b", True, True),
        ("gemma-4-31b", True, True),
        # Muse takes a reasoning strength instead of a switch and thinks by default.
        ("muse-glimmer-30b", True, False),
        ("qwen3-omni-30b", False, False))
}
SUITE = LEGACY | MODELS
# A second instance started with LOCAL_LLM_COPY (see llm) answers the same cases.
SUITE |= {f"{name}-copy": profile for name, profile in LEGACY.items()}
CODE_SCHEMA = {"type": "object", "properties": {"code": {"type": "integer"}},
               "required": ["code"], "additionalProperties": False}
REMAINING_SCHEMA = {"type": "object", "properties": {"remaining": {"type": "integer"}},
                    "required": ["remaining"], "additionalProperties": False}
COUNTERS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
            "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
            "vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def metric_counters(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in COUNTERS:
            result[name] = result.get(name, 0) + float(line.rsplit(None, 1)[-1])
    return result


def metrics(url):
    with urllib.request.urlopen(url.removesuffix("/v1") + "/metrics", timeout=5) as response:
        text = response.read().decode()
    return parse_backend_metrics(text) | {"counters": metric_counters(text)}


def gpu_busy():
    values = []
    for path in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"):
        try:
            values.append(float(path.read_text()))
        except (OSError, ValueError):
            pass
    return max(values) if values else None


class Guard:
    """Observe every configured backend, including peers that start mid-trial."""

    def __init__(self, target, backends, max_idle_gpu_busy=20):
        self.target = target
        self.backends = backends
        self.max_idle_gpu_busy = max_idle_gpu_busy

    def snapshot(self):
        readings = {}
        for name, url in self.backends.items():
            try:
                readings[name] = metrics(url)
            except OSError:
                if name == self.target:
                    raise RuntimeError(f"Target backend {name} is not reachable")
        require_completion_count(self.target, readings[self.target])
        return readings

    def trial(self, call, count):
        before = self.snapshot()
        busy = gpu_busy()
        early = []
        if any(row["running"] + row["waiting"] for row in before.values()):
            early.append("a backend was busy before the trial")
        if busy is not None and busy > self.max_idle_gpu_busy:
            early.append(f"GPU was {busy:g}% busy before the trial")
        if early:
            return None, {"before": before, "idle_gpu_busy_percent": busy, "reasons": early}
        peak = {name: 0 for name in before}
        errors = []
        observed = set(before)
        done = threading.Event()

        def observe():
            while not done.is_set():
                try:
                    sample = self.snapshot()
                    observed.update(sample)
                    if before.keys() - sample.keys():
                        errors.append("a previously reachable backend disappeared during the trial")
                    for name, row in sample.items():
                        peak[name] = max(peak.get(name, 0), row["running"] + row["waiting"])
                except (OSError, ValueError, RuntimeError) as exc:
                    errors.append(str(exc))
                done.wait(0.2)

        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        value = None
        try:
            value = call()
            deadline = time.monotonic() + 3
            after = self.snapshot()
            while after[self.target]["completed"] - before[self.target]["completed"] < count and time.monotonic() < deadline:
                time.sleep(0.1)
                after = self.snapshot()
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
            after = {}
        finally:
            done.set()
            observer.join(timeout=6)
        errors += audit_reasons(self.target, before, after, peak, observed, count)
        if observer.is_alive():
            errors.append("metrics observer did not finish")
        return value, {"before": before, "after": after, "peak_active": peak,
                       "idle_gpu_busy_percent": busy, "reasons": sorted(set(errors))}


def audit_reasons(target, before, after, peak, observed, count):
    reasons = []
    if set(before) != set(after) or set(before) != set(observed):
        reasons.append("the set of reachable backends changed during the trial")
    if target not in after:
        return reasons + ["target metrics unavailable after the trial"]
    names = [target, *sorted((before.keys() & after.keys()) - {target})]
    reasons += contamination_reasons([before[n] for n in names], [after[n] for n in names],
                                     [peak.get(n, 0) for n in names], count)
    return reasons


def selected_cases(requested, capabilities):
    groups = {"all": CASES, "quality": QUALITY, "extended": EXTENDED}
    names = set()
    for name in requested:
        names.update(groups.get(name, [name]))
    return [name for name in (*CASES, *EXTRA_CASES) if name in names
            and (not name.startswith("tool_") or capabilities["tools"])
            and (not name.startswith("reasoning") or capabilities["reasoning"])]


def deep_inventory(lines):
    return "\n".join(f"Item {i:04}: code {i * 13 + 7}; state ready." for i in range(lines))


def deep_question(case, lines):
    if case == "deep_lookup":
        return f'Return a JSON object containing only the integer code for item {lines // 2:04} as "code".'
    return "Describe how this inventory is organised. Use full paragraphs."


def deep_messages(case, lines):
    return [{"role": "system", "content": "benchmark-deep. Benchmark request."},
            {"role": "user", "content": f"{deep_inventory(lines)}\n{deep_question(case, lines)}"}]


def deep_lines(url, model, target=DEEP_TOKENS):
    """Size the inventory in the model's own tokens: tokenizers differ by a third."""
    def count(lines):
        body = json.dumps({"model": model, "messages": deep_messages("deep_decode", lines)}).encode()
        req = urllib.request.Request(url.removesuffix("/v1") + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)["count"]

    lines = 1000
    for _ in range(3):
        lines = max(1, round(lines * target / count(lines)))
    tokens = count(lines)
    if abs(tokens - target) > target * 0.02:
        raise RuntimeError(f"Could not size the deep prompt: {lines} lines are {tokens} tokens")
    return {"lines": lines, "prompt_tokens": tokens}


def payload(model, case, nonce, lines=None):
    capabilities = SUITE[model]["capabilities"]
    always_thinks = capabilities["reasoning"] and not capabilities["thinking_toggle"]
    token_limit = 1024 if always_thinks else 128
    prompt = "Reply with exactly READY."
    if case in ("text", *CONCURRENCY):
        prompt, token_limit = TEXT, 256 if case == "text" else 128
    elif case in DEEP_CASES:
        if lines is None:
            raise ValueError("deep cases need the inventory length for this model's tokenizer")
        # One token ends the fresh read; the follow-ups then start from its cache.
        token_limit = {"deep_prefill": 1, "deep_decode": 128}.get(case, token_limit)
    elif case == "copy":
        prompt = f"{INVENTORY}\nRepeat items 100 to 119 exactly as written, one per line, and nothing else."
        token_limit += 400
    elif case == "prefill":
        prompt = f"{INVENTORY}\nReturn only the integer code for item 298."
    elif case == "prefill_json":
        prompt = f'{INVENTORY}\nReturn a JSON object containing only the integer code for item 298 as "code".'
    elif case == "prefill_ack":
        prompt = f"{INVENTORY}\nAcknowledge receipt of this inventory. Reply with exactly READY."
    elif case == "reasoning":
        prompt = ("A box starts with 17 rows of 23 bolts. Five bolts are removed. "
                  "Think through the calculation, then give only the remaining number in your final answer.")
        token_limit = 1024
    elif case == "reasoning_json":
        prompt = ("A box starts with 17 rows of 23 bolts. Five bolts are removed. "
                  "Think through the calculation, then give only a JSON object containing the remaining "
                  'number as the integer field "remaining" in your final answer.')
        token_limit = 1024
    elif case.startswith("json"):
        prompt = "Return a JSON object reporting job number 4711 and zero errors."
    elif case.startswith("tool_") and case != "tool_none":
        prompt = "Use report_job to report job number 4711 and zero errors. Do not add text."
    result = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": token_limit, "stream": True,
              "stream_options": {"include_usage": True}}
    result.update(copy.deepcopy(SUITE[model].get("benchmark_sampling", {})))
    if case in DEEP_CASES:
        # All three share one prompt prefix and salt. deep_prefill and
        # deep_decode ask the same question, so the decode starts fully cached.
        result["messages"] = deep_messages("deep_decode" if case == "deep_prefill" else case, lines)
        result["cache_salt"] = nonce
    elif case in PREFILL_CASES:
        # Olmo otherwise inserts a long shared system prompt before the nonce,
        # allowing prefix-cache hits even for a fresh inventory request.
        result["messages"].insert(0, {"role": "system", "content": f"{nonce}. Benchmark request."})
        # Templates can prepend tokens before any user-supplied system text.
        # vLLM salts the first cache block, preventing those hidden-prefix hits.
        result["cache_salt"] = nonce
    if capabilities["thinking_toggle"]:
        result["chat_template_kwargs"] = {"enable_thinking": case in ("reasoning", "reasoning_json")}
    if case.startswith("json"):
        result["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "job_report", "strict": True, "schema": copy.deepcopy(JOB_SCHEMA)}}
    if case in ("prefill_json", "deep_lookup"):
        result["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "inventory_code", "strict": True, "schema": copy.deepcopy(CODE_SCHEMA)}}
    if case == "reasoning_json":
        result["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "remaining_bolts", "strict": True, "schema": copy.deepcopy(REMAINING_SCHEMA)}}
    if case.startswith("tool_"):
        if not capabilities["tools"]:
            raise ValueError(f"{model} does not advertise tool calling")
        result["tools"] = [copy.deepcopy(TOOL)]
        choice = case.removeprefix("tool_").removesuffix("_nonstream")
        result["tool_choice"] = ({"type": "function", "function": {"name": "report_job"}}
                                 if choice == "named" else choice)
        result["parallel_tool_calls"] = False
        if case == "tool_none":
            result["messages"][0]["content"] = "Do not call a tool. Reply with exactly READY."
    if case.endswith("_nonstream"):
        result["stream"] = False
        result.pop("stream_options")
    return result


def trial_payload(model, case, run_nonce, attempt, accepted, max_tokens=None, deep=None):
    if case in DEEP_CASES:
        # The salt changes only when a fresh read has to be repeated.
        body = payload(model, case, f"{run_nonce}-deep-{deep['reads']}", deep["lines"])
    elif model != "olmo-3.1-32b-think":
        body = payload(model, case, f"{run_nonce}-{attempt}")
    else:
        # GPU-idle rejections must not change a seeded comparison's visible text.
        # Cache salts still vary per attempt, including rejected windows.
        body = payload(model, case, f"benchmark-{case}-{accepted}")
        if case in PREFILL_CASES:
            body["cache_salt"] = f"{run_nonce}-{case}-{attempt}"
    if max_tokens is not None:
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max-tokens must be a positive integer")
        body["max_tokens"] = max_tokens
    return body


def correct(case, row):
    if row.get("error"):
        return False
    answer = row["answer"].strip()
    if "<think>" in answer or "</think>" in answer:
        return False
    if case in ("text", "deep_decode", *CONCURRENCY):
        return bool(answer or row["reasoning"]) and row["usage"]["completion_tokens"] > 1
    if case == "deep_prefill":
        return row["usage"]["completion_tokens"] == 1
    if row["finish_reason"] == "length":
        return False
    if case in ("chat", "tool_none"):
        return answer == "READY" and not row["tool_calls"]
    if case == "copy":
        return [line.strip() for line in answer.splitlines() if line.strip()] == COPIED
    if case == "reasoning":
        return answer == "386" and bool(row["reasoning"].strip())
    if case == "reasoning_json":
        try:
            decoded = json.loads(answer)
            return (isinstance(decoded, dict) and set(decoded) == {"remaining"}
                    and type(decoded["remaining"]) is int and decoded["remaining"] == 386
                    and bool(row["reasoning"].strip()) and not row["tool_calls"]
                    and row["finish_reason"] == "stop")
        except (ValueError, TypeError):
            return False
    if case == "prefill":
        return answer == "3881"
    if case == "prefill_ack":
        return answer == "READY" and not row["tool_calls"] and row["finish_reason"] == "stop"
    if case in ("prefill_json", "deep_lookup"):
        expected = 3881 if case == "prefill_json" else (row["deep_lines"] // 2) * 13 + 7
        try:
            decoded = json.loads(answer)
            return (isinstance(decoded, dict) and set(decoded) == {"code"}
                    and type(decoded["code"]) is int and decoded["code"] == expected
                    and not row["tool_calls"] and row["finish_reason"] == "stop")
        except (ValueError, TypeError):
            return False
    if case.startswith("json"):
        try:
            decoded = json.loads(answer)
            return (decoded == EXPECTED_JOB and isinstance(decoded, dict)
                    and all(type(decoded[k]) is int for k in EXPECTED_JOB))
        except (ValueError, TypeError):
            return False
    if case.startswith("tool_"):
        calls = row["tool_calls"]
        try:
            decoded = json.loads(calls[0]["function"]["arguments"])
            # vLLM deliberately finishes a forced named call with "stop";
            # auto/required calls finish with "tool_calls". Both must still
            # contain exactly one complete call with correct typed arguments.
            finishes = {"tool_calls", "stop"} if case.startswith("tool_named") else {"tool_calls"}
            return (len(calls) == 1 and calls[0]["function"]["name"] == "report_job"
                    and row["finish_reason"] in finishes and decoded == EXPECTED_JOB
                    and all(type(decoded[k]) is int for k in EXPECTED_JOB))
        except (ValueError, TypeError, IndexError, KeyError):
            return False
    return False


def nonstream_request(url, key, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    began = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    elapsed = time.perf_counter() - began
    choice = value["choices"][0]
    message, usage = choice["message"], value["usage"]
    thought = message.get("reasoning") or message.get("reasoning_content") or ""
    tools = message.get("tool_calls") or []
    answer = message.get("content") or ""
    answer_tokens = answer_token_count(usage, answer, thought, tools)
    return {"ttft_s": None, "first_answer_s": None, "total_s": elapsed, "decode_tokens_per_s": None,
            "answer_tokens": answer_tokens,
            "answer_tokens_per_s": answer_tokens / elapsed if answer_tokens is not None else None,
            "usage": usage, "finish_reason": choice["finish_reason"], "answer": answer,
            "reasoning": thought, "tool_calls": tools}


def counter_delta(audit, model, metric):
    before = audit.get("before", {}).get(model, {}).get("counters", {})
    after = audit.get("after", {}).get(model, {}).get("counters", {})
    if metric in before and metric in after and after[metric] >= before[metric]:
        return after[metric] - before[metric]
    return None


def cached_tokens(rows, audit, model):
    """Tokens answered from the prefix cache in one audited trial.

    A server started without --enable-prompt-tokens-details leaves the usage
    field out. Its cache-hit counter says the same for a trial whose audit
    shows that only these requests ran.
    """
    reported = [(row["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") for row in rows]
    if all(type(n) is int for n in reported):
        return sum(reported)
    hits = counter_delta(audit, model, "vllm:prefix_cache_hits_total")
    return int(hits) if hits is not None else None


def prefill_evidence(rows, audit, model):
    cached = [(row["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") for row in rows]
    hits = counter_delta(audit, model, "vllm:prefix_cache_hits_total")
    seconds = counter_delta(audit, model, "vllm:request_prefill_time_seconds_sum")
    count = counter_delta(audit, model, "vllm:request_prefill_time_seconds_count")
    known_cached = cached_tokens(rows, audit, model)
    uncached = (sum(row["usage"]["prompt_tokens"] for row in rows) - known_cached
                if known_cached is not None else None)
    # These are server-observed request prefill times, not isolated kernel timings.
    rate = uncached / seconds if uncached is not None and seconds and count == len(rows) else None
    return {"reported_cached_tokens": cached, "backend_cache_hit_tokens_delta": hits,
            "server_prefill_seconds_delta": seconds, "server_prefill_requests_delta": count,
            "server_uncached_prefill_tokens_per_s": rate,
            "cache_freshness_verified": known_cached == 0,
            "prompt_tokens_over_ttft_is_not_a_prefill_measurement": True}


def summary(batches):
    rows = [row for batch in batches for row in batch["rows"]]
    elapsed = sum(batch["elapsed_s"] for batch in batches)

    def median(field):
        values = [row[field] for row in rows if row[field] is not None]
        return statistics.median(values) if values else None

    return {"requests": len(rows), "median_ttft_s": median("ttft_s"),
            "median_first_answer_s": median("first_answer_s"), "median_total_s": median("total_s"),
            "median_total_decode_tokens_per_s": median("decode_tokens_per_s"),
            "median_useful_answer_tokens_per_s": median("answer_tokens_per_s"),
            "aggregate_total_output_tokens_per_s": sum(row["usage"]["completion_tokens"] for row in rows) / elapsed,
            "aggregate_useful_answer_tokens_per_s": (sum(row["answer_tokens"] for row in rows) / elapsed
                if all(row["answer_tokens"] is not None for row in rows) else None),
            "reasoning_tokens": [(row["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") for row in rows]}


def serving_metadata(model):
    try:
        result = subprocess.run(["podman", "inspect", f"llm-{model}"], check=True, capture_output=True, text=True)
        container = json.loads(result.stdout)[0]
        allowed = {"LOCAL_LLM_FAST_KERNELS", "OMP_NUM_THREADS", "GPU_MAX_HW_QUEUES",
                   "VLLM_TUNED_CONFIG_FOLDER", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_V1",
                   "VLLM_ATTENTION_BACKEND", "VLLM_CACHE_ROOT"}
        environment = dict(value.split("=", 1) for value in container["Config"].get("Env", []) if "=" in value)
        return {"image_id": container.get("Image"), "image_name": container.get("ImageName"),
                "command": container["Config"].get("Cmd"),
                "environment": {name: value for name, value in environment.items() if name in allowed}}
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        return {"unavailable": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=SUITE)
    parser.add_argument("label", help="Report label only; this never changes a server")
    parser.add_argument("--case", action="append", choices=("all", "quality", "extended", *CASES, *EXTRA_CASES),
                        help="extended adds 16 concurrent requests and the long-context cases to all")
    parser.add_argument("--keep-going", action="store_true",
                        help="Record a failed case and continue other cases; still exit unsuccessfully")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int,
                        help="Explicit generation-token budget for selected cases; omitted preserves each fixture's budget")
    parser.add_argument("--max-attempts", type=int, default=400,
                        help="Includes samples rejected while the GPU cools down or another model answers; "
                             "the default waits a few minutes per case for someone using a model")
    parser.add_argument("--max-idle-gpu-busy", type=float, default=20)
    parser.add_argument("--url", help="Request endpoint, default target backend /v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, default=Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "local-llm/api-key")
    args = parser.parse_args()
    if min(args.repeats, args.timeout, args.max_attempts) <= 0 or args.warmups < 0:
        parser.error("repeats, timeout and max-attempts must be positive; warmups must be nonnegative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("max-tokens must be positive")
    lock_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "local-llm-benchmark.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another added-model benchmark owns the GPU timing lock")
        with benchmark(args.model):
            run(args)


def run(args):
    config = json.loads(subprocess.check_output([str(HERE / "llm"), "config"], text=True))
    backends = {model["name"]: f"http://127.0.0.1:{model['port']}/v1" for model in config["models"]}
    if args.model in MODELS:
        # A standalone benchmark server can be reached before the public registry is refreshed.
        backends.setdefault(args.model, f"http://127.0.0.1:{MODELS[args.model]['port']}/v1")
    url = args.url or backends[args.model]
    key = args.key_file.read_text().strip()
    guard = Guard(args.model, backends, args.max_idle_gpu_busy)
    cases = selected_cases(args.case or ["all"], SUITE[args.model]["capabilities"])
    # Tokenizing goes to the model server itself; the gateway does not pass it on.
    deep = ({"reads": 0} | deep_lines(backends[args.model], args.model)
            if any(case in DEEP_CASES for case in cases) else None)
    report = {"at": timestamp(), "model": args.model, "label": args.label,
              "checkpoint": SUITE[args.model], "runtime": serving_metadata(args.model), "deep": deep,
              "configured_model": next((m for m in config["models"] if m["name"] == args.model), None),
              "options": {"cases": cases, "repeats": args.repeats, "warmups": args.warmups,
                          "timeout_s": args.timeout, "max_idle_gpu_busy": args.max_idle_gpu_busy,
                          "max_tokens": getattr(args, "max_tokens", None),
                          "keep_going": getattr(args, "keep_going", False)},
              "backends_observed": backends, "trials": [], "summaries": {}, "failures": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        partial.write_text(json.dumps(report, indent=2) + "\n")
        partial.replace(args.output)

    def fail(case, message):
        report["failures"][case] = message
        report["passed"] = False
        save()
        if not getattr(args, "keep_going", False):
            raise RuntimeError(message)

    run_nonce = uuid.uuid4().hex
    save()
    for case in cases:
        accepted = []
        warmups, repeats = (0, 1) if case in SINGLE_PASS else (args.warmups, args.repeats)
        for attempt in range(args.max_attempts):
            body = trial_payload(args.model, case, run_nonce, attempt, len(accepted),
                                 getattr(args, "max_tokens", None), deep)
            concurrency = CONCURRENCY.get(case, 1)

            def run_batch():
                began = time.perf_counter()

                def send():
                    try:
                        call = request if body["stream"] else nonstream_request
                        return call(url, key, body, args.timeout)
                    except urllib.error.HTTPError as exc:
                        return {"error": f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"}
                    except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
                        return {"error": str(exc)}

                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows = list(pool.map(lambda _: send(), range(concurrency)))
                return {"rows": rows, "elapsed_s": time.perf_counter() - began}

            value, audit = guard.trial(run_batch, concurrency)
            trial = {"case": case, "attempt": attempt, "at": timestamp(),
                     "warmup": len(accepted) < warmups, "request": body, "audit": audit, "result": value}
            if case in DEEP_CASES:
                # The whole inventory would otherwise be stored once per trial.
                trial["request"] = body | {"messages": f"deep inventory, {deep['lines']} lines; {deep_question('deep_decode' if case == 'deep_prefill' else case, deep['lines'])}"}
                if value:
                    for row in value["rows"]:
                        row["deep_lines"] = deep["lines"]
            report["trials"].append(trial)
            save()
            if value and case == "deep_prefill":
                # This read is in the cache now, clean window or not. Repeating
                # it needs a new salt, and the follow-ups must use that one.
                if audit["reasons"] or any(row.get("error") for row in value["rows"]):
                    deep["reads"] += 1
            if value and any(row.get("error") for row in value["rows"]):
                fail(case, f"Request failed for {case}; inspect {args.output}")
                break
            if value:
                trial["correct"] = all(correct(case, row) for row in value["rows"])
                if case in PREFILL_CASES:
                    trial["prefill"] = prefill_evidence(value["rows"], audit, args.model)
            if audit["reasons"]:
                print(json.dumps({"case": case, "discarded": audit["reasons"]}), flush=True)
                time.sleep(0.5)
                continue
            if not trial["correct"]:
                fail(case, f"Correctness check failed for {case}; inspect {args.output}")
                break
            if case in PREFILL_CASES:
                if not trial["prefill"]["cache_freshness_verified"]:
                    fail(case, f"Fresh-prefill cache state was not verified for {case}; inspect {args.output}")
                    break
            if case in ("deep_decode", "deep_lookup"):
                cached = cached_tokens(value["rows"], audit, args.model)
                trial["cached_fraction"] = (cached / sum(row["usage"]["prompt_tokens"] for row in value["rows"])
                                            if cached is not None else None)
                # Without the cache these would time a second fresh read instead.
                if trial["cached_fraction"] is None or trial["cached_fraction"] < 0.9:
                    fail(case, f"The deep prompt was not answered from the prefix cache for {case}; inspect {args.output}")
                    break
            trial["accepted"] = True
            accepted.append(value)
            save()
            print(json.dumps({"case": case, "warmup": trial["warmup"],
                              "total_s": [row["total_s"] for row in value["rows"]],
                              "correct": True}), flush=True)
            if len(accepted) == warmups + repeats:
                measured = accepted[warmups:]
                report["summaries"][case] = summary(measured)
                if case in DEEP_CASES:
                    report["summaries"][case]["prompt_tokens"] = measured[0]["rows"][0]["usage"]["prompt_tokens"]
                if case in PREFILL_CASES:
                    rates = [t["prefill"]["server_uncached_prefill_tokens_per_s"] for t in report["trials"]
                             if t["case"] == case and t.get("accepted") and not t["warmup"]]
                    rates = [value for value in rates if value is not None]
                    report["summaries"][case]["median_server_uncached_prefill_tokens_per_s"] = statistics.median(rates) if rates else None
                save()
                print(json.dumps({"case": case, "summary": report["summaries"][case]}), flush=True)
                break
        else:
            fail(case, f"No clean measurement window for {case}; partial report saved to {args.output}")
    report["completed_at"] = timestamp()
    report["passed"] = not report["failures"]
    save()
    if report["failures"]:
        raise RuntimeError(f"Required cases failed: {', '.join(report['failures'])}; inspect {args.output}")
    return report


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
