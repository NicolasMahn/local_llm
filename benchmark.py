#!/usr/bin/env python3
"""Measure client-visible latency without changing or restarting a model."""

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_backend_metrics(text):
    names = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
             "vllm:request_success_total": "completed"}
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in names:
            key = names[name]
            result[key] = result.get(key, 0) + float(line.rsplit(None, 1)[-1])
    if set(result) != set(names.values()):
        raise ValueError("Backend metrics do not include request gauges and completion counters")
    return result


def contamination_reasons(before, after, peak_active, expected):
    """Reject overlap, including short requests that finished between gauge polls."""
    reasons = []
    for index, (start, end, peak) in enumerate(zip(before, after, peak_active)):
        label = "target" if index == 0 else f"peer {index}"
        own = expected if index == 0 else 0
        if start["running"] + start["waiting"]:
            reasons.append(f"{label} was busy before the trial")
        if end["running"] + end["waiting"]:
            reasons.append(f"{label} was busy after the trial")
        if peak > own:
            reasons.append(f"{label} held more than {own} expected requests")
        if end["completed"] - start["completed"] != own:
            reasons.append(f"{label} completion count did not match {own} benchmark requests")
    return reasons


def answer_token_count(usage, answer, reasoning="", tool_calls=()):
    """Count useful final tokens only when observed fields support the split.

    Some reasoning parsers report zero when they do not implement token
    counting. Keep that raw usage, but do not treat observed thought as final
    output. An empty final field establishes zero useful tokens independently
    of the server's reasoning counter.
    """
    if tool_calls:
        return None
    if not answer.strip():
        return 0
    tokens = usage.get("completion_tokens")
    if type(tokens) is not int or tokens < 0:
        return None
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning_tokens is None and not reasoning:
        return tokens
    if type(reasoning_tokens) is not int or not 0 <= reasoning_tokens <= tokens:
        return None
    if reasoning and reasoning_tokens in (0, tokens):
        return None
    return tokens - reasoning_tokens


def consume_stream(lines, started):
    first = None
    first_answer = None
    usage = {}
    finish = None
    text = []
    reasoning = []
    tools = {}
    complete = False
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            complete = True
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(str(event["error"]))
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            thought = delta.get("reasoning") or delta.get("reasoning_content")
            # Role announcements and usage-only events are not a first token.
            if content or thought or delta.get("tool_calls"):
                if first is None or (content and first_answer is None):
                    arrived = time.perf_counter()
                    if first is None:
                        first = arrived
                    if content and first_answer is None:
                        first_answer = arrived
                if content:
                    text.append(content)
                if thought:
                    reasoning.append(thought)
                for fragment in delta.get("tool_calls", []):
                    tool = tools.setdefault(fragment.get("index", 0), {"function": {"name": "", "arguments": ""}})
                    if fragment.get("id"):
                        tool["id"] = fragment["id"]
                    for field in ("name", "arguments"):
                        tool["function"][field] += fragment.get("function", {}).get(field) or ""
            finish = choice.get("finish_reason") or finish
    ended = time.perf_counter()
    if not complete:
        raise RuntimeError("The response stream ended without [DONE]; discard this measurement")
    if "completion_tokens" not in usage:
        raise RuntimeError("The server did not report token usage; cannot measure throughput")
    tokens = usage["completion_tokens"]
    answer = "".join(text)
    thought = "".join(reasoning)
    answer_tokens = answer_token_count(usage, answer, thought, tools)
    total = ended - started
    return {
        "ttft_s": first - started if first is not None else None,
        "first_answer_s": first_answer - started if first_answer is not None else None,
        "total_s": total,
        # Tool parsers can buffer several tokens before their first event;
        # their chunk timings cannot establish a token generation rate. This
        # rate includes reasoning; it is not useful final-answer throughput.
        "decode_tokens_per_s": (tokens - 1) / (ended - first) if first is not None and tokens > 1 and not tools else None,
        "answer_tokens": answer_tokens,
        "answer_tokens_per_s": answer_tokens / total if answer_tokens is not None and total > 0 else None,
        "usage": usage,
        "finish_reason": finish,
        "answer": answer,
        "reasoning": thought,
        "tool_calls": [tools[index] for index in sorted(tools)],
    }


def request(url, key, body, timeout):
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return consume_stream(response, started)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def media_part(path, kind):
    mime = mimetypes.guess_type(path)[0]
    # These containers carry browser microphone recordings as well as video.
    if kind == "audio" and Path(path).suffix.lower() in (".webm", ".mp4"):
        mime = {".webm": "audio/webm", ".mp4": "audio/mp4"}[Path(path).suffix.lower()]
    if mime is None or not mime.startswith(f"{kind}/"):
        raise ValueError(f"Cannot identify {kind} format: {path}")
    data = base64.b64encode(Path(path).read_bytes()).decode()
    field = f"{kind}_url"
    return {"type": field, field: {"url": f"data:{mime};base64,{data}"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--url", default="http://localhost:8400/v1")
    parser.add_argument("--key-file", type=Path, default=Path(os.environ.get(
        "XDG_CONFIG_HOME", Path.home() / ".config")) / "local-llm/api-key")
    parser.add_argument("--prompt", default="Explain how a compiler works in detail. Use full paragraphs.")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--pixels", type=int, help="Qwen Omni image pixel budget; keeps the 401408-pixel floor")
    parser.add_argument("--no-thinking", action="store_true", help="For models supporting enable_thinking")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, help="Save requests, answers and measurements as JSON (never the key)")
    args = parser.parse_args()
    if min(args.max_tokens, args.concurrency, args.repeats, args.timeout) <= 0 or args.warmups < 0:
        parser.error("Token limit, concurrency, repeats and timeout must be positive; warmups must be nonnegative")
    if args.pixels is not None and (args.model != "qwen3-omni-30b" or args.pixels < 401408):
        parser.error("--pixels is for qwen3-omni-30b and must be at least 401408")
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    parts = [media_part(args.image, "image")] if args.image else []
    parts.append({"type": "text", "text": prompt})
    if args.audio:
        parts.append(media_part(args.audio, "audio"))
    body = {
        "model": args.model, "messages": [{"role": "user", "content": parts}],
        "temperature": 0, "max_tokens": args.max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if args.pixels is not None:
        body["mm_processor_kwargs"] = {"size": {"longest_edge": args.pixels, "shortest_edge": 401408}}
    if args.no_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    key = args.key_file.read_text().strip()
    rows = []
    elapsed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for index in range(args.warmups + args.repeats):
            started = time.perf_counter()
            futures = [pool.submit(request, args.url, key, body, args.timeout) for _ in range(args.concurrency)]
            batch = [future.result() for future in futures]
            took = time.perf_counter() - started
            warmup = index < args.warmups
            for row in batch:
                row.update(warmup=warmup, batch=index)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ("answer", "reasoning")}), flush=True)
            if not warmup:
                elapsed += took
    measured = [row for row in rows if not row["warmup"]]

    def median(field):
        values = [row[field] for row in measured if row[field] is not None]
        return statistics.median(values) if values else None

    summary = {
        "model": args.model, "concurrency": args.concurrency, "requests": len(measured),
        "median_ttft_s": median("ttft_s"), "median_total_s": median("total_s"),
        "median_first_answer_s": median("first_answer_s"),
        "median_decode_tokens_per_s": median("decode_tokens_per_s"),
        "median_answer_tokens_per_s": median("answer_tokens_per_s"),
        "aggregate_output_tokens_per_s": sum(row["usage"].get("completion_tokens", 0) for row in measured) / elapsed,
        "aggregate_answer_tokens_per_s": (sum(row["answer_tokens"] for row in measured) / elapsed
                                          if all(row["answer_tokens"] is not None for row in measured) else None),
    }
    print(json.dumps({"summary": summary}), flush=True)
    if args.output:
        # Media bytes can be large; keep file references and options for reproduction.
        options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "key_file"}
        report = {"at": datetime.now(timezone.utc).isoformat(), "options": options, "summary": summary, "requests": rows}
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
