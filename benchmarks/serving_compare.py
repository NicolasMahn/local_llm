"""Measure a running model, excluding trials overlapped by other clients."""

import argparse
import concurrent.futures
import json
import statistics
import sys
import threading
import time
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import contamination_reasons, media_part, parse_backend_metrics, request, require_completion_count


class Guard:
    def __init__(self, urls):
        self.urls = urls

    def snapshot(self):
        readings = []
        for url in self.urls:
            with urllib.request.urlopen(url.removesuffix("/v1") + "/metrics", timeout=5) as response:
                readings.append(parse_backend_metrics(response.read().decode()))
        require_completion_count(self.urls[0], readings[0])
        return readings

    def trial(self, call, count):
        before = self.snapshot()
        if any(row["running"] + row["waiting"] for row in before):
            return None, {"reasons": ["a backend was busy before the trial"], "before": before}
        peak = [0] * len(before)
        errors = []
        done = threading.Event()

        def observe():
            while not done.is_set():
                try:
                    for index, row in enumerate(self.snapshot()):
                        peak[index] = max(peak[index], row["running"] + row["waiting"])
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))
                done.wait(0.2)

        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        try:
            value = call()
            deadline = time.monotonic() + 3
            after = self.snapshot()
            # The engine can publish its final stats just after the API's DONE.
            while after[0]["completed"] - before[0]["completed"] < count and time.monotonic() < deadline:
                time.sleep(0.1)
                after = self.snapshot()
        finally:
            done.set()
            observer.join(timeout=6)
        reasons = contamination_reasons(before, after, peak, count) + errors
        if observer.is_alive():
            reasons.append("metrics observer did not finish")
        return value, {"before": before, "after": after, "peak_active": peak, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("qwen3-omni-30b", "gemma-4-26b"))
    parser.add_argument("mode", choices=("profiles", "kernels"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, default=Path("logs/performance-2026-11-02/fixtures"))
    parser.add_argument("--case", choices=("text", "batch4", "prefill", "media", "quality", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    port, peer = (8006, 8002) if args.model == "qwen3-omni-30b" else (8002, 8006)
    url = f"http://127.0.0.1:{port}/v1"
    key = (Path.home() / ".config/local-llm/api-key").read_text().strip()
    guard = Guard([url, f"http://127.0.0.1:{peer}/v1"])
    report = {"at": datetime.now(timezone.utc).isoformat(), "model": args.model,
              "mode": args.mode, "trials": [], "summaries": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def body(content, max_tokens):
        return {"model": args.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": max_tokens, "stream": True,
                "stream_options": {"include_usage": True}}

    def measured(label, make_body, concurrency=1, check=None, check_response=None):
        accepted = []
        attempts = 0
        deadline = time.monotonic() + 240
        while len(accepted) < args.repeats + 1:
            if attempts >= 30 or time.monotonic() > deadline:
                raise RuntimeError(f"No clean measurement window for {label}; partial results saved")
            attempt = attempts
            attempts += 1
            payload = make_body(attempt)

            def run_batch():
                began = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(request, url, key, payload, 120) for _ in range(concurrency)]
                    rows = [future.result() for future in futures]
                return {"rows": rows, "elapsed_s": time.perf_counter() - began}

            value, audit = guard.trial(run_batch, concurrency)
            warmup = not accepted
            row = {"case": label, "attempt": attempt, "warmup": warmup,
                   "at": datetime.now(timezone.utc).isoformat(), "audit": audit, "result": value}
            report["trials"].append(row)
            save()
            if audit["reasons"]:
                print(json.dumps({"case": label, "discarded": audit["reasons"]}), flush=True)
                time.sleep(0.5)
                continue
            if check and not all(check(item["answer"]) for item in value["rows"]):
                raise RuntimeError(f"Answer check failed for {label}; inspect saved answers")
            if check_response and not all(check_response(item) for item in value["rows"]):
                raise RuntimeError(f"Response check failed for {label}; inspect saved responses")
            accepted.append(value)
            print(json.dumps({"case": label, "warmup": warmup,
                              "ttft_s": [item["ttft_s"] for item in value["rows"]],
                              "decode_tps": [item["decode_tokens_per_s"] for item in value["rows"]]}), flush=True)
        batches = accepted[1:]
        flat = [item for batch in batches for item in batch["rows"]]

        def median(field):
            values = [item[field] for item in flat if item[field] is not None]
            return statistics.median(values) if values else None

        summary = {"requests": len(flat), "median_ttft_s": median("ttft_s"),
                   "median_total_s": median("total_s"),
                   "median_decode_tokens_per_s": median("decode_tokens_per_s"),
                   "aggregate_output_tokens_per_s": sum(item["usage"]["completion_tokens"] for item in flat)
                   / sum(batch["elapsed_s"] for batch in batches),
                   "discarded_attempts": attempts - len(accepted)}
        report["summaries"][label] = summary
        save()
        print(json.dumps({"case": label, "summary": summary}), flush=True)

    selected = lambda name: args.case in (name, "all")
    text = ("Write a detailed twenty-paragraph explanation of how a compiler works, covering parsing, "
            "type checking, optimization and machine code. Use full paragraphs.")
    if selected("text"):
        measured("text", lambda _: body(text, 256))
    if selected("batch4"):
        measured("batch4", lambda _: body(text, 128), concurrency=4)
    if selected("prefill"):
        inventory = (args.fixture_dir / "long.txt").read_text()
        measured("fresh_prefill", lambda i: body(f"Independent case {i}.\n{inventory}", 48),
                 check=lambda answer: "3881" in answer)
    if selected("media"):
        from PIL import Image
        fresh = args.output.parent / "fresh-media"
        fresh.mkdir(exist_ok=True)
        image = Image.open(args.fixture_dir / "screen.png").convert("RGB")
        with wave.open(str(args.fixture_dir / "question.wav"), "rb") as clip:
            params, audio = clip.getparams(), clip.readframes(clip.getnframes())

        def media(i):
            picture = image.copy()
            picture.putpixel((picture.width - 1, picture.height - 1), (i % 255, 30, 80))
            path = fresh / f"screen-{i}.png"
            picture.save(path)
            parts = [media_part(path, "image")]
            prompt = "What is the job number and error count? Answer in one short sentence."
            if args.model == "qwen3-omni-30b":
                sound = fresh / f"question-{i}.wav"
                with wave.open(str(sound), "wb") as clip:
                    clip.setparams(params)
                    clip.writeframes(audio + b"\0" * params.sampwidth * params.nchannels * i * 100)
                parts += [{"type": "text", "text": "Answer the spoken question about this screenshot in one short sentence."},
                          media_part(sound, "audio")]
            else:
                parts.append({"type": "text", "text": prompt})
            return body(parts, 48)

        measured("fresh_media", media, check=lambda answer: "4711" in answer and "0" in answer)
    if selected("quality"):
        measured("arithmetic", lambda _: body("What is 17 times 23? Return only the digits.", 16),
                 check=lambda answer: answer.strip() == "391")
        def json_case(_):
            payload = body('Return exactly a JSON object with integer fields job=4711 and errors=0.', 48)
            payload["response_format"] = {"type": "json_object"}
            return payload

        measured("json", json_case, check=lambda answer: json.loads(answer) == {"job": 4711, "errors": 0})
        def tool_case(_):
            payload = body('Use report_job to report job number 4711 and zero errors. Do not add text.', 128)
            payload["tools"] = [{"type": "function", "function": {"name": "report_job",
                "description": "Report a job number and its error count.",
                "parameters": {"type": "object", "properties": {"job": {"type": "integer"},
                    "errors": {"type": "integer"}}, "required": ["job", "errors"], "additionalProperties": False}}}]
            payload["tool_choice"] = {"type": "function", "function": {"name": "report_job"}}
            return payload

        def correct_tool(item):
            calls = item["tool_calls"]
            return (len(calls) == 1 and calls[0]["function"]["name"] == "report_job"
                    and json.loads(calls[0]["function"]["arguments"]) == {"job": 4711, "errors": 0})

        measured("tool_call", tool_case, check_response=correct_tool)
    save()


if __name__ == "__main__":
    main()
