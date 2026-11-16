"""Measure an audited fresh-inventory/follow-up pair on an already-running model.

The second request sends only the first final answer in its history. This tests
the cache behavior of an ordinary chat client that omits reasoning. Retention
settings are never selected or changed here. Prompt length / TTFT is not used
as a prefill-rate estimate.
"""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.added_models import (CODE_SCHEMA, Guard, HERE, INVENTORY, audit_reasons,
                                     serving_metadata, timestamp)
from benchmarks.api_smoke import send_request
from model_catalog import MODELS

EXPECTED_FIRST = 3881
EXPECTED_FOLLOWUP = 3868


def first_payload(model, nonce, json_output=False):
    capabilities = MODELS[model]["capabilities"]
    instruction = ('Return a JSON object containing only the integer code for item 298 as "code".'
                   if json_output else "Return only the integer code for item 298.")
    body = {"model": model, "messages": [
        {"role": "system", "content": f"{nonce}. Cache follow-up check."},
        {"role": "user", "content": f"{INVENTORY}\n{instruction}"}],
        "cache_salt": nonce, "temperature": 0,
        "max_tokens": 128 if capabilities["thinking_toggle"] else 1024,
        "stream": True, "stream_options": {"include_usage": True}}
    body.update(copy.deepcopy(MODELS[model].get("benchmark_sampling", {})))
    if capabilities["thinking_toggle"]:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if json_output:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "inventory_code", "strict": True, "schema": copy.deepcopy(CODE_SCHEMA)}}
    return body


def followup_payload(first, answer, json_output=False):
    if not isinstance(answer, str):
        raise ValueError("The first final answer must be text")
    body = copy.deepcopy(first)
    instruction = ('Return a JSON object containing only the integer code for item 297 as "code".'
                   if json_output else "Return only the integer code for item 297.")
    body["messages"] += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": instruction}]
    return body


def integer_correct(row, expected, json_output=False):
    if not row or row.get("error") or row.get("tool_calls") or row.get("finish_reason") != "stop":
        return False
    try:
        value = json.loads(row["answer"].strip())
        if json_output:
            return (isinstance(value, dict) and set(value) == {"code"}
                    and type(value["code"]) is int and value["code"] == expected)
        return type(value) is int and value == expected
    except (KeyError, ValueError, TypeError, AttributeError):
        return False


def cached_tokens(row):
    usage = (row or {}).get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens") if isinstance(details, dict) else None
    prompt = usage.get("prompt_tokens")
    return value if type(value) is int and type(prompt) is int and 0 <= value <= prompt else None


def diagnostic_row(row):
    fields = ("answer", "reasoning", "tool_calls", "finish_reason", "usage",
              "total_s", "ttft_s", "first_answer_s", "protocol_complete",
              "error", "http_status", "raw_sse", "raw_response")
    result = {name: row[name] for name in fields if name in row}
    result["reported_cached_tokens"] = cached_tokens(row)
    return result


def stream_request(url, key, body, timeout):
    row = send_request(url, key, body, {"protocol": "openai", "stream": True}, timeout)
    return diagnostic_row(row)


def execute_pair(model, nonce, url, key, timeout, json_output=False):
    first = first_payload(model, nonce, json_output)
    first_row = stream_request(url, key, first, timeout)
    result = {"first_request": first, "first": first_row, "followup_request": None,
              "followup": None, "sent_requests": 1}
    # A malformed/truncated answer must not become fabricated assistant history.
    if not integer_correct(first_row, EXPECTED_FIRST, json_output):
        return result
    followup = followup_payload(first, first_row["answer"], json_output)
    result.update(followup_request=followup, sent_requests=2,
                  followup=stream_request(url, key, followup, timeout))
    return result


def pair_audit(target, pair, audit):
    count = pair.get("sent_requests", 2) if pair else 2
    if count == 2 or "peak_active" not in audit:
        return audit
    before, after, peak = audit["before"], audit["after"], audit["peak_active"]
    observed = set(peak)
    previous = set(audit_reasons(target, before, after, peak, observed, 2))
    reasons = set(audit["reasons"]) - previous
    reasons.update(audit_reasons(target, before, after, peak, observed, count))
    return audit | {"reasons": sorted(reasons), "expected_completions": count}


def summarize(pairs, json_output=False):
    first = [pair["first"] for pair in pairs]
    followup = [pair["followup"] for pair in pairs]

    def median(rows, field):
        values = [row[field] for row in rows if row.get(field) is not None]
        return statistics.median(values) if values else None

    first_cached = [cached_tokens(row) for row in first]
    followup_cached = [cached_tokens(row) for row in followup]
    known = [value for value in followup_cached if value is not None]
    return {"pairs": len(pairs), "output_format": "json" if json_output else "plain",
            "median_first_total_s": median(first, "total_s"),
            "median_first_ttft_s": median(first, "ttft_s"),
            "median_first_answer_s": median(first, "first_answer_s"),
            "median_followup_total_s": median(followup, "total_s"),
            "median_followup_ttft_s": median(followup, "ttft_s"),
            "median_followup_first_answer_s": median(followup, "first_answer_s"),
            "first_cached_tokens": first_cached, "followup_cached_tokens": followup_cached,
            "median_followup_cached_tokens": statistics.median(known) if known else None,
            "pairs_with_reported_followup_cache": len(known),
            "pairs_with_followup_cache_hits": sum(value > 0 for value in known),
            "cache_hit_pair_fraction_when_reported": sum(value > 0 for value in known) / len(known) if known else None}


def run(args):
    json_output = getattr(args, "json", False)
    config = json.loads(subprocess.check_output([str(HERE / "llm"), "config"], text=True))
    backends = {model["name"]: f"http://127.0.0.1:{model['port']}/v1" for model in config["models"]}
    if args.model not in backends:
        raise ValueError("Target model is missing from the configured registry")
    url = args.url or backends[args.model]
    key = args.key_file.read_text().strip()
    guard = Guard(args.model, backends, args.max_idle_gpu_busy)
    report = {"at": timestamp(), "model": args.model, "label": args.label, "url": url,
              "output_format": "json" if json_output else "plain",
              "checkpoint": MODELS[args.model], "runtime": serving_metadata(args.model),
              "backends_observed": backends, "options": {"repeats": args.repeats, "warmups": args.warmups,
              "max_attempts": args.max_attempts, "timeout_s": args.timeout}, "trials": [],
              "fixture": {"inventory_items": 300, "first_item": 298, "first_expected": EXPECTED_FIRST,
                          "followup_item": 297, "followup_expected": EXPECTED_FOLLOWUP},
              "prefill_rate_estimated_from_ttft": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        partial.write_text(json.dumps(report, indent=2) + "\n")
        partial.replace(args.output)

    accepted = []
    save()
    for attempt in range(args.max_attempts):
        nonce = uuid.uuid4().hex
        try:
            pair, audit = guard.trial(lambda: execute_pair(args.model, nonce, url, key, args.timeout, json_output), 2)
        except (OSError, ValueError, RuntimeError) as exc:
            pair, audit = None, {"reasons": [f"{type(exc).__name__}: {exc}"]}
        audit = pair_audit(args.model, pair, audit)
        trial = {"attempt": attempt + 1, "at": timestamp(), "nonce": nonce,
                 "warmup": len(accepted) < args.warmups, "audit": audit, "result": pair}
        report["trials"].append(trial)
        save()
        retry = audit["reasons"]
        if pair and any((pair.get(step) or {}).get("error") for step in ("first", "followup")):
            retry = [reason for reason in retry if not reason.startswith("target completion count did not match ")]
        if retry:
            print(json.dumps({"attempt": attempt + 1, "discarded": retry}), flush=True)
            if attempt + 1 < args.max_attempts:
                time.sleep(args.retry_delay)
            continue
        if pair is None:
            continue
        trial["correct"] = (integer_correct(pair["first"], EXPECTED_FIRST, json_output)
                            and integer_correct(pair["followup"], EXPECTED_FOLLOWUP, json_output))
        first_cache = cached_tokens(pair["first"])
        trial["first_freshness_verified"] = first_cache == 0 if first_cache is not None else None
        if not trial["correct"]:
            report["failure"] = "A required final integer was wrong, truncated, missing or unavailable"
            save()
            break
        accepted.append(pair)
        print(json.dumps({"attempt": attempt + 1, "warmup": trial["warmup"], "correct": True,
                          "first_total_s": pair["first"]["total_s"], "followup_total_s": pair["followup"]["total_s"],
                          "followup_cached_tokens": cached_tokens(pair["followup"])}), flush=True)
        if len(accepted) == args.warmups + args.repeats:
            break
    else:
        report["failure"] = "No sufficient uncontaminated idle windows within the attempt limit"
    report["passed"] = len(accepted) == args.warmups + args.repeats and "failure" not in report
    report["summary"] = summarize(accepted[args.warmups:], json_output)
    report["completed_at"] = timestamp()
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("label", help="Report label; does not change the running model")
    parser.add_argument("--json", action="store_true",
                        help="Require strict integer-code JSON in both final answers; default plain integer fixture is unchanged")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--retry-delay", type=float, default=0.5)
    parser.add_argument("--max-idle-gpu-busy", type=float, default=20)
    parser.add_argument("--url", help="OpenAI-compatible /v1 URL; defaults to the target backend")
    parser.add_argument("--key-file", type=Path, default=Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "local-llm/api-key")
    args = parser.parse_args()
    if (args.repeats <= 0 or args.warmups < 0 or args.timeout <= 0 or args.retry_delay < 0
            or not 1 <= args.max_attempts <= 64 or args.max_attempts < args.repeats + args.warmups):
        parser.error("Require positive repeats/timeout, nonnegative warmups/delay and enough attempts (at most 64)")
    lock_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "local-llm-benchmark.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another added-model benchmark owns the GPU timing lock")
        report = run(args)
    if not report["passed"]:
        raise SystemExit(f"Cache follow-up check failed; inspect {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
