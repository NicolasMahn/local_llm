"""Check both gateway APIs with synthetic messages and audited backend activity.

This is a correctness smoke test. Recorded HTTP latency is diagnostic only;
neither streamed chunks nor these short answers establish token throughput.
No server is started, stopped or reconfigured.
"""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import consume_stream
from benchmarks.added_models import (Guard, HERE, TOOL, audit_reasons, correct,
                                     payload, serving_metadata, timestamp)
from model_catalog import MODELS


class ProtocolError(RuntimeError):
    pass


def cases_for(model):
    capabilities = MODELS[model]["capabilities"]
    cases = []
    for protocol in ("openai", "anthropic"):
        kinds = ["chat", "reasoning"] + (["json"] if protocol == "openai" else []) + ["tool_named"]
        for kind in kinds:
            for stream in (True, False):
                supported = capabilities["tools"] if kind.startswith("tool_") else capabilities.get(
                    "structured_output", False) if kind == "json" else True
                cases.append({"id": f"{protocol}_{kind}_{'stream' if stream else 'nonstream'}",
                              "protocol": protocol, "kind": kind, "stream": stream,
                              "required": supported})
        if protocol == "openai":
            cases.append({"id": "openai_tool_auto_stream", "protocol": protocol,
                          "kind": "tool_auto", "stream": True, "required": capabilities["tools"]})
    return cases


def request_body(model, case, max_tokens=None):
    body = payload(model, case["kind"], "gateway-smoke")
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    body["stream"] = case["stream"]
    if not case["stream"]:
        body.pop("stream_options", None)
    if case["protocol"] == "openai":
        return body
    result = {name: copy.deepcopy(body[name]) for name in
              ("model", "messages", "max_tokens", "temperature", "stream")}
    if "top_p" in body:
        result["top_p"] = body["top_p"]
    if "chat_template_kwargs" in body:
        # The installed vLLM Anthropic schema supports this extension, not
        # Anthropic's top-level thinking/budget_tokens request object.
        result["chat_template_kwargs"] = copy.deepcopy(body["chat_template_kwargs"])
    if case["kind"].startswith("tool_"):
        function = TOOL["function"]
        result["tools"] = [{"name": function["name"], "description": function["description"],
                            "input_schema": copy.deepcopy(function["parameters"])}]
        result["tool_choice"] = {"type": "tool", "name": function["name"],
                                  "disable_parallel_tool_use": True}
    return result


def sse_events(lines):
    event, data = None, []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = None, []
        elif line.startswith("event:"):
            event = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line.startswith(":") or line.startswith(("id:", "retry:")):
            continue
        else:
            raise ProtocolError(f"Unexpected SSE field: {line}")
    if data:
        yield event, json.loads("\n".join(data))


def require_string(value, field, *, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value):
        raise ProtocolError(f"Invalid {field}: expected {'nonempty ' if nonempty else ''}string")
    return value


def validate_block(block):
    if not isinstance(block, dict):
        raise ProtocolError("Content block must be an object")
    kind = block.get("type")
    if kind == "text":
        require_string(block.get("text"), "text block")
    elif kind == "thinking":
        require_string(block.get("thinking"), "thinking block")
        if block.get("signature") is not None:
            require_string(block["signature"], "thinking signature")
    elif kind == "tool_use":
        require_string(block.get("id"), "tool ID", nonempty=True)
        require_string(block.get("name"), "tool name", nonempty=True)
        if not isinstance(block.get("input"), dict):
            raise ProtocolError("Tool input must be a JSON object")
    else:
        raise ProtocolError(f"Unsupported content block type: {kind!r}")


def anthropic_message(value):
    if not isinstance(value, dict) or value.get("type") != "message" or value.get("role") != "assistant":
        raise ProtocolError("Expected an Anthropic assistant message")
    require_string(value.get("id"), "message ID", nonempty=True)
    require_string(value.get("model"), "message model", nonempty=True)
    blocks = value.get("content")
    if not isinstance(blocks, list):
        raise ProtocolError("Message content must be a list of blocks")
    for block in blocks:
        validate_block(block)
    stop = value.get("stop_reason")
    finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls"}
    if stop not in finish:
        raise ProtocolError(f"Missing or unsupported stop reason: {stop!r}")
    usage = value.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                                          for key in ("input_tokens", "output_tokens")):
        raise ProtocolError("Missing or invalid Anthropic token usage")
    return {"answer": "".join(block["text"] for block in blocks if block["type"] == "text"),
            "reasoning": "".join(block["thinking"] for block in blocks if block["type"] == "thinking"),
            "tool_calls": [{"id": block["id"], "function": {"name": block["name"],
                            "arguments": json.dumps(block["input"])}}
                           for block in blocks if block["type"] == "tool_use"],
            "usage": usage, "finish_reason": finish[stop], "stop_reason": stop,
            "content_blocks": blocks, "protocol_complete": True}


def anthropic_stream(lines):
    message, blocks, active, fragments = None, {}, set(), {}
    stopped, final_delta = False, False
    for event_name, value in sse_events(lines):
        if not isinstance(value, dict):
            raise ProtocolError("SSE data must be a JSON object")
        kind = value.get("type")
        if event_name is not None and event_name != kind:
            raise ProtocolError(f"SSE event/data types differ: {event_name!r} / {kind!r}")
        if kind == "ping":
            continue
        if kind == "error":
            raise ProtocolError(f"Anthropic stream error: {json.dumps(value.get('error'))}")
        if stopped:
            raise ProtocolError("Received content after message_stop")
        if kind == "message_start":
            if message is not None:
                raise ProtocolError("Duplicate message_start")
            message = copy.deepcopy(value.get("message"))
            if not isinstance(message, dict) or message.get("content") != []:
                raise ProtocolError("message_start must contain an empty message content list")
        elif message is None:
            raise ProtocolError("Content arrived before message_start")
        elif kind == "content_block_start":
            index = value.get("index")
            if type(index) is not int or index < 0 or index in blocks or final_delta:
                raise ProtocolError("Invalid or duplicate content block index")
            block = copy.deepcopy(value.get("content_block"))
            validate_block(block)
            blocks[index], fragments[index] = block, ""
            active.add(index)
        elif kind == "content_block_delta":
            index = value.get("index")
            if type(index) is not int or index not in active:
                raise ProtocolError("Delta references a block that is not open")
            delta = value.get("delta")
            if not isinstance(delta, dict):
                raise ProtocolError("Content delta must be an object")
            block = blocks[index]
            delta_kind = delta.get("type")
            if delta_kind == "text_delta" and block["type"] == "text":
                block["text"] += require_string(delta.get("text"), "text delta")
            elif delta_kind == "thinking_delta" and block["type"] == "thinking":
                block["thinking"] += require_string(delta.get("thinking"), "thinking delta")
            elif delta_kind == "signature_delta" and block["type"] == "thinking":
                block["signature"] = (block.get("signature") or "") + require_string(delta.get("signature"), "signature delta")
            elif delta_kind == "input_json_delta" and block["type"] == "tool_use":
                if block["input"]:
                    raise ProtocolError("Tool block mixes initial input and streamed JSON fragments")
                fragments[index] += require_string(delta.get("partial_json"), "tool JSON delta")
            else:
                raise ProtocolError(f"Delta {delta_kind!r} does not match block {block['type']!r}")
        elif kind == "content_block_stop":
            index = value.get("index")
            if type(index) is not int or index not in active:
                raise ProtocolError("Stop references a block that is not open")
            if fragments[index]:
                blocks[index]["input"] = json.loads(fragments[index])
            validate_block(blocks[index])
            active.remove(index)
        elif kind == "message_delta":
            if active:
                raise ProtocolError("Message finished before its content blocks closed")
            delta = value.get("delta")
            if not isinstance(delta, dict):
                raise ProtocolError("Message delta must be an object")
            if delta.get("stop_reason") is not None:
                for field in ("stop_reason", "stop_sequence"):
                    if field in delta:
                        message[field] = delta[field]
                final_delta = True
            if value.get("usage") is not None:
                if not isinstance(value["usage"], dict):
                    raise ProtocolError("Message usage must be an object")
                message["usage"] = (message.get("usage") or {}) | value["usage"]
        elif kind == "message_stop":
            if active or not final_delta or set(blocks) != set(range(len(blocks))):
                raise ProtocolError("message_stop arrived before complete block/final-delta assembly")
            message["content"] = [blocks[index] for index in sorted(blocks)]
            stopped = True
        else:
            raise ProtocolError(f"Unexpected Anthropic event: {kind!r}")
    if not stopped:
        raise ProtocolError("Stream ended without message_stop")
    return anthropic_message(message)


def openai_message(value):
    if not isinstance(value, dict) or value.get("object") != "chat.completion":
        raise ProtocolError("Expected an OpenAI chat completion")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ProtocolError("Expected one completion choice")
    message = choices[0]["message"]
    if message.get("role") != "assistant":
        raise ProtocolError("Expected an assistant completion message")
    return {"answer": message.get("content") or "", "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
            "tool_calls": message.get("tool_calls") or [], "finish_reason": choices[0].get("finish_reason"),
            "usage": value.get("usage") or {}, "protocol_complete": True}


def diagnostics(row):
    fields = ("answer", "reasoning", "tool_calls", "usage", "finish_reason", "stop_reason",
              "total_s", "ttft_s", "first_answer_s", "content_blocks", "protocol_complete")
    return {name: row[name] for name in fields if name in row}


def send_request(url, key, body, case, timeout):
    protocol = case["protocol"]
    endpoint = "/v1/chat/completions" if protocol == "openai" else "/v1/messages"
    headers = {"Content-Type": "application/json"}
    headers.update({"Authorization": f"Bearer {key}"} if protocol == "openai" else
                   {"x-api-key": key, "anthropic-version": "2023-06-01"})
    req = urllib.request.Request(url.rstrip("/").removesuffix("/v1") + endpoint,
                                 data=json.dumps(body).encode(), headers=headers)
    raw_lines, raw_value = [], None
    began = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if case["stream"]:
                def recorded():
                    for line in response:
                        raw_lines.append(line.decode("utf-8", errors="replace").rstrip("\r\n"))
                        yield line
                row = consume_stream(recorded(), began) if protocol == "openai" else anthropic_stream(recorded())
            else:
                raw_value = response.read().decode("utf-8", errors="replace")
                value = json.loads(raw_value)
                row = openai_message(value) if protocol == "openai" else anthropic_message(value)
        if protocol == "openai":
            if row.get("finish_reason") not in ("stop", "length", "tool_calls"):
                raise ProtocolError("Missing or unsupported OpenAI finish_reason")
            if type(row.get("usage", {}).get("completion_tokens")) is not int:
                raise ProtocolError("Missing OpenAI completion token usage")
        row["total_s"] = time.perf_counter() - began
        row["protocol_complete"] = True
        result = diagnostics(row)
    except urllib.error.HTTPError as exc:
        raw_value = exc.read().decode(errors="replace")
        result = {"error": f"HTTP {exc.code}: {raw_value}", "http_status": exc.code}
    except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError) as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    if "error" in result:
        result["total_s"] = time.perf_counter() - began
    if raw_lines:
        result["raw_sse"] = raw_lines
    if raw_value is not None:
        result["raw_response"] = raw_value
    return result


def quality(case, row):
    if row.get("error") or not row.get("protocol_complete"):
        return False
    try:
        if any(not isinstance(call.get("id"), str) or not call["id"] for call in row["tool_calls"]):
            return False
        return correct(case["kind"], row)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


def rejection_audit(target, result, audit):
    """A rejected frontend request never becomes an engine completion."""
    status = (result or {}).get("http_status")
    if status is None or not (400 <= status < 500 or status == 501) or "peak_active" not in audit:
        return audit
    before, after, peak = audit["before"], audit["after"], audit["peak_active"]
    observed = set(peak)
    previous = set(audit_reasons(target, before, after, peak, observed, 1))
    reasons = set(audit["reasons"]) - previous
    reasons.update(audit_reasons(target, before, after, peak, observed, 0))
    return audit | {"reasons": sorted(reasons), "expected_completions": 0}


def run(args):
    config = json.loads(subprocess.check_output([str(HERE / "llm"), "config"], text=True))
    backends = {model["name"]: f"http://127.0.0.1:{model['port']}/v1" for model in config["models"]}
    if args.model not in backends:
        raise ValueError("Target is missing from the configured backend registry")
    key = args.key_file.read_text().strip()
    guard = Guard(args.model, backends, args.max_idle_gpu_busy)
    report = {"at": timestamp(), "model": args.model, "gateway": args.url,
              "checkpoint": MODELS[args.model], "runtime": serving_metadata(args.model),
              "max_tokens_override": getattr(args, "max_tokens", None),
              "latency_is_diagnostic_only": True, "backends_observed": backends,
              "anthropic_thinking_control": "chat_template_kwargs.enable_thinking (vLLM extension)",
              "anthropic_structured_output": "not tested", "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        partial.write_text(json.dumps(report, indent=2) + "\n")
        partial.replace(args.output)

    for case in cases_for(args.model):
        entry = case | {"attempts": []}
        report["cases"].append(entry)
        if not case["required"]:
            entry.update(status="not_applicable", reason="Model capability is not advertised")
            save()
            continue
        body = request_body(args.model, case, getattr(args, "max_tokens", None))
        entry["request"] = body
        for attempt in range(args.max_attempts):
            try:
                result, audit = guard.trial(lambda: send_request(args.url, key, body, case, args.timeout), 1)
            except (OSError, ValueError, RuntimeError) as exc:
                result, audit = None, {"reasons": [f"{type(exc).__name__}: {exc}"]}
            audit = rejection_audit(args.model, result, audit)
            entry["attempts"].append({"attempt": attempt + 1, "at": timestamp(), "audit": audit, "result": result})
            save()
            retry = audit["reasons"]
            if result and result.get("error"):
                # Preserve a protocol/network failure even when its missing own
                # completion creates a counter mismatch. Peer overlap still retries.
                retry = [reason for reason in retry
                         if reason != "target completion count did not match 1 benchmark requests"]
            if retry:
                print(json.dumps({"case": case["id"], "discarded": retry}), flush=True)
                if attempt + 1 < args.max_attempts:
                    time.sleep(args.retry_delay)
                continue
            if result is None:
                continue
            entry["correct"] = quality(case, result)
            entry["status"] = ("passed" if entry["correct"] else "unsupported" if result.get("http_status")
                               in (404, 405, 501) else "failed")
            print(json.dumps({"case": case["id"], "status": entry["status"], "total_s": result.get("total_s")}), flush=True)
            break
        else:
            entry.update(status="failed", correct=False, reason="No uncontaminated idle window within the attempt limit")
        save()
    report["passed"] = all(entry["status"] == "passed" for entry in report["cases"] if entry["required"])
    report["completed_at"] = timestamp()
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--url", default="http://127.0.0.1:8400", help="Gateway root URL, optionally ending in /v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override each case's generation cap; otherwise retain its fixture default")
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--retry-delay", type=float, default=0.5)
    parser.add_argument("--max-idle-gpu-busy", type=float, default=20)
    parser.add_argument("--key-file", type=Path, default=Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "local-llm/api-key")
    args = parser.parse_args()
    if not 1 <= args.max_attempts <= 64 or args.timeout <= 0 or args.retry_delay < 0:
        parser.error("timeout must be positive, max-attempts 1–64, and retry-delay nonnegative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("max-tokens must be positive when supplied")
    address = urllib.parse.urlsplit(args.url)
    if (address.scheme not in ("http", "https") or not address.netloc or address.username or address.password
            or address.query or address.fragment or address.path.rstrip("/") not in ("", "/v1")):
        parser.error("url must be an HTTP(S) gateway root without credentials, query or fragment")
    lock_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "local-llm-benchmark.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another added-model benchmark owns the GPU timing lock")
        report = run(args)
    if not report["passed"]:
        raise SystemExit(f"Required API smoke cases failed; inspect {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
