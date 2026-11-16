import contextlib
import copy
import ast
import importlib.machinery
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import types
import unittest
from unittest.mock import Mock, patch
import urllib.error

from benchmarks.api_smoke import (ProtocolError, anthropic_message, anthropic_stream,
                                  cases_for, diagnostics, main, quality,
                                  rejection_audit, request_body, run, send_request,
                                  sse_events)
from benchmarks.added_models import audit_reasons


MODEL = "granite-4.2-30b"


def message(content=None, stop="end_turn"):
    return {"id": "msg-synthetic", "type": "message", "role": "assistant", "model": MODEL,
            "content": content or [], "stop_reason": stop, "usage": {"input_tokens": 5, "output_tokens": 10}}


def sse(*events):
    lines = []
    for value in events:
        lines += [f"event: {value['type']}\r\n".encode(),
                  ("data: " + json.dumps(value) + "\r\n").encode(), b"\r\n"]
    return lines


def start():
    return {"type": "message_start", "message": message(stop=None)}


def end(stop="end_turn"):
    return [{"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 10}},
            {"type": "message_stop"}]


def thought_events():
    return [start(), {"type": "content_block_start", "index": 0,
                     "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "17 times 23 is 391. "}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "391 minus 5 is 386."}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "signature_delta", "signature": "synthetic-signature"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "3"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "86"}},
            {"type": "content_block_stop", "index": 1}, *end()]


def tool_events():
    return [start(), {"type": "content_block_start", "index": 0,
                     "content_block": {"type": "tool_use", "id": "call-synthetic", "name": "report_job", "input": {}}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": '{"job":47'}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": '11,"errors":0}'}},
            {"type": "content_block_stop", "index": 0}, *end()]


def case(kind="chat", protocol="anthropic", stream=True):
    return {"id": "synthetic", "kind": kind, "protocol": protocol, "stream": stream, "required": True}


def passing_row(item):
    kind = item["kind"]
    answer = "386" if kind == "reasoning" else '{"job":4711,"errors":0}' if kind == "json" else "READY"
    calls = []
    if kind.startswith("tool_"):
        answer = ""
        calls = [{"id": "call-synthetic", "function": {"name": "report_job", "arguments": '{"job":4711,"errors":0}'}}]
    return {"answer": answer, "reasoning": "391 minus 5 is 386." if kind == "reasoning" else "",
            "tool_calls": calls, "finish_reason": "tool_calls" if kind == "tool_auto" else "stop",
            "usage": {"completion_tokens": 10}, "protocol_complete": True, "total_s": 1.0}


class ApiSmokeTest(unittest.TestCase):
    def test_anthropic_stream_keeps_thinking_and_final_text_in_separate_complete_blocks(self):
        row = anthropic_stream(sse(*thought_events()))
        self.assertEqual(row["answer"], "386")
        self.assertEqual(row["reasoning"], "17 times 23 is 391. 391 minus 5 is 386.")
        self.assertEqual(row["content_blocks"][0]["signature"], "synthetic-signature")
        self.assertEqual(row["usage"], {"input_tokens": 5, "output_tokens": 10})
        self.assertTrue(quality(case("reasoning"), row))

    def test_streamed_tool_json_is_assembled_and_checked_for_exact_types(self):
        row = anthropic_stream(sse(*tool_events()))
        self.assertEqual(row["tool_calls"][0]["id"], "call-synthetic")
        self.assertTrue(quality(case("tool_named"), row))
        row["tool_calls"][0]["function"]["arguments"] = '{"job":4711,"errors":false}'
        self.assertFalse(quality(case("tool_named"), row))
        self.assertFalse(quality(case("tool_named"), {**row, "tool_calls": [{"function": {}}]}))

    def test_incomplete_or_mistyped_sse_cannot_pass_as_a_complete_response(self):
        events = thought_events()
        variants = [events[:-1], events[:-2] + [events[-1]],
                    [start(), {"type": "content_block_delta", "index": 3,
                               "delta": {"type": "text_delta", "text": "386"}}, *end()]]
        wrong_delta = copy.deepcopy(events)
        wrong_delta[2]["delta"] = {"type": "text_delta", "text": "leaked thought"}
        variants.append(wrong_delta)
        final_before_stop = copy.deepcopy(events)
        final_before_stop.pop(-3)
        variants.append(final_before_stop)
        for values in variants:
            with self.subTest(values=values), self.assertRaises(ProtocolError):
                anthropic_stream(sse(*values))
        malformed_tool = tool_events()
        malformed_tool[3]["delta"]["partial_json"] = "not-json"
        with self.assertRaises(ValueError):
            anthropic_stream(sse(*malformed_tool))

    def test_sse_framing_checks_event_names_errors_comments_and_multiline_data(self):
        values = list(sse_events([b": keepalive\n", b"event: ping\n", b"data: {\n", b'data: "type": "ping"}\n', b"\n"]))
        self.assertEqual(values, [("ping", {"type": "ping"})])
        with self.assertRaisesRegex(ProtocolError, "types differ"):
            anthropic_stream([b"event: ping\n", b'data: {"type":"message_stop"}\n', b"\n"])
        with self.assertRaisesRegex(ProtocolError, "unsupported test"):
            anthropic_stream(sse({"type": "error", "error": {"message": "unsupported test"}}))

    def test_nonstreamed_anthropic_reasoning_and_tools_use_real_block_shapes(self):
        row = anthropic_message(message([{"type": "thinking", "thinking": "Calculate 391 - 5.", "signature": "s"},
                                         {"type": "text", "text": "386"}]))
        self.assertTrue(quality(case("reasoning", stream=False), row))
        row = anthropic_message(message([{"type": "tool_use", "id": "call-synthetic", "name": "report_job",
                                           "input": {"job": 4711, "errors": 0}}], "tool_use"))
        self.assertTrue(quality(case("tool_named", stream=False), row))
        with self.assertRaisesRegex(ProtocolError, "token usage"):
            anthropic_message(message() | {"usage": {"input_tokens": 1, "output_tokens": True}})

    def test_payloads_use_supported_template_extension_and_named_tool_contract(self):
        thinking = request_body(MODEL, case("reasoning"))
        self.assertEqual(thinking["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("thinking", thinking)
        self.assertNotIn("response_format", thinking)
        tool = request_body(MODEL, case("tool_named"))
        self.assertEqual(tool["tool_choice"], {"type": "tool", "name": "report_job", "disable_parallel_tool_use": True})
        self.assertEqual(tool["tools"][0]["input_schema"]["properties"]["job"]["type"], "integer")
        olmo = "olmo-3.1-32b-think"
        self.assertNotIn("chat_template_kwargs", request_body(olmo, case("reasoning")))
        self.assertTrue(all(not item["required"] for item in cases_for(olmo) if item["kind"].startswith("tool_")))
        self.assertFalse(any(item["kind"] == "json" and item["protocol"] == "anthropic" for item in cases_for(MODEL)))

    def test_olmo_sampling_uses_supported_fields_for_each_protocol(self):
        olmo = "olmo-3.1-32b-think"
        for protocol in ("openai", "anthropic"):
            for stream in (False, True):
                body = request_body(olmo, case("reasoning", protocol=protocol, stream=stream))
                self.assertEqual(body["temperature"], .6)
                self.assertEqual(body["top_p"], .95)
                self.assertEqual(body["max_tokens"], 1024)
                self.assertNotIn("chat_template_kwargs", body)
                if protocol == "openai":
                    self.assertEqual(body["seed"], 42)
                else:
                    self.assertNotIn("seed", body)
        for name in (MODEL, "nemotron-3.5-lightning-30b-a3b"):
            for protocol in ("openai", "anthropic"):
                body = request_body(name, case("reasoning", protocol=protocol))
                self.assertEqual(body["temperature"], 0)
                self.assertNotIn("top_p", body)
                self.assertNotIn("seed", body)

    def test_token_override_changes_only_the_cap_for_both_protocols_and_stream_modes(self):
        for model in (MODEL, "nemotron-3.5-lightning-30b-a3b", "olmo-3.1-32b-think"):
            for item in cases_for(model):
                if not item["required"]:
                    continue
                with self.subTest(model=model, case=item["id"]):
                    default = request_body(model, item)
                    explicit_none = request_body(model, item, None)
                    self.assertEqual(json.dumps(explicit_none), json.dumps(default))
                    expected_cap = 1024 if model == "olmo-3.1-32b-think" or item["kind"] == "reasoning" else 128
                    self.assertEqual(default["max_tokens"], expected_cap)
                    overridden = request_body(model, item, 2048)
                    self.assertEqual(overridden, default | {"max_tokens": 2048})

    def test_anthropic_payloads_validate_against_installed_vllm_schema_when_available(self):
        spec = importlib.machinery.PathFinder.find_spec("vllm")
        if spec is None:
            self.skipTest("Installed serving schema is checked in the runtime image")
        location = Path(spec.origin).parent / "entrypoints/anthropic/protocol.py"
        source = ast.parse(location.read_text())
        # Importing vLLM itself probes ROCm. Load its actual schema definitions
        # on CPU, replacing only the unrelated stop-string environment limit.
        source.body = [node for node in source.body if not (
            isinstance(node, ast.Import) and any(alias.name == "vllm.envs" for alias in node.names))]
        module = types.ModuleType("cpu_anthropic_schema")
        module.envs = SimpleNamespace(VLLM_MAX_STOP_STRINGS=1000)
        with patch.dict("sys.modules", {module.__name__: module}):
            exec(compile(source, str(location), "exec"), module.__dict__)
        AnthropicMessagesRequest = module.AnthropicMessagesRequest
        for model in (MODEL, "nemotron-3.5-lightning-30b-a3b", "olmo-3.1-32b-think"):
            for item in cases_for(model):
                if item["required"] and item["protocol"] == "anthropic":
                    body = request_body(model, item)
                    parsed = AnthropicMessagesRequest.model_validate(body)
                    self.assertEqual(parsed.model, model)
                    self.assertEqual(parsed.chat_template_kwargs, body.get("chat_template_kwargs"))

    def test_sender_uses_gateway_routes_retains_exact_http_error_and_never_records_key(self):
        error = urllib.error.HTTPError("http://gateway/v1/messages", 404, "missing", {},
                                       io.BytesIO(b'{"error":{"message":"messages unsupported"}}'))
        with patch("benchmarks.api_smoke.urllib.request.urlopen", side_effect=error) as call:
            result = send_request("http://gateway/v1", "synthetic-private-key", {}, case(), 10)
        self.assertEqual(call.call_args.args[0].full_url, "http://gateway/v1/messages")
        self.assertEqual(result["http_status"], 404)
        self.assertIn("messages unsupported", result["error"])
        self.assertEqual(result["raw_response"], '{"error":{"message":"messages unsupported"}}')
        self.assertNotIn("synthetic-private-key", json.dumps(result))

    def test_openai_stream_outputs_are_retained_without_throughput_claims(self):
        events = [{"choices": [{"delta": {"role": "assistant"}}]},
                  {"choices": [{"delta": {"content": "READY"}, "finish_reason": "stop"}]},
                  {"usage": {"prompt_tokens": 5, "completion_tokens": 2}, "choices": []}]
        raw = b"".join(("data: " + json.dumps(event) + "\n\n").encode() for event in events) + b"data: [DONE]\n\n"
        with patch("benchmarks.api_smoke.urllib.request.urlopen", return_value=io.BytesIO(raw)):
            result = send_request("http://gateway", "private", {}, case(protocol="openai"), 10)
        self.assertTrue(quality(case(protocol="openai"), result))
        self.assertIn("raw_sse", result)
        self.assertFalse(any("per_s" in name or "tokens_per" in name for name in result))
        self.assertNotIn("decode_tokens_per_s", diagnostics({"decode_tokens_per_s": 999, "answer": "READY"}))

    def test_invalid_nonstream_protocol_body_is_preserved_for_diagnosis(self):
        raw = b"synthetic invalid JSON response"
        with patch("benchmarks.api_smoke.urllib.request.urlopen", return_value=io.BytesIO(raw)):
            result = send_request("http://gateway", "private", {}, case(stream=False), 10)
        self.assertIn("JSONDecodeError", result["error"])
        self.assertEqual(result["raw_response"], raw.decode())

    def test_rejected_requests_use_zero_completion_audit_and_still_catch_external_work(self):
        before = {name: {"running": 0, "waiting": 0, "completed": 10} for name in ("target", "peer")}
        after = copy.deepcopy(before)
        peak = {"target": 0, "peer": 0}
        audit = {"before": before, "after": after, "peak_active": peak,
                 "reasons": audit_reasons("target", before, after, peak, set(peak), 1)}
        result = rejection_audit("target", {"http_status": 422}, audit)
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["expected_completions"], 0)
        after["target"]["completed"] += 1
        self.assertTrue(rejection_audit("target", {"http_status": 422}, audit)["reasons"])

    def test_runner_continues_after_unsupported_case_audits_all_backends_and_omits_key(self):
        class AuditedGuard:
            seen = []

            def __init__(self, target, backends, threshold):
                self.target, self.backends = target, backends
                self.seen.append(backends)

            def trial(self, call, count):
                value = call()
                before = {name: {"running": 0, "waiting": 0, "completed": 10} for name in self.backends}
                after = copy.deepcopy(before)
                completed = 0 if value.get("error") else 1
                after[self.target]["completed"] += completed
                peak = {name: completed if name == self.target else 0 for name in self.backends}
                return value, {"before": before, "after": after, "peak_active": peak,
                               "reasons": audit_reasons(self.target, before, after, peak, set(peak), count)}

        def send(url, key, body, item, timeout):
            if item["id"] == "anthropic_chat_nonstream":
                return {"error": "HTTP 404: exact synthetic error", "http_status": 404,
                        "raw_response": "exact synthetic error"}
            return passing_row(item)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "api-key"
            key.write_text("synthetic-private-key")
            args = SimpleNamespace(model=MODEL, url="http://gateway", key_file=key, output=root / "smoke.json",
                                   timeout=10, max_attempts=2, retry_delay=0, max_idle_gpu_busy=20)
            config = {"models": [{"name": MODEL, "port": 8007}] +
                      [{"name": f"peer-{i}", "port": 8100 + i} for i in range(8)]}
            with patch("benchmarks.api_smoke.subprocess.check_output", return_value=json.dumps(config)), \
                 patch("benchmarks.api_smoke.Guard", AuditedGuard), \
                 patch("benchmarks.api_smoke.serving_metadata", return_value={}), \
                 patch("benchmarks.api_smoke.send_request", side_effect=send), contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertFalse(report["passed"])
            self.assertIsNone(report["max_tokens_override"])
            self.assertEqual(len(AuditedGuard.seen[0]), 9)
            self.assertEqual(len(report["cases"]), 15)
            self.assertEqual(report["cases"][-1]["status"], "passed")
            unsupported = next(item for item in report["cases"] if item["id"] == "anthropic_chat_nonstream")
            self.assertEqual(unsupported["status"], "unsupported")
            self.assertEqual(len(unsupported["attempts"]), 1)
            self.assertNotIn("synthetic-private-key", args.output.read_text())

    def test_runner_records_and_passes_the_explicit_generation_cap(self):
        guard = Mock()
        guard.trial.side_effect = lambda call, count: (call(), {"reasons": []})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "key"
            key.write_text("private")
            args = SimpleNamespace(model=MODEL, url="http://gateway", key_file=key, output=root / "report.json",
                                   timeout=10, max_attempts=1, retry_delay=0, max_idle_gpu_busy=20, max_tokens=2048)
            items = [case("tool_named", protocol=protocol, stream=stream)
                     for protocol in ("openai", "anthropic") for stream in (True, False)]
            with patch("benchmarks.api_smoke.subprocess.check_output", return_value=json.dumps({"models": [{"name": MODEL, "port": 8007}]})), \
                 patch("benchmarks.api_smoke.Guard", return_value=guard), patch("benchmarks.api_smoke.serving_metadata", return_value={}), \
                 patch("benchmarks.api_smoke.cases_for", return_value=items), \
                 patch("benchmarks.api_smoke.send_request", side_effect=lambda url, key, body, item, timeout: passing_row(item)) as send, \
                 contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(report["max_tokens_override"], 2048)
            self.assertEqual(json.loads(args.output.read_text())["max_tokens_override"], 2048)
            self.assertEqual(send.call_count, 4)
            self.assertTrue(all(call.args[2]["max_tokens"] == 2048 for call in send.call_args_list))
            self.assertTrue(all(entry["request"]["max_tokens"] == 2048 for entry in report["cases"]))

    def test_contaminated_windows_retry_to_bound_without_sending_more_requests(self):
        guard = Mock()
        guard.trial.return_value = (None, {"reasons": ["GPU was busy before the trial"]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "key"
            key.write_text("private")
            args = SimpleNamespace(model=MODEL, url="http://gateway", key_file=key, output=root / "report.json",
                                   timeout=10, max_attempts=2, retry_delay=0, max_idle_gpu_busy=20)
            with patch("benchmarks.api_smoke.subprocess.check_output", return_value=json.dumps({"models": [{"name": MODEL, "port": 8007}]})), \
                 patch("benchmarks.api_smoke.Guard", return_value=guard), patch("benchmarks.api_smoke.serving_metadata", return_value={}), \
                 patch("benchmarks.api_smoke.cases_for", return_value=[case()]), patch("benchmarks.api_smoke.time.sleep"), \
                 patch("benchmarks.api_smoke.send_request") as send, contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["cases"][0]["attempts"]), 2)
            send.assert_not_called()

    def test_main_refuses_a_contended_gpu_lock_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"XDG_RUNTIME_DIR": directory}), \
                 patch("sys.argv", ["api_smoke.py", MODEL, "--output", str(Path(directory) / "report.json")]), \
                 patch("benchmarks.api_smoke.fcntl.flock", side_effect=BlockingIOError), \
                 patch("benchmarks.api_smoke.run") as execute:
                with self.assertRaisesRegex(SystemExit, "owns the GPU timing lock"):
                    main()
                execute.assert_not_called()

    def test_nonpositive_generation_caps_are_rejected_before_lock_or_requests(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), \
                 patch("sys.argv", ["api_smoke.py", MODEL, "--output", "unused.json", "--max-tokens", value]), \
                 patch("benchmarks.api_smoke.fcntl.flock") as lock, \
                 patch("benchmarks.api_smoke.run") as execute, \
                 patch("benchmarks.api_smoke.send_request") as send, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                lock.assert_not_called()
                execute.assert_not_called()
                send.assert_not_called()

    def test_required_failures_return_nonzero_after_the_report_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"XDG_RUNTIME_DIR": directory}), \
                 patch("sys.argv", ["api_smoke.py", MODEL, "--output", str(Path(directory) / "report.json")]), \
                 patch("benchmarks.api_smoke.run", return_value={"passed": False}):
                with self.assertRaisesRegex(SystemExit, "Required API smoke cases failed"):
                    main()


if __name__ == "__main__":
    unittest.main()
