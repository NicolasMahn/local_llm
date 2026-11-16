import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks import cache_followup as follow
from benchmarks.added_models import INVENTORY, audit_reasons


MODEL = "nemotron-3.5-lightning-30b-a3b"


def row(answer="3881", cached=0, elapsed=1):
    usage = {"prompt_tokens": 4200, "completion_tokens": 8,
             "completion_tokens_details": {"reasoning_tokens": 4}}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"answer": answer, "reasoning": "Synthetic private thought; exclude from history.",
            "tool_calls": [], "finish_reason": "stop", "usage": usage,
            "total_s": elapsed, "ttft_s": elapsed / 2, "first_answer_s": elapsed * .75}


def pair(elapsed=1, cached=4000):
    return {"first": row(elapsed=elapsed), "followup": row("3868", cached, elapsed),
            "sent_requests": 2}


def audit(names=(MODEL, "peer"), completions=2, peer_completed=0):
    before = {name: {"running": 0, "waiting": 0, "completed": 10} for name in names}
    after = copy.deepcopy(before)
    after[MODEL]["completed"] += completions
    if peer_completed:
        after[next(name for name in names if name != MODEL)]["completed"] += peer_completed
    peak = dict.fromkeys(names, 0)
    return {"before": before, "after": after, "peak_active": peak,
            "reasons": audit_reasons(MODEL, before, after, peak, set(peak), 2)}


class CacheFollowupTest(unittest.TestCase):
    def test_fresh_fixture_and_final_only_history_keep_one_nonce_per_pair(self):
        first = follow.first_payload(MODEL, "unique-first")
        other = follow.first_payload(MODEL, "unique-second")
        self.assertTrue(first["messages"][0]["content"].startswith("unique-first"))
        self.assertNotEqual(first["cache_salt"], other["cache_salt"])
        self.assertEqual(first["messages"][1]["content"], INVENTORY + "\nReturn only the integer code for item 298.")
        self.assertNotIn("response_format", first)
        self.assertEqual(first, follow.first_payload(MODEL, "unique-first", False))
        snapshot = copy.deepcopy(first)
        second = follow.followup_payload(first, " 3881\n")
        self.assertEqual(first, snapshot)
        self.assertEqual(second["messages"][:-2], first["messages"])
        self.assertEqual(second["messages"][-2], {"role": "assistant", "content": " 3881\n"})
        self.assertEqual(second["messages"][-1]["content"], "Return only the integer code for item 297.")
        self.assertEqual(second["cache_salt"], first["cache_salt"])
        with self.assertRaises(ValueError):
            follow.followup_payload(first, None)

    def test_thinking_control_preserves_olmo_always_reasoning_capability(self):
        self.assertEqual(follow.first_payload(MODEL, "n")["chat_template_kwargs"], {"enable_thinking": False})
        olmo = follow.first_payload("olmo-3.1-32b-think", "n")
        self.assertNotIn("chat_template_kwargs", olmo)
        self.assertEqual(olmo["max_tokens"], 1024)

    def test_olmo_cache_pair_inherits_catalog_sampling_in_plain_and_json_modes(self):
        for json_output in (False, True):
            body = follow.first_payload("olmo-3.1-32b-think", "fixed", json_output)
            second = follow.followup_payload(body, '{"code":3881}' if json_output else "3881", json_output)
            for item in (body, second):
                self.assertEqual({key: item[key] for key in ("temperature", "top_p", "seed")},
                                 {"temperature": .6, "top_p": .95, "seed": 42})
                self.assertEqual(item["max_tokens"], 1024)
                self.assertNotIn("chat_template_kwargs", item)
            for name in (MODEL, "granite-4.2-30b"):
                body = follow.first_payload(name, "fixed", json_output)
                self.assertEqual(body["temperature"], 0)
                self.assertNotIn("top_p", body)
                self.assertNotIn("seed", body)

    def test_json_pair_schema_preserves_actual_final_json_and_excludes_reasoning(self):
        first_answer = ' {"code": 3881}\n'
        with patch.object(follow, "stream_request", side_effect=[row(first_answer), row('{"code":3868}')]) as send:
            result = follow.execute_pair(MODEL, "json-nonce", "http://synthetic/v1", "private", 10, True)
        first, second = result["first_request"], result["followup_request"]
        self.assertTrue(first["messages"][1]["content"].startswith(INVENTORY + "\n"))
        self.assertIn("item 298", first["messages"][1]["content"])
        self.assertIn("item 297", second["messages"][-1]["content"])
        self.assertEqual(first["response_format"], {"type": "json_schema", "json_schema": {
            "name": "inventory_code", "strict": True, "schema": {"type": "object",
            "properties": {"code": {"type": "integer"}}, "required": ["code"], "additionalProperties": False}}})
        self.assertNotIn("3881", json.dumps(first["response_format"]))
        self.assertNotIn("3868", json.dumps(first["response_format"]))
        self.assertEqual(second["response_format"], first["response_format"])
        self.assertEqual(first["cache_salt"], second["cache_salt"])
        self.assertEqual(first["messages"][0], second["messages"][0])
        self.assertEqual(second["messages"][-2], {"role": "assistant", "content": first_answer})
        self.assertNotIn(result["first"]["reasoning"], json.dumps(second))
        self.assertEqual(first["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("chat_template_kwargs", follow.first_payload("olmo-3.1-32b-think", "n", True))
        self.assertEqual(send.call_count, 2)

    def test_json_correctness_requires_exact_typed_code_for_each_inventory_item(self):
        self.assertTrue(follow.integer_correct(row('{"code":3881}'), 3881, True))
        self.assertTrue(follow.integer_correct(row('{"code":3868}'), 3868, True))
        for answer in ('{"code":3868}', '{"code":true}', '{"code":3881.0}', '{"code":"3881"}',
                       '{"code":3881,"extra":0}', '3881', '<think>x</think>{"code":3881}'):
            with self.subTest(answer=answer):
                self.assertFalse(follow.integer_correct(row(answer), 3881, True))
        self.assertFalse(follow.integer_correct(row('{"code":3881}') | {"finish_reason": "length"}, 3881, True))
        self.assertFalse(follow.integer_correct(row('{"code":3881}') | {"tool_calls": [{}]}, 3881, True))

    def test_correctness_requires_complete_typed_integer_and_no_tool(self):
        self.assertTrue(follow.integer_correct(row(), 3881))
        for answer in ('"3881"', "3881.0", "true", "3882", "<think>x</think>3881", "3881 extra"):
            with self.subTest(answer=answer):
                self.assertFalse(follow.integer_correct(row(answer), 3881))
        self.assertFalse(follow.integer_correct(row() | {"finish_reason": "length"}, 3881))
        self.assertFalse(follow.integer_correct(row() | {"tool_calls": [{"id": "synthetic"}]}, 3881))

    def test_cache_accounting_distinguishes_unknown_zero_and_real_hits(self):
        self.assertEqual(follow.cached_tokens(row(cached=0)), 0)
        self.assertIsNone(follow.cached_tokens(row(cached=None)))
        for value in (True, -1, 4201, "4000"):
            self.assertIsNone(follow.cached_tokens(row(cached=value)))
        self.assertIsNone(follow.cached_tokens({"usage": {"prompt_tokens_details": "malformed"}}))
        result = follow.summarize([pair(1, 0), pair(3, None), pair(5, 4000)])
        self.assertEqual(result["median_followup_total_s"], 3)
        self.assertEqual(result["followup_cached_tokens"], [0, None, 4000])
        self.assertEqual(result["median_followup_cached_tokens"], 2000)
        self.assertEqual(result["pairs_with_reported_followup_cache"], 2)
        self.assertEqual(result["cache_hit_pair_fraction_when_reported"], .5)
        self.assertIsNone(follow.summarize([])["median_followup_total_s"])

    def test_pair_uses_actual_final_answer_and_retains_reasoning_separately(self):
        first = row(" 3881\n")
        with patch.object(follow, "stream_request", side_effect=[first, row("3868")]) as send:
            result = follow.execute_pair(MODEL, "nonce", "http://synthetic/v1", "private", 10)
        self.assertEqual(result["sent_requests"], 2)
        self.assertEqual(send.call_args_list[1].args[2]["messages"][-2]["content"], " 3881\n")
        self.assertNotIn(first["reasoning"], json.dumps(result["followup_request"]))
        self.assertEqual(result["first"]["reasoning"], first["reasoning"])
        with patch.object(follow, "stream_request", return_value=row("wrong")) as send:
            result = follow.execute_pair(MODEL, "nonce", "http://synthetic", "private", 10)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(result["sent_requests"], 1)
        self.assertIsNone(result["followup_request"])

    def test_stream_diagnostics_keep_raw_responses_without_pseudo_rates(self):
        events = [{"choices": [{"delta": {"reasoning": "think first"}}]},
                  {"choices": [{"delta": {"content": "3881"}, "finish_reason": "stop"}]},
                  {"usage": row()["usage"], "choices": []}]
        raw = b"".join(("data: " + json.dumps(event) + "\n\n").encode() for event in events) + b"data: [DONE]\n\n"
        with patch("benchmarks.api_smoke.urllib.request.urlopen", return_value=io.BytesIO(raw)):
            result = follow.stream_request("http://synthetic/v1", "synthetic-private-key", {}, 10)
        self.assertEqual(result["answer"], "3881")
        self.assertEqual(result["reasoning"], "think first")
        self.assertEqual(result["reported_cached_tokens"], 0)
        self.assertIn("raw_sse", result)
        self.assertFalse(any("per_s" in name for name in result))
        self.assertNotIn("synthetic-private-key", json.dumps(result))

    def test_partial_pair_audit_adjusts_only_own_expected_count(self):
        result = follow.pair_audit(MODEL, {"sent_requests": 1}, audit(completions=1))
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["expected_completions"], 1)
        contaminated = audit(completions=1, peer_completed=1)
        contaminated["reasons"].append("observer unavailable")
        result = follow.pair_audit(MODEL, {"sent_requests": 1}, contaminated)
        self.assertTrue(any("peer" in reason for reason in result["reasons"]))
        self.assertIn("observer unavailable", result["reasons"])
        self.assertTrue(follow.pair_audit(MODEL, pair(), audit(completions=3))["reasons"])

    def test_runner_discards_peer_overlap_and_excludes_warmups_from_medians(self):
        class AuditedGuard:
            observations = []
            calls = 0

            def __init__(self, target, backends, threshold):
                self.backends = backends
                self.observations.append(backends)

            def trial(self, call, count):
                self.assert_count = count
                if count != 2:
                    raise AssertionError("The guard must audit both requests")
                self.__class__.calls += 1
                result = call()
                return result, audit(self.backends, peer_completed=self.calls == 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, max_attempts=4)
            config = self.config()
            with patch.object(follow.subprocess, "check_output", return_value=json.dumps(config)), \
                 patch.object(follow, "Guard", AuditedGuard), \
                 patch.object(follow, "serving_metadata", return_value={"command": ["actual-launch"]}), \
                 patch.object(follow, "execute_pair", side_effect=[pair(100), pair(50), pair(2, 0), pair(4, 4000)]) as execute, \
                 contextlib.redirect_stdout(io.StringIO()):
                report = follow.run(args)
            self.assertTrue(report["passed"])
            self.assertEqual(report["output_format"], "plain")
            self.assertEqual(len(AuditedGuard.observations[0]), 9)
            self.assertEqual(len(report["trials"]), 4)
            self.assertEqual(len({call.args[1] for call in execute.call_args_list}), 4)
            self.assertEqual(report["summary"]["pairs"], 2)
            self.assertEqual(report["summary"]["median_followup_total_s"], 3)
            self.assertEqual(report["summary"]["followup_cached_tokens"], [0, 4000])
            self.assertEqual(report["runtime"]["command"], ["actual-launch"])
            self.assertNotIn("synthetic-private-key", args.output.read_text())
            self.assertEqual(json.loads(args.output.read_text())["passed"], True)

    def test_busy_windows_are_bounded_without_sending_or_stopping_anything(self):
        class BusyGuard:
            def __init__(self, *args):
                pass

            def trial(self, call, count):
                return None, {"reasons": ["a backend was busy before the trial"]}

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), max_attempts=2, warmups=0)
            with patch.object(follow.subprocess, "check_output", return_value=json.dumps(self.config())), \
                 patch.object(follow, "Guard", BusyGuard), \
                 patch.object(follow, "serving_metadata", return_value={}), \
                 patch.object(follow, "execute_pair") as execute, contextlib.redirect_stdout(io.StringIO()):
                report = follow.run(args)
            execute.assert_not_called()
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["trials"]), 2)
            self.assertIn("attempt limit", report["failure"])

    def test_no_cache_reuse_is_not_a_quality_failure(self):
        class CleanGuard:
            def __init__(self, *args):
                pass

            def trial(self, call, count):
                return call(), audit()

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), repeats=1, warmups=0)
            with patch.object(follow.subprocess, "check_output", return_value=json.dumps(self.config())), \
                 patch.object(follow, "Guard", CleanGuard), \
                 patch.object(follow, "serving_metadata", return_value={}), \
                 patch.object(follow, "execute_pair", return_value=pair(cached=0)), contextlib.redirect_stdout(io.StringIO()):
                report = follow.run(args)
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["pairs_with_followup_cache_hits"], 0)

    def test_json_run_is_labelled_and_checks_both_answers(self):
        class CleanGuard:
            def __init__(self, *args):
                pass

            def trial(self, call, count):
                return call(), audit()

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), repeats=1, warmups=0, json=True)
            result = pair()
            result["first"]["answer"] = '{"code":3881}'
            result["followup"]["answer"] = '{"code":3868}'
            with patch.object(follow.subprocess, "check_output", return_value=json.dumps(self.config())), \
                 patch.object(follow, "Guard", CleanGuard), \
                 patch.object(follow, "serving_metadata", return_value={}), \
                 patch.object(follow, "execute_pair", return_value=result) as execute, contextlib.redirect_stdout(io.StringIO()):
                report = follow.run(args)
            self.assertTrue(report["passed"])
            self.assertEqual(report["output_format"], "json")
            self.assertEqual(report["summary"]["output_format"], "json")
            self.assertIs(execute.call_args.args[-1], True)

    def test_required_answer_failure_retains_raw_error_and_fails_without_more_requests(self):
        class RejectedGuard:
            def __init__(self, *args):
                pass

            def trial(self, call, count):
                return call(), audit(completions=0)

        rejected = {"first": {"error": "HTTP 422: exact synthetic rejection", "http_status": 422,
                              "raw_response": "exact synthetic rejection"},
                    "followup": None, "sent_requests": 1}
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), repeats=1, warmups=0)
            with patch.object(follow.subprocess, "check_output", return_value=json.dumps(self.config())), \
                 patch.object(follow, "Guard", RejectedGuard), \
                 patch.object(follow, "serving_metadata", return_value={}), \
                 patch.object(follow, "execute_pair", return_value=rejected) as execute:
                report = follow.run(args)
            self.assertEqual(execute.call_count, 1)
            self.assertFalse(report["passed"])
            self.assertIn("required final integer", report["failure"])
            self.assertEqual(report["trials"][0]["result"]["first"]["raw_response"], "exact synthetic rejection")

    def test_shared_gpu_lock_refuses_before_any_request(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ["cache_followup.py", MODEL, "test", "--output", str(Path(directory) / "result.json")]
            with patch("sys.argv", argv), patch.dict("os.environ", {"XDG_RUNTIME_DIR": directory}), \
                 patch.object(follow.fcntl, "flock", side_effect=BlockingIOError), \
                 patch.object(follow, "run") as run:
                with self.assertRaisesRegex(SystemExit, "owns the GPU timing lock"):
                    follow.main()
            run.assert_not_called()

    @staticmethod
    def args(root, **options):
        key = root / "key"
        key.write_text("synthetic-private-key")
        return SimpleNamespace(**({"model": MODEL, "label": "synthetic", "url": None,
            "key_file": key, "output": root / "report.json", "repeats": 2, "warmups": 1,
            "timeout": 10, "max_attempts": 4, "retry_delay": 0, "max_idle_gpu_busy": 20} | options))

    @staticmethod
    def config():
        return {"models": [{"name": MODEL, "port": 8008}] +
                [{"name": f"peer-{i}", "port": 8100 + i} for i in range(8)]}


if __name__ == "__main__":
    unittest.main()
