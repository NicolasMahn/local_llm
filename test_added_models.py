import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.added_models import (CASES, DEEP_CASES, EXTENDED, EXTRA_CASES, INVENTORY, LEGACY, QUALITY, SUITE,
                                     Guard, audit_reasons, correct, deep_inventory, metric_counters,
                                     nonstream_request, payload, prefill_evidence,
                                     main, run, selected_cases, serving_metadata, summary, trial_payload)
from model_catalog import MODELS


def response(**changes):
    row = {"answer": "READY", "reasoning": "", "tool_calls": [], "finish_reason": "stop",
           "usage": {"completion_tokens": 3, "prompt_tokens": 100}, "total_s": 2,
           "ttft_s": 0.5, "first_answer_s": 0.5, "decode_tokens_per_s": 2,
           "answer_tokens": 3, "answer_tokens_per_s": 1.5}
    return row | changes


class AddedModelsTest(unittest.TestCase):
    def test_explicit_generation_budget_changes_only_max_tokens_for_every_supported_fixture(self):
        for name, model in MODELS.items():
            for case in selected_cases(["all", *EXTRA_CASES], model["capabilities"]):
                with self.subTest(model=name, case=case):
                    deep = {"reads": 0, "lines": 5}
                    default = trial_payload(name, case, "run", attempt=2, accepted=1, deep=deep)
                    unchanged = trial_payload(name, case, "run", attempt=2, accepted=1, max_tokens=None, deep=deep)
                    changed = trial_payload(name, case, "run", attempt=2, accepted=1, max_tokens=2048, deep=deep)
                    self.assertEqual(default, unchanged)
                    self.assertEqual(changed, default | {"max_tokens": 2048})
                    if name != "olmo-3.1-32b-think" and case not in DEEP_CASES:
                        self.assertEqual(default, payload(name, case, "run-2"))
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                trial_payload("olmo-3.1-32b-think", "reasoning_json", "run", 0, 0, invalid)

    def test_nonpositive_generation_budgets_are_refused_before_lock_or_gpu_requests(self):
        for value in ("0", "-1"):
            argv = ["added_models.py", "olmo-3.1-32b-think", "test", "--max-tokens", value,
                    "--output", "/tmp/unused-benchmark-result.json"]
            with self.subTest(value=value), patch("sys.argv", argv), \
                 patch("benchmarks.added_models.fcntl.flock") as lock, \
                 patch("benchmarks.added_models.run") as runner, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            lock.assert_not_called()
            runner.assert_not_called()

    def test_explicit_generation_budget_is_recorded_in_options_and_raw_request(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.runner_args(Path(directory), max_tokens=2048)
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.Guard", self.clean_guard()), \
                 patch("benchmarks.added_models.request", return_value=response()) as request, \
                 contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertTrue(report["passed"])
            self.assertEqual(report["options"]["max_tokens"], 2048)
            self.assertEqual(report["trials"][0]["request"]["max_tokens"], 2048)
            self.assertEqual(request.call_args.args[2]["max_tokens"], 2048)

    def test_extra_prefill_case_is_explicit_and_original_fixture_stays_unchanged(self):
        name = "granite-4.2-30b"
        self.assertEqual(CASES, ("text", "batch4", "prefill", *QUALITY))
        self.assertEqual(EXTRA_CASES, ("prefill_json", "reasoning_json", "prefill_ack", "batch16", *DEEP_CASES, "copy"))
        self.assertEqual(selected_cases(["all"], MODELS[name]["capabilities"]), list(CASES))
        self.assertNotIn("prefill_json", selected_cases(["quality"], MODELS[name]["capabilities"]))
        self.assertEqual(selected_cases(["prefill_json"], MODELS[name]["capabilities"]), ["prefill_json"])
        self.assertEqual(payload(name, "prefill", "nonce"), {
            "model": name, "messages": [{"role": "system", "content": "nonce. Benchmark request."},
            {"role": "user", "content": INVENTORY + "\nReturn only the integer code for item 298."}],
            "temperature": 0, "max_tokens": 128, "stream": True,
            "stream_options": {"include_usage": True}, "cache_salt": "nonce",
            "chat_template_kwargs": {"enable_thinking": False}})
        self.assertFalse(correct("prefill", response(answer="3881 is the answer.")))

    def test_schema_prefill_keeps_inventory_nonce_and_exact_integer_correctness(self):
        first = payload("nemotron-3.5-lightning-30b-a3b", "prefill_json", "first")
        second = payload("nemotron-3.5-lightning-30b-a3b", "prefill_json", "second")
        self.assertTrue(first["messages"][1]["content"].startswith(INVENTORY + "\n"))
        self.assertTrue(first["messages"][0]["content"].startswith("first."))
        self.assertNotEqual(first["cache_salt"], second["cache_salt"])
        self.assertEqual(first["messages"][1], second["messages"][1])
        self.assertEqual(first["response_format"], {"type": "json_schema", "json_schema": {
            "name": "inventory_code", "strict": True, "schema": {"type": "object",
            "properties": {"code": {"type": "integer"}}, "required": ["code"], "additionalProperties": False}}})
        self.assertTrue(correct("prefill_json", response(answer='{"code":3881}')))
        for answer in ('{"code":3881.0}', '{"code":"3881"}', '{"code":true}', '{"code":3882}',
                       '{"code":3881,"extra":0}', '3881', '{}', '{"code":3881} extra'):
            with self.subTest(answer=answer):
                self.assertFalse(correct("prefill_json", response(answer=answer)))
        self.assertFalse(correct("prefill_json", response(answer='{"code":3881}', finish_reason="length")))
        self.assertFalse(correct("prefill_json", response(answer='{"code":3881}', tool_calls=[{}])))

    def test_explicit_inventory_ack_has_its_own_strict_quality_contract(self):
        name = "olmo-3.1-32b-think"
        self.assertNotIn("prefill_ack", selected_cases(["all"], MODELS[name]["capabilities"]))
        self.assertNotIn("prefill_ack", selected_cases(["quality"], MODELS[name]["capabilities"]))
        self.assertEqual(selected_cases(["prefill_ack"], MODELS[name]["capabilities"]), ["prefill_ack"])
        body = payload(name, "prefill_ack", "visible")
        self.assertEqual(body["messages"], [
            {"role": "system", "content": "visible. Benchmark request."},
            {"role": "user", "content": INVENTORY + "\nAcknowledge receipt of this inventory. Reply with exactly READY."}])
        self.assertEqual(body["cache_salt"], "visible")
        self.assertEqual(body["max_tokens"], 1024)
        self.assertEqual(body["temperature"], .6)
        self.assertEqual(body["top_p"], .95)
        self.assertEqual(body["seed"], 42)
        self.assertNotIn("response_format", body)
        self.assertTrue(correct("prefill_ack", response()))
        for changes in ({"answer": "Inventory received. READY"}, {"answer": "3881"},
                        {"answer": "<think>read it</think>READY"}, {"tool_calls": [{}]},
                        {"finish_reason": "length"}, {"finish_reason": "tool_calls"}):
            with self.subTest(changes=changes):
                self.assertFalse(correct("prefill_ack", response(**changes)))

    def test_inventory_ack_gets_fresh_cache_proof_without_claiming_lookup_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.runner_args(Path(directory), model="olmo-3.1-32b-think", case=["prefill_ack"])
            row = response(usage={"prompt_tokens": 4000, "completion_tokens": 8,
                                 "prompt_tokens_details": {"cached_tokens": 0}})
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.Guard", self.clean_guard()), \
                 patch("benchmarks.added_models.request", return_value=row), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertTrue(report["passed"])
            self.assertEqual(set(report["summaries"]), {"prefill_ack"})
            self.assertTrue(report["trials"][0]["prefill"]["cache_freshness_verified"])
            self.assertIsNone(report["summaries"]["prefill_ack"]["median_server_uncached_prefill_tokens_per_s"])
        cached = response(usage={"prompt_tokens": 4000, "completion_tokens": 8,
                                 "prompt_tokens_details": {"cached_tokens": 16}})
        self.assertFalse(prefill_evidence([cached], {}, "olmo-3.1-32b-think")["cache_freshness_verified"])

    def test_explicit_reasoning_json_requires_real_math_and_separate_thought(self):
        name = "nemotron-3.5-lightning-30b-a3b"
        capabilities = MODELS[name]["capabilities"]
        self.assertNotIn("reasoning_json", selected_cases(["all"], capabilities))
        self.assertNotIn("reasoning_json", selected_cases(["quality"], capabilities))
        self.assertEqual(selected_cases(["reasoning_json"], capabilities), ["reasoning_json"])
        body = payload(name, "reasoning_json", "unused")
        prompt = body["messages"][0]["content"]
        self.assertIn("17 rows of 23 bolts", prompt)
        self.assertIn("Five bolts are removed", prompt)
        self.assertIn('integer field "remaining"', prompt)
        self.assertEqual(body["max_tokens"], 1024)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(body["response_format"], {"type": "json_schema", "json_schema": {
            "name": "remaining_bolts", "strict": True, "schema": {"type": "object",
            "properties": {"remaining": {"type": "integer"}}, "required": ["remaining"], "additionalProperties": False}}})
        self.assertNotIn("386", json.dumps(body))
        olmo = payload("olmo-3.1-32b-think", "reasoning_json", "unused")
        self.assertNotIn("chat_template_kwargs", olmo)
        good = response(answer='{"remaining":386}', reasoning="391 minus 5 gives 386.")
        self.assertTrue(correct("reasoning_json", good))
        for answer in ('{"remaining":391}', '{"remaining":false}', '{"remaining":386.0}',
                       '{"remaining":"386"}', '{"remaining":386,"extra":0}',
                       '<think>391-5</think>{"remaining":386}', '386'):
            with self.subTest(answer=answer):
                self.assertFalse(correct("reasoning_json", good | {"answer": answer}))
        self.assertFalse(correct("reasoning_json", good | {"reasoning": "  "}))
        self.assertFalse(correct("reasoning_json", good | {"finish_reason": "length"}))
        self.assertFalse(correct("reasoning_json", good | {"tool_calls": [{}]}))
        self.assertFalse(correct("reasoning", response(answer="386 is the answer.", reasoning="391 minus 5.")))

    def test_olmo_never_advertises_or_sends_standard_tools_or_thinking_toggle(self):
        name = "olmo-3.1-32b-think"
        self.assertFalse(any(case.startswith("tool_") for case in selected_cases(["all"], MODELS[name]["capabilities"])))
        self.assertNotIn("chat_template_kwargs", payload(name, "json", "fresh"))
        with self.assertRaisesRegex(ValueError, "does not advertise"):
            payload(name, "tool_auto", "fresh")

    def test_olmo_sampling_changes_no_other_models_or_existing_token_budgets(self):
        olmo = "olmo-3.1-32b-think"
        for case in selected_cases(["all", *EXTRA_CASES], MODELS[olmo]["capabilities"]):
            body = payload(olmo, case, "fixed", lines=5)
            self.assertEqual({key: body[key] for key in ("temperature", "top_p", "seed")},
                             {"temperature": .6, "top_p": .95, "seed": 42})
            self.assertEqual(body["max_tokens"], {"text": 256, "batch4": 128, "batch16": 128,
                                                  "deep_prefill": 1, "deep_decode": 128, "copy": 1424}.get(case, 1024))
            self.assertNotIn("chat_template_kwargs", body)
        for name in ("granite-4.2-30b", "nemotron-3.5-lightning-30b-a3b"):
            for case in selected_cases(["all", *EXTRA_CASES], MODELS[name]["capabilities"]):
                body = payload(name, case, "fixed", lines=5)
                self.assertEqual(body["temperature"], 0)
                self.assertNotIn("top_p", body)
                self.assertNotIn("seed", body)

    def test_olmo_visible_prefill_text_is_stable_between_variants_but_cache_salt_is_fresh(self):
        bodies = []

        def send(url, key, body, timeout):
            bodies.append(copy.deepcopy(body))
            return response(answer='{"code":3881}' if "response_format" in body else "3881",
                            usage={"prompt_tokens": 4000, "completion_tokens": 10,
                                   "prompt_tokens_details": {"cached_tokens": 0}})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.runner_args(root, model="olmo-3.1-32b-think", case=["prefill", "prefill_json"], repeats=2)
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.Guard", self.clean_guard()), \
                 patch("benchmarks.added_models.request", side_effect=send), \
                 patch("benchmarks.added_models.uuid.uuid4", side_effect=[SimpleNamespace(hex="native-run"), SimpleNamespace(hex="tuned-run")]), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = run(args)
                args.output = root / "tuned.json"
                second = run(args)
            self.assertTrue(first["passed"] and second["passed"])
            self.assertEqual(len(bodies), 8)
            self.assertEqual(len({body["messages"][0]["content"] for body in bodies[:4]}), 4)
            for before, after in zip(bodies[:4], bodies[4:]):
                self.assertEqual(before["messages"], after["messages"])
                self.assertNotEqual(before["cache_salt"], after["cache_salt"])
                self.assertTrue(before["cache_salt"].startswith("native-run-"))
                self.assertTrue(after["cache_salt"].startswith("tuned-run-"))
                self.assertEqual(before["seed"], after["seed"])
            self.assertTrue(all(trial["prefill"]["cache_freshness_verified"] for trial in first["trials"] + second["trials"]))

    def test_idle_rejections_do_not_change_olmo_seeded_prefill_prompt(self):
        for case in ("prefill", "prefill_json", "prefill_ack"):
            before = trial_payload("olmo-3.1-32b-think", case, "native-run", attempt=3, accepted=1)
            after = trial_payload("olmo-3.1-32b-think", case, "tuned-run", attempt=11, accepted=1)
            self.assertEqual(before["messages"], after["messages"])
            self.assertEqual(before["messages"][0]["content"], f"benchmark-{case}-1. Benchmark request.")
            self.assertNotEqual(before["cache_salt"], after["cache_salt"])
            self.assertEqual(before["cache_salt"], f"native-run-{case}-3")
            self.assertEqual(after["cache_salt"], f"tuned-run-{case}-11")
            self.assertEqual(before | {"cache_salt": None}, after | {"cache_salt": None})
            next_trial = trial_payload("olmo-3.1-32b-think", case, "native-run", attempt=4, accepted=2)
            self.assertNotEqual(before["messages"], next_trial["messages"])
        for name in ("granite-4.2-30b", "nemotron-3.5-lightning-30b-a3b"):
            self.assertEqual(trial_payload(name, "prefill", "original", attempt=7, accepted=1),
                             payload(name, "prefill", "original-7"))

    def test_reasoning_and_direct_cases_keep_the_same_fixture_across_variants(self):
        name = "granite-4.2-30b"
        direct = payload(name, "chat", "first")
        thought = payload(name, "reasoning", "second")
        self.assertFalse(direct["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(thought["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(thought["max_tokens"], 1024)
        first, second = payload(name, "prefill", "first"), payload(name, "prefill", "second")
        self.assertNotEqual(first["messages"], second["messages"])
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertTrue(first["messages"][0]["content"].startswith("first."))
        self.assertNotEqual(first["cache_salt"], second["cache_salt"])
        self.assertEqual(first["messages"][1], second["messages"][1])

    def test_strict_json_rejects_wrong_types_extra_fields_and_truncation(self):
        self.assertTrue(correct("json", response(answer='{"job":4711,"errors":0}')))
        for answer in ('{"job":4711,"errors":false}', '{"job":4711,"errors":0,"extra":1}',
                       '```json\n{"job":4711,"errors":0}\n```'):
            self.assertFalse(correct("json", response(answer=answer)))
        self.assertFalse(correct("json", response(answer='{"job":4711,"errors":0}', finish_reason="length")))
        schema = payload("granite-4.2-30b", "json", "fresh")["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])

    def test_reasoning_must_be_separate_and_final_arithmetic_must_be_right(self):
        self.assertTrue(correct("reasoning", response(answer="386", reasoning="391 minus 5 is 386.")))
        self.assertFalse(correct("reasoning", response(answer="386")))
        self.assertFalse(correct("reasoning", response(answer="<think>work</think>386")))
        self.assertFalse(correct("reasoning", response(answer="391", reasoning="work")))

    def test_tool_checks_require_a_single_correct_call_and_none_forbids_calls(self):
        row = response(answer="", finish_reason="tool_calls", tool_calls=[{"function": {
            "name": "report_job", "arguments": '{"job":4711,"errors":0}'}}])
        for case in ("tool_auto", "tool_required", "tool_named", "tool_named_nonstream"):
            self.assertTrue(correct(case, row))
        self.assertFalse(correct("tool_none", row))
        self.assertTrue(correct("tool_named", row | {"finish_reason": "stop"}))
        self.assertFalse(correct("tool_required", row | {"finish_reason": "stop"}))
        self.assertTrue(correct("tool_none", response()))
        duplicated = copy.deepcopy(row)
        duplicated["tool_calls"] *= 2
        self.assertFalse(correct("tool_auto", duplicated))
        wrong = copy.deepcopy(row)
        wrong["tool_calls"][0]["function"]["arguments"] = '{"job":4711,"errors":false}'
        self.assertFalse(correct("tool_auto", wrong))

    def test_peer_requests_and_new_backends_cannot_look_like_clean_trials(self):
        idle = {"running": 0, "waiting": 0, "completed": 10}
        before = {"target": idle, "peer": idle.copy()}
        after = {"target": idle | {"completed": 11}, "peer": idle.copy()}
        self.assertEqual(audit_reasons("target", before, after, {"target": 1, "peer": 0}, set(before), 1), [])
        after["peer"]["completed"] += 1
        self.assertTrue(audit_reasons("target", before, after, {"target": 1, "peer": 0}, set(before), 1))
        after["peer"]["completed"] -= 1
        self.assertTrue(audit_reasons("target", before, after, {"target": 1, "peer": 0}, set(before) | {"new"}, 1))

    def test_prefill_rate_requires_uncached_token_counts_and_server_timing(self):
        counters = {"vllm:request_prefill_time_seconds_sum": 10,
                    "vllm:request_prefill_time_seconds_count": 5, "vllm:prefix_cache_hits_total": 100}
        after = counters | {"vllm:request_prefill_time_seconds_sum": 12,
                            "vllm:request_prefill_time_seconds_count": 6}
        audit = {"before": {"target": {"counters": counters}}, "after": {"target": {"counters": after}}}
        row = response(usage={"prompt_tokens": 4000, "completion_tokens": 5,
                              "prompt_tokens_details": {"cached_tokens": 0}})
        evidence = prefill_evidence([row], audit, "target")
        self.assertTrue(evidence["cache_freshness_verified"])
        self.assertEqual(evidence["server_uncached_prefill_tokens_per_s"], 2000)
        # A launch without per-request cache details still has the backend's hit counter.
        row["usage"].pop("prompt_tokens_details")
        evidence = prefill_evidence([row], audit, "target")
        self.assertTrue(evidence["cache_freshness_verified"])
        self.assertEqual(evidence["server_uncached_prefill_tokens_per_s"], 2000)
        hit = audit | {"after": {"target": {"counters": after | {"vllm:prefix_cache_hits_total": 1100}}}}
        evidence = prefill_evidence([row], hit, "target")
        self.assertFalse(evidence["cache_freshness_verified"])
        self.assertEqual(evidence["server_uncached_prefill_tokens_per_s"], 1500)
        evidence = prefill_evidence([row], {}, "target")
        self.assertFalse(evidence["cache_freshness_verified"])
        self.assertIsNone(evidence["server_uncached_prefill_tokens_per_s"])

    def test_summaries_keep_total_generation_distinct_from_useful_output(self):
        a = response(answer_tokens=None, answer_tokens_per_s=None)
        value = summary([{"rows": [a], "elapsed_s": 2}])
        self.assertEqual(value["aggregate_total_output_tokens_per_s"], 1.5)
        self.assertIsNone(value["aggregate_useful_answer_tokens_per_s"])
        self.assertIsNone(value["median_useful_answer_tokens_per_s"])
        metrics = metric_counters('vllm:request_prefill_time_seconds_sum{model="a"} 2\n'
                                  'vllm:request_prefill_time_seconds_count{model="a"} 1\n'
                                  'vllm:unrelated_metric 99\n')
        self.assertEqual(metrics, {"vllm:request_prefill_time_seconds_sum": 2,
                                   "vllm:request_prefill_time_seconds_count": 1})

    def test_a_busy_backend_or_gpu_prevents_even_sending_the_trial(self):
        guard = Guard("target", {"target": "http://unused"})
        with patch.object(guard, "snapshot", return_value={"target": {
                "running": 1, "waiting": 0, "completed": 0}}), \
             patch("benchmarks.added_models.gpu_busy", return_value=0):
            call = unittest.mock.Mock()
            value, audit = guard.trial(call, 1)
            self.assertIsNone(value)
            self.assertTrue(audit["reasons"])
            call.assert_not_called()
        with patch.object(guard, "snapshot", return_value={"target": {
                "running": 0, "waiting": 0, "completed": 0}}), \
             patch("benchmarks.added_models.gpu_busy", return_value=95):
            call = unittest.mock.Mock()
            value, audit = guard.trial(call, 1)
            self.assertIsNone(value)
            self.assertIn("GPU was 95% busy before the trial", audit["reasons"])
            call.assert_not_called()

    def test_nonstreaming_results_have_total_latency_but_no_invented_token_timing(self):
        value = {"choices": [{"message": {"content": '{"job":4711,"errors":0}',
                  "reasoning": "Check the facts."}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                           "completion_tokens_details": {"reasoning_tokens": 8}}}
        with patch("benchmarks.added_models.urllib.request.urlopen", return_value=io.StringIO(json.dumps(value))), \
             patch("benchmarks.added_models.time.perf_counter", side_effect=[10, 12]):
            row = nonstream_request("http://unused/v1", "synthetic-key", {}, 10)
        self.assertTrue(correct("json_nonstream", row))
        self.assertEqual(row["total_s"], 2)
        self.assertEqual(row["answer_tokens_per_s"], 6)
        self.assertIsNone(row["decode_tokens_per_s"])
        self.assertIsNone(row["ttft_s"])
        self.assertIsNone(row["first_answer_s"])

    def test_runtime_metadata_keeps_only_serving_options_never_credentials(self):
        container = {"Image": "sha256:test", "ImageName": "local/test", "Config": {
            "Cmd": ["vllm", "serve", "/model"], "Env": ["LOCAL_LLM_FAST_KERNELS=0",
            "OMP_NUM_THREADS=1", "HF_TOKEN=synthetic-private-token", "LLM_API_KEY=synthetic-api-key"]}}
        result = unittest.mock.Mock(stdout=json.dumps([container]))
        with patch("benchmarks.added_models.subprocess.run", return_value=result):
            metadata = serving_metadata("example")
        self.assertEqual(metadata["environment"], {"LOCAL_LLM_FAST_KERNELS": "0", "OMP_NUM_THREADS": "1"})
        self.assertNotIn("synthetic-private-token", json.dumps(metadata))
        self.assertNotIn("synthetic-api-key", json.dumps(metadata))

    def test_keep_going_records_failure_once_and_measures_other_cases(self):
        for keep_going in (False, True):
            with self.subTest(keep_going=keep_going), tempfile.TemporaryDirectory() as directory:
                args = self.runner_args(Path(directory), case=["prefill", "chat"], keep_going=keep_going)
                rows = [response(answer="3881 is the answer."), response()]
                with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                     patch("benchmarks.added_models.serving_metadata", return_value={}), \
                     patch("benchmarks.added_models.Guard", self.clean_guard()), \
                     patch("benchmarks.added_models.request", side_effect=rows) as request, \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "Required cases failed" if keep_going else "Correctness check failed"):
                        run(args)
                report = json.loads(args.output.read_text())
                self.assertFalse(report["passed"])
                self.assertEqual(list(report["failures"]), ["prefill"])
                self.assertNotIn("prefill", report["summaries"])
                self.assertEqual(report["trials"][0]["result"]["rows"][0]["answer"], "3881 is the answer.")
                self.assertFalse(report["trials"][0]["correct"])
                self.assertEqual(request.call_count, 2 if keep_going else 1)
                self.assertEqual("chat" in report["summaries"], keep_going)

    def test_contaminated_bad_format_is_retained_but_only_a_later_clean_trial_is_accepted(self):
        class ContaminatedGuard:
            def __init__(self, *args):
                self.attempts = 0

            def trial(self, call, count):
                self.attempts += 1
                return call(), {"reasons": ["peer request overlapped"] if self.attempts == 1 else []}

        with tempfile.TemporaryDirectory() as directory:
            args = self.runner_args(Path(directory), case=["chat"])
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.Guard", ContaminatedGuard), \
                 patch("benchmarks.added_models.request", side_effect=[response(answer="READY."), response()]) as request, \
                 patch("benchmarks.added_models.time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertTrue(report["passed"])
            self.assertEqual(report["failures"], {})
            self.assertEqual(request.call_count, 2)
            self.assertEqual(report["summaries"]["chat"]["requests"], 1)
            discarded, accepted = json.loads(args.output.read_text())["trials"]
            self.assertEqual(discarded["result"]["rows"][0]["answer"], "READY.")
            self.assertFalse(discarded["correct"])
            self.assertEqual(discarded["audit"]["reasons"], ["peer request overlapped"])
            self.assertNotIn("accepted", discarded)
            self.assertTrue(accepted["correct"])
            self.assertTrue(accepted["accepted"])

    def test_request_error_still_fails_immediately_even_when_the_window_is_contaminated(self):
        class ContaminatedGuard:
            def __init__(self, *args):
                pass

            def trial(self, call, count):
                return call(), {"reasons": ["peer request overlapped"]}

        with tempfile.TemporaryDirectory() as directory:
            args = self.runner_args(Path(directory), case=["chat"])
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.Guard", ContaminatedGuard), \
                 patch("benchmarks.added_models.request", side_effect=RuntimeError("synthetic engine failure")) as request:
                with self.assertRaisesRegex(RuntimeError, "Request failed"):
                    run(args)
            self.assertEqual(request.call_count, 1)
            report = json.loads(args.output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(list(report["failures"]), ["chat"])
            self.assertEqual(len(report["trials"]), 1)
            self.assertEqual(report["trials"][0]["result"]["rows"][0]["error"], "synthetic engine failure")
            self.assertNotIn("chat", report["summaries"])

    def test_schema_prefill_summary_uses_only_correct_accepted_fresh_trials(self):
        class FreshGuard:
            attempts = 0

            def __init__(self, target, *args):
                self.target = target

            def trial(self, call, count):
                self.attempts += 1
                before = {"vllm:request_prefill_time_seconds_sum": 0,
                          "vllm:request_prefill_time_seconds_count": 0}
                after = {"vllm:request_prefill_time_seconds_sum": .01 if self.attempts == 1 else 2,
                         "vllm:request_prefill_time_seconds_count": 1}
                return call(), {"before": {self.target: {"counters": before}},
                    "after": {self.target: {"counters": after}},
                    "reasons": ["peer overlap"] if self.attempts == 1 else []}

        with tempfile.TemporaryDirectory() as directory:
            args = self.runner_args(Path(directory), case=["prefill_json"])
            row = response(answer='{"code":3881}', usage={"prompt_tokens": 100, "completion_tokens": 5,
                           "prompt_tokens_details": {"cached_tokens": 0}})
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.Guard", FreshGuard), \
                 patch("benchmarks.added_models.request", return_value=row), \
                 patch("benchmarks.added_models.time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["trials"]), 2)
            self.assertTrue(all(t["prefill"]["cache_freshness_verified"] for t in report["trials"]))
            self.assertNotIn("accepted", report["trials"][0])
            self.assertTrue(report["trials"][1]["accepted"])
            self.assertEqual(report["summaries"]["prefill_json"]["median_server_uncached_prefill_tokens_per_s"], 50)

    def test_every_served_model_runs_the_suite_with_only_the_cases_it_can_answer(self):
        self.assertEqual({name for name in SUITE if not name.endswith("-copy")}, set(LEGACY) | set(MODELS))
        self.assertEqual(len(set(LEGACY) | set(MODELS)), 16)
        self.assertIs(SUITE["qwen3-omni-30b-copy"], SUITE["qwen3-omni-30b"])
        omni = selected_cases(["extended"], SUITE["qwen3-omni-30b"]["capabilities"])
        # Omni has no thinking to separate, so a reasoning check could only fail.
        self.assertEqual(omni, [case for case in EXTENDED if case != "reasoning"])
        self.assertEqual(payload("qwen3-omni-30b", "chat", "n")["max_tokens"], 128)
        self.assertNotIn("chat_template_kwargs", payload("qwen3-omni-30b", "chat", "n"))
        # Muse always thinks, like Olmo, and needs the same room before its answer.
        self.assertEqual(payload("muse-glimmer-30b", "chat", "n")["max_tokens"], 1024)
        self.assertEqual(payload("gemma-4-31b", "reasoning", "n")["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(selected_cases(["extended"], SUITE["gemma-4-31b"]["capabilities"]), list(EXTENDED))

    def test_deep_cases_share_one_cached_prefix_and_differ_only_in_the_question(self):
        name = "granite-4.2-30b"
        read, decode, lookup = (payload(name, case, "salt", lines=40) for case in DEEP_CASES)
        # The fresh read and the decode are one prompt, so the decode starts fully cached.
        self.assertEqual(read["messages"], decode["messages"])
        self.assertEqual((read["max_tokens"], decode["max_tokens"]), (1, 128))
        self.assertEqual({body["cache_salt"] for body in (read, decode, lookup)}, {"salt"})
        prefix = deep_inventory(40) + "\n"
        self.assertTrue(all(body["messages"][1]["content"].startswith(prefix) for body in (read, decode, lookup)))
        self.assertEqual(read["messages"][0], lookup["messages"][0])
        self.assertIn("item 0020", lookup["messages"][1]["content"])
        self.assertTrue(correct("deep_lookup", response(answer='{"code":267}', deep_lines=40)))
        self.assertFalse(correct("deep_lookup", response(answer='{"code":3881}', deep_lines=40)))
        with self.assertRaisesRegex(ValueError, "inventory length"):
            payload(name, "deep_decode", "salt")

    def test_a_discarded_deep_read_is_repeated_under_a_new_salt_that_the_follow_ups_share(self):
        class OnceContaminated:
            attempts = 0

            def __init__(self, target, *args):
                self.target = target

            def trial(self, call, count):
                type(self).attempts += 1
                counters = {"vllm:request_prefill_time_seconds_sum": 0, "vllm:request_prefill_time_seconds_count": 0}
                after = {"vllm:request_prefill_time_seconds_sum": 60, "vllm:request_prefill_time_seconds_count": 1}
                return call(), {"before": {self.target: {"counters": counters}},
                                "after": {self.target: {"counters": after}},
                                "reasons": ["peer overlap"] if self.attempts == 1 else []}

        def send(url, key, body, timeout):
            cached = 0 if body["max_tokens"] == 1 else 23900
            return response(usage={"prompt_tokens": 24000, "completion_tokens": body["max_tokens"],
                                   "prompt_tokens_details": {"cached_tokens": cached}})

        with tempfile.TemporaryDirectory() as directory:
            args = self.runner_args(Path(directory), case=["deep_prefill", "deep_decode"], repeats=2, warmups=1)
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.deep_lines", return_value={"lines": 40, "prompt_tokens": 24000}), \
                 patch("benchmarks.added_models.Guard", OnceContaminated), \
                 patch("benchmarks.added_models.request", side_effect=send), \
                 patch("benchmarks.added_models.time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
        self.assertTrue(report["passed"])
        salts = [trial["request"]["cache_salt"] for trial in report["trials"]]
        # One fresh read is measured, however many repeats the other cases get.
        self.assertEqual([trial["case"] for trial in report["trials"]],
                         ["deep_prefill"] * 2 + ["deep_decode"] * 3)
        self.assertNotEqual(salts[0], salts[1])
        self.assertEqual(set(salts[1:]), {salts[1]})
        self.assertEqual(report["summaries"]["deep_prefill"]["median_server_uncached_prefill_tokens_per_s"], 400)
        self.assertEqual(report["summaries"]["deep_decode"]["prompt_tokens"], 24000)
        self.assertNotIn("Item 0001", json.dumps(report["trials"]))

    def test_a_deep_follow_up_that_missed_the_cache_is_not_reported_as_a_decode_rate(self):
        row = response(usage={"prompt_tokens": 24000, "completion_tokens": 128,
                              "prompt_tokens_details": {"cached_tokens": 512}})
        with tempfile.TemporaryDirectory() as directory:
            args = self.runner_args(Path(directory), case=["deep_decode"], keep_going=True)
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.deep_lines", return_value={"lines": 40, "prompt_tokens": 24000}), \
                 patch("benchmarks.added_models.Guard", self.clean_guard()), \
                 patch("benchmarks.added_models.request", return_value=row), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "deep_decode"):
                run(args)
            report = json.loads(args.output.read_text())
        self.assertIn("prefix cache", report["failures"]["deep_decode"])
        self.assertNotIn("deep_decode", report["summaries"])

    def test_copy_case_requires_the_twenty_prompt_lines_verbatim(self):
        body = payload("granite-4.2-30b", "copy", "n")
        self.assertEqual(body["max_tokens"], 528)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        lines = [f"Item {i:03}: code {i * 13 + 7}; state ready." for i in range(100, 120)]
        self.assertTrue(all(line in body["messages"][0]["content"] for line in lines))
        self.assertTrue(correct("copy", response(answer="\n".join(lines) + "\n")))
        self.assertFalse(correct("copy", response(answer="\n".join(lines[:-1]))))
        self.assertFalse(correct("copy", response(answer="\n".join(lines).replace("1307", "1308"))))
        self.assertFalse(correct("copy", response(answer="\n".join(lines), finish_reason="length")))
        self.assertNotIn("copy", selected_cases(["extended"], MODELS["granite-4.2-30b"]["capabilities"]))

    def test_sixteen_concurrent_requests_are_sent_and_audited_as_sixteen(self):
        counts = []

        class CountingGuard:
            def __init__(self, *args):
                pass

            def trial(self, call, count):
                counts.append(count)
                return call(), {"reasons": []}

        with tempfile.TemporaryDirectory() as directory:
            args = self.runner_args(Path(directory), case=["batch16"])
            with patch("benchmarks.added_models.subprocess.check_output", return_value=self.runner_config()), \
                 patch("benchmarks.added_models.serving_metadata", return_value={}), \
                 patch("benchmarks.added_models.Guard", CountingGuard), \
                 patch("benchmarks.added_models.request", return_value=response()) as request, \
                 contextlib.redirect_stdout(io.StringIO()):
                report = run(args)
        self.assertEqual(counts, [16])
        self.assertEqual(request.call_count, 16)
        self.assertEqual(report["summaries"]["batch16"]["requests"], 16)

    @staticmethod
    def runner_args(root, **overrides):
        key = root / "key"
        key.write_text("synthetic-private-key")
        return SimpleNamespace(**({"model": "nemotron-3.5-lightning-30b-a3b", "label": "test", "url": None,
            "key_file": key, "output": root / "result.json", "case": ["chat"], "repeats": 1,
            "warmups": 0, "timeout": 10, "max_attempts": 3, "max_idle_gpu_busy": 20,
            "keep_going": False} | overrides))

    @staticmethod
    def runner_config():
        return json.dumps({"models": [{"name": "nemotron-3.5-lightning-30b-a3b", "port": 8008}]})

    @staticmethod
    def clean_guard():
        class CleanGuard:
            def __init__(self, *args):
                pass

            def trial(self, call, count):
                return call(), {"reasons": []}

        return CleanGuard


if __name__ == "__main__":
    unittest.main()
