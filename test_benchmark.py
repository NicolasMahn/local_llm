import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark import answer_token_count, consume_stream, contamination_reasons, media_part, parse_backend_metrics


def events(*values):
    return [b"data: " + json.dumps(value).encode() + b"\n" for value in values] + [b"data: [DONE]\n"]


class BenchmarkTest(unittest.TestCase):
    def test_browser_audio_containers_use_the_requested_audio_mime_type(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, mime in ((".webm", "audio/webm"), (".mp4", "audio/mp4")):
                path = Path(directory) / f"recording{suffix}"
                path.write_bytes(b"recording")
                part = media_part(path, "audio")
                self.assertEqual(part["type"], "audio_url")
                self.assertTrue(part["audio_url"]["url"].startswith(f"data:{mime};base64,"))
                with self.assertRaises(ValueError):
                    media_part(path, "image")

    def test_backend_counters_include_all_finish_reasons_but_ignore_created_and_reason_gauges(self):
        parsed = parse_backend_metrics('''# HELP ignored
vllm:num_requests_running{model="tiny"} 1.0
vllm:num_requests_waiting{model="tiny"} 0.0
vllm:num_requests_waiting_by_reason{reason="capacity"} 2.0
vllm:request_success_total{finished_reason="stop"} 10.0
vllm:request_success_total{finished_reason="abort"} 2.0
vllm:request_success_created{finished_reason="stop"} 1234.0
''')
        self.assertEqual(parsed, {"running": 1, "waiting": 0, "completed": 12})
        with self.assertRaisesRegex(ValueError, "do not include"):
            parse_backend_metrics('vllm:num_requests_running 0.0')

    def test_a_llama_cpp_peer_is_observed_instead_of_stopping_the_benchmark(self):
        parsed = parse_backend_metrics("llamacpp:n_decode_total 1763\n"
                                       "llamacpp:requests_processing 0\nllamacpp:requests_deferred 0\n")
        self.assertEqual(parsed, {"running": 0, "waiting": 0, "completed": 1763})

    def test_a_clean_trial_is_accepted_and_short_external_work_is_detected_by_counters(self):
        before = [{"running": 0, "waiting": 0, "completed": 10},
                  {"running": 0, "waiting": 0, "completed": 20}]
        after = [{"running": 0, "waiting": 0, "completed": 11}, before[1].copy()]
        self.assertEqual(contamination_reasons(before, after, [1, 0], 1), [])
        after[0]["completed"] = 12
        self.assertTrue(contamination_reasons(before, after, [1, 0], 1))
        after[0]["completed"] = 11
        after[1]["completed"] = 21
        self.assertTrue(contamination_reasons(before, after, [1, 0], 1))

    def test_unfinished_overlap_and_missing_completion_updates_are_rejected(self):
        idle = {"running": 0, "waiting": 0, "completed": 10}
        completed = {**idle, "completed": 11}
        busy = {**completed, "running": 1}
        self.assertTrue(contamination_reasons([idle], [busy], [2], 1))
        self.assertTrue(contamination_reasons([idle], [idle], [1], 1))
        self.assertTrue(contamination_reasons([busy], [completed], [1], 1))

    def test_role_and_usage_events_do_not_count_as_tokens(self):
        stream = events(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " there"}, "finish_reason": "length"}]},
            {"choices": [], "usage": {"completion_tokens": 4, "prompt_tokens": 20}},
        )
        with patch("benchmark.time.perf_counter", side_effect=[10, 13]):
            result = consume_stream(stream, 9)
        self.assertEqual(result["ttft_s"], 1)
        self.assertEqual(result["total_s"], 4)
        self.assertEqual(result["decode_tokens_per_s"], 1)
        self.assertEqual(result["answer"], "Hello there")
        self.assertEqual(result["finish_reason"], "length")

    def test_empty_output_has_no_invented_latency_or_throughput(self):
        with patch("benchmark.time.perf_counter", return_value=12):
            result = consume_stream(events({"choices": [], "usage": {"completion_tokens": 0}}), 9)
        self.assertIsNone(result["ttft_s"])
        self.assertIsNone(result["decode_tokens_per_s"])

    def test_reasoning_counts_and_stream_errors_are_reported(self):
        with patch("benchmark.time.perf_counter", side_effect=[10, 12]):
            result = consume_stream(events(
                {"choices": [{"delta": {"reasoning_content": "Thinking"}}]},
                {"usage": {"completion_tokens": 1}},
            ), 9)
        self.assertEqual(result["ttft_s"], 1)
        self.assertEqual(result["reasoning"], "Thinking")
        self.assertEqual(result["answer"], "")
        self.assertIsNone(result["first_answer_s"])
        self.assertEqual(result["answer_tokens"], 0)
        self.assertEqual(result["answer_tokens_per_s"], 0)
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            consume_stream(events({"error": {"message": "engine failed"}}), 9)

    def test_reasoning_and_answer_have_distinct_text_latency_and_throughput(self):
        stream = events(
            {"choices": [{"delta": {"reasoning": "Compute "}}]},
            {"choices": [{"delta": {"reasoning": "the sum.", "content": "42"}}]},
            {"usage": {"completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 8}}},
        )
        with patch("benchmark.time.perf_counter", side_effect=[10, 12, 14]):
            result = consume_stream(stream, 9)
        self.assertEqual(result["ttft_s"], 1)
        self.assertEqual(result["first_answer_s"], 3)
        self.assertEqual(result["total_s"], 5)
        self.assertEqual(result["reasoning"], "Compute the sum.")
        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["decode_tokens_per_s"], 9 / 4)
        self.assertEqual(result["answer_tokens"], 2)
        self.assertEqual(result["answer_tokens_per_s"], 2 / 5)

    def test_reasoning_without_a_token_split_does_not_invent_answer_throughput(self):
        for details in (None, {}, {"reasoning_tokens": 0}, {"reasoning_tokens": -1}, {"reasoning_tokens": 4}):
            with self.subTest(details=details):
                stream = events(
                    {"choices": [{"delta": {"reasoning_content": "Why", "content": "42"}}]},
                    {"usage": {"completion_tokens": 3, "completion_tokens_details": details}},
                )
                with patch("benchmark.time.perf_counter", side_effect=[10, 12]):
                    result = consume_stream(stream, 9)
                self.assertEqual(result["answer"], "42")
                self.assertEqual(result["reasoning"], "Why")
                self.assertIsNone(result["answer_tokens"])
                self.assertIsNone(result["answer_tokens_per_s"])

    def test_olmo_truncation_cannot_turn_thought_into_useful_answer_tokens(self):
        # Fields from olmo-3.1-32b-think-native-profile.json: the installed
        # parser emits thought but inherits a reasoning counter returning 0.
        usage = {"prompt_tokens": 84, "total_tokens": 340, "completion_tokens": 256,
                 "prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 80},
                 "completion_tokens_details": {"reasoning_tokens": 0}}
        thought = "Okay, the user wants a 20-paragraph explanation of how a compiler works."
        with patch("benchmark.time.perf_counter", side_effect=[10, 35]):
            result = consume_stream(events(
                {"choices": [{"delta": {"reasoning": thought}, "finish_reason": "length"}]},
                {"usage": usage},
            ), 9)
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["reasoning"], thought)
        self.assertEqual(result["answer_tokens"], 0)
        self.assertEqual(result["answer_tokens_per_s"], 0)
        self.assertIsNone(result["first_answer_s"])
        self.assertEqual(result["usage"], usage)
        self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 0)
        self.assertGreater(result["decode_tokens_per_s"], 0)

    def test_useful_token_count_is_shared_by_stream_and_nonstream(self):
        from benchmarks.added_models import nonstream_request

        usage = {"completion_tokens": 256, "completion_tokens_details": {"reasoning_tokens": 0}}
        for answer, thought, expected in (("", "Thinking", 0), ("42", "Thinking", None),
                                          ("", "", 0), ("42", "", 256)):
            with self.subTest(answer=answer, thought=thought):
                value = {"choices": [{"message": {"content": answer, "reasoning": thought},
                                     "finish_reason": "length"}], "usage": usage}
                with patch("benchmarks.added_models.urllib.request.urlopen",
                           return_value=io.BytesIO(json.dumps(value).encode())), \
                     patch("benchmarks.added_models.time.perf_counter", side_effect=[1, 3]):
                    result = nonstream_request("http://unused/v1", "unused", {}, 1)
                self.assertEqual(result["answer_tokens"], expected)
                self.assertEqual(result["answer_tokens_per_s"], expected / 2 if expected is not None else None)
                self.assertEqual(result["usage"], usage)
                self.assertEqual(answer_token_count(usage, answer, thought), expected)

    def test_empty_answer_is_zero_even_with_unknown_split_but_tools_stay_unknown(self):
        for usage in ({}, {"completion_tokens": 1024}, {"completion_tokens": 1024,
                      "completion_tokens_details": {"reasoning_tokens": 0}}):
            self.assertEqual(answer_token_count(usage, "", "Thinking"), 0)
            self.assertIsNone(answer_token_count(usage, "", "Thinking", [{}]))
        usage = {"completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 8}}
        self.assertEqual(answer_token_count(usage, "42", "Thinking"), 2)
        self.assertIsNone(answer_token_count(usage, "42", "Thinking", [{}]))

    def test_nonempty_final_cannot_have_all_completion_tokens_counted_as_thought(self):
        usage = {"completion_tokens": 479, "completion_tokens_details": {"reasoning_tokens": 479}}
        self.assertIsNone(answer_token_count(usage, "\nREADY", "Thinking"))
        self.assertEqual(answer_token_count(usage, "\n", "Thinking"), 0)

    def test_plain_final_text_counts_only_as_answer(self):
        with patch("benchmark.time.perf_counter", side_effect=[10, 13]):
            result = consume_stream(events(
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"usage": {"completion_tokens": 4}},
            ), 9)
        self.assertEqual(result["first_answer_s"], result["ttft_s"])
        self.assertEqual(result["reasoning"], "")
        self.assertEqual(result["answer_tokens"], 4)
        self.assertEqual(result["answer_tokens_per_s"], 1)

    def test_broken_stream_or_missing_usage_cannot_look_like_a_fast_response(self):
        with self.assertRaisesRegex(RuntimeError, "without \\[DONE\\]"):
            consume_stream(events({"usage": {"completion_tokens": 0}})[:-1], 9)
        with self.assertRaisesRegex(RuntimeError, "did not report token usage"):
            consume_stream(events({"choices": []}), 9)

    def test_streamed_tool_name_and_arguments_are_reconstructed(self):
        stream = events(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-test",
                "function": {"name": "report_", "arguments": '{"job":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"name": "job", "arguments": '4711,"errors":0}'}}]}, "finish_reason": "tool_calls"}]},
            {"usage": {"completion_tokens": 12}},
        )
        with patch("benchmark.time.perf_counter", side_effect=[10, 12]):
            result = consume_stream(stream, 9)
        self.assertEqual(result["tool_calls"], [{"id": "call-test", "function": {
            "name": "report_job", "arguments": '{"job":4711,"errors":0}'}}])
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertIsNone(result["decode_tokens_per_s"])
        self.assertIsNone(result["first_answer_s"])
        self.assertIsNone(result["answer_tokens_per_s"])


if __name__ == "__main__":
    unittest.main()
