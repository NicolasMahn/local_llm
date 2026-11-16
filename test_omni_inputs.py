"""Serving timer counters must describe exactly one accepted request."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "benchmarks"))
from omni_inputs import cache_reasons, engine_measurements, input_counters


def test_timer_buckets_and_created_fields_cannot_pollute_sums():
    raw = '''# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_sum{engine="0"} 2.5
vllm:request_prefill_time_seconds_sum{engine="1"} 1.0
vllm:request_prefill_time_seconds_count{engine="0"} 3
vllm:request_prefill_time_seconds_bucket{le="inf"} 99
vllm:request_prefill_time_seconds_created 123456789
vllm:mm_cache_hits_total{engine="0"} 4
'''
    assert input_counters(raw) == {"vllm:request_prefill_time_seconds_sum": 3.5,
                                   "vllm:request_prefill_time_seconds_count": 3,
                                   "vllm:mm_cache_hits_total": 4}


def test_only_one_completed_timer_sample_can_be_attributed_to_a_request():
    before = {"vllm:request_prefill_time_seconds_sum": 8,
              "vllm:request_prefill_time_seconds_count": 4,
              "vllm:mm_cache_hits_total": 3}
    after = {**before, "vllm:request_prefill_time_seconds_sum": 8.25,
             "vllm:request_prefill_time_seconds_count": 5, "vllm:mm_cache_hits_total": 4}
    assert engine_measurements(before, after) == {"request_prefill_time_seconds": .25, "mm_cache_hits_total": 1}
    after["vllm:request_prefill_time_seconds_count"] = 6
    assert engine_measurements(before, after) == {"mm_cache_hits_total": 1}


def test_fresh_and_reused_cases_require_the_measured_cache_behavior():
    assert cache_reasons({}, media_count=0, reuse=False, warmed=True) == []
    assert cache_reasons({"mm_cache_hits_total": 0}, media_count=1, reuse=False, warmed=True) == []
    assert cache_reasons({"mm_cache_hits_total": 1}, media_count=1, reuse=False, warmed=True)
    assert cache_reasons({"mm_cache_hits_total": 2}, media_count=2, reuse=True, warmed=True) == []
    assert cache_reasons({"mm_cache_hits_total": 1}, media_count=2, reuse=True, warmed=True)
    assert cache_reasons({"mm_cache_hits_total": 0}, media_count=2, reuse=True, warmed=False) == []
    assert cache_reasons({}, media_count=1, reuse=False, warmed=False)
