"""Observed variation must not become invented precision or transcript accuracy."""

import json
import math

import pytest

from benchmarks.consistency_stats import (
    normalize_transcript,
    numeric_summary,
    response_summary,
    word_error,
)


def test_latency_spread_uses_sample_deviation_and_preserves_observed_range():
    summary = numeric_summary([1, None, 2, 3])
    assert summary == {"n": 3, "median": 2, "min": 1, "max": 3, "mean": 2,
                       "sample_stdev": 1, "coefficient_of_variation": .5}
    assert "p90" not in summary


def test_tail_interpolation_requires_ten_available_samples():
    assert "p90" not in numeric_summary([None, *range(9)])
    assert numeric_summary(reversed(range(1, 11)))["p90"] == pytest.approx(9.1)


def test_empty_and_singleton_samples_do_not_claim_known_variation():
    empty = numeric_summary([None])
    assert empty["n"] == 0
    assert all(value is None for key, value in empty.items() if key != "n")
    single = numeric_summary([4])
    assert single["median"] == 4
    assert single["sample_stdev"] is None
    assert single["coefficient_of_variation"] is None
    assert numeric_summary([0, 0])["coefficient_of_variation"] is None


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_samples_cannot_escape_as_invalid_json_statistics(value):
    with pytest.raises(ValueError, match="finite"):
        numeric_summary([1, value])


def test_normalization_preserves_unicode_words_and_numeric_content():
    assert normalize_transcript("  STRAẞE, Café! ４２ — don't; whisper-local. ") == "strasse café 42 dont whisper local"
    assert normalize_transcript("cafe\u0301") == "café"
    assert normalize_transcript("किताब!") == "किताब"
    assert normalize_transcript("Job_4711 / errors: 0") == "job 4711 errors 0"
    assert normalize_transcript("one") != normalize_transcript("1")


@pytest.mark.parametrize("reference,hypothesis,errors,rate", [
    ("Hello, WORLD!", "hello world", 0, 0),
    ("the job is 4711", "the job was 4711", 1, .25),
    ("the job is 4711", "the job 4711", 1, .25),
    ("the job is 4711", "the new job is 4711", 1, .25),
    ("a b c", "b c a", 2, 2 / 3),
    ("a", "a b c", 2, 2),
    ("a b", "", 2, 1),
])
def test_word_error_uses_sequence_edit_distance(reference, hypothesis, errors, rate):
    result = word_error(reference, hypothesis)
    assert result["errors"] == errors
    assert result["rate"] == pytest.approx(rate)
    assert result["reference_words"] == len(normalize_transcript(reference).split())


def test_empty_reference_has_error_count_without_an_infinite_rate():
    assert word_error("...", "invented words") == {"errors": 2, "reference_words": 0, "rate": None}
    assert word_error("", "") == {"errors": 0, "reference_words": 0, "rate": None}


def test_response_failures_and_missing_data_cannot_look_like_fast_successes():
    rows = [
        {"ttft_s": 1, "total_s": 4, "answer": "Job 4711.", "finish_reason": "stop",
         "usage": {"completion_tokens": 5}},
        {"ttft_s": None, "total_s": 6, "answer": "job 4711!", "finish_reason": "length",
         "usage": None},
        {"error": "connection reset", "ttft_s": .01, "total_s": .02,
         "answer": "partial", "finish_reason": "length"},
        {"error": "", "total_s": .03},
        {"total_s": None},
    ]
    summary = response_summary(rows, reference="job 4711")
    assert summary["requests"] == 5
    assert summary["successful"] == 3
    assert summary["errors"] == 2
    assert summary["cap_hits"] == 2
    assert summary["missing_answers"] == 1
    assert summary["ttft_s"]["n"] == 1
    assert summary["total_s"]["median"] == 5
    assert summary["completion_tokens"]["n"] == 1
    assert summary["distinct_normalized_answers"] == 1
    assert summary["normalized_answer_counts"] == {"job 4711": 2}
    assert summary["word_error_rate"]["n"] == 2
    assert summary["word_error_rate"]["max"] == 0
    json.dumps(summary, allow_nan=False)


def test_response_summary_distinguishes_repeatability_from_accuracy():
    rows = [{"answer": "job 4711"}, {"answer": "job 4712"}, {"answer": "job 4712"}]
    summary = response_summary(rows, reference="job 4711")
    assert summary["distinct_normalized_answers"] == 2
    assert summary["normalized_answer_counts"] == {"job 4711": 1, "job 4712": 2}
    assert summary["reference_words"] == 2
    assert summary["word_error_rate"]["median"] == .5
    assert summary["word_errors"]["max"] == 1
    assert "word_error_rate" not in response_summary(rows)


def test_blank_answers_are_checked_but_missing_answers_are_not_invented():
    summary = response_summary([{"answer": ""}, {}], reference="expected words")
    assert summary["missing_answers"] == 1
    assert summary["normalized_answer_counts"] == {"": 1}
    assert summary["word_error_rate"]["n"] == 1
    assert summary["word_error_rate"]["median"] == 1
    empty_reference = response_summary([{"answer": "hallucinated"}], reference="")
    assert empty_reference["word_error_rate"]["n"] == 0
    assert empty_reference["word_errors"]["median"] == 1
    json.dumps(empty_reference, allow_nan=False)
