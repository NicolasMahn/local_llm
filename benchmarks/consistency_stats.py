"""Describe observed latency and transcript variation without hiding failures."""

from collections import Counter
import math
import statistics
import unicodedata


def numeric_summary(values):
    """Summarize available finite samples; undefined statistics stay ``None``.

    p90 uses linear interpolation at index ``0.9 * (n - 1)`` in the sorted
    samples. It is omitted below ten samples rather than implying a useful
    tail estimate from a handful of observations.
    """
    samples = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("Summary samples must be finite")
    count = len(samples)
    mean = statistics.mean(samples) if samples else None
    deviation = statistics.stdev(samples) if count > 1 else None
    result = {
        "n": count,
        "median": statistics.median(samples) if samples else None,
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
        "mean": mean,
        "sample_stdev": deviation,
        "coefficient_of_variation": deviation / abs(mean) if deviation is not None and mean else None,
    }
    if count >= 10:
        ordered = sorted(samples)
        position = 0.9 * (count - 1)
        lower = math.floor(position)
        fraction = position - lower
        result["p90"] = ordered[lower] + fraction * (ordered[math.ceil(position)] - ordered[lower])
    return result


def normalize_transcript(text):
    """Ignore case and punctuation while retaining Unicode words and digits.

    Apostrophes within contractions are ignored; hyphens and underscores
    separate words. Number words remain distinct from digits, so the summary
    cannot silently correct a transcription's numeric content.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("'", "").replace("’", "")
    words = "".join(character if character.isalnum() or unicodedata.category(character).startswith("M")
                    else " " for character in normalized)
    return " ".join(words.split())


def word_error(reference, hypothesis):
    """Return word-level Levenshtein errors and a JSON-safe error rate."""
    expected = normalize_transcript(reference).split()
    actual = normalize_transcript(hypothesis).split()
    previous = list(range(len(actual) + 1))
    for index, word in enumerate(expected, 1):
        current = [index]
        for column, observed in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (word != observed)))
        previous = current
    errors = previous[-1]
    return {"errors": errors, "reference_words": len(expected),
            "rate": errors / len(expected) if expected else None}


def response_summary(rows, reference=None):
    """Summarize responses without mixing failed attempts into successful timing.

    Missing timing, usage or answer fields contribute no invented samples.
    A token-limit finish is counted separately: it can be expected for a text
    throughput test while making a dictation incomplete.
    """
    rows = list(rows)
    successful = [row for row in rows if row.get("error") is None]
    answers = [row["answer"] for row in successful if isinstance(row.get("answer"), str)]
    normalized = Counter(normalize_transcript(answer) for answer in answers)
    result = {
        "requests": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "cap_hits": sum(row.get("finish_reason") == "length" for row in rows),
        "missing_answers": len(successful) - len(answers),
        "ttft_s": numeric_summary(row.get("ttft_s") for row in successful),
        "total_s": numeric_summary(row.get("total_s") for row in successful),
        "completion_tokens": numeric_summary((row.get("usage") or {}).get("completion_tokens")
                                              for row in successful),
        "distinct_normalized_answers": len(normalized),
        "normalized_answer_counts": dict(sorted(normalized.items())),
    }
    if reference is not None:
        comparisons = [word_error(reference, answer) for answer in answers]
        result.update(
            reference_words=len(normalize_transcript(reference).split()),
            word_error_rate=numeric_summary(item["rate"] for item in comparisons),
            word_errors=numeric_summary(item["errors"] for item in comparisons),
        )
    return result
