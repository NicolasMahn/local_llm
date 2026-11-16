"""Print one comparison table from serving-suite reports (added_models.py).

    python3 benchmarks/suite_table.py logs/all-models-2026-11-14/*.json

Later reports for the same model and label replace the cases they repeat, so
a rerun of a few cases can be passed after the full run.
"""

import argparse
import json
import statistics
from pathlib import Path

QUALITY = ("chat", "reasoning", "json", "json_nonstream", "tool_auto", "tool_required",
           "tool_named", "tool_named_nonstream", "tool_none", "prefill", "deep_lookup", "copy")


def merged(paths):
    """One row per (model, label), in first-seen order."""
    rows = {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        row = rows.setdefault((report["model"], report["label"]),
                              {"summaries": {}, "failures": {}, "cases": [], "deep": None, "read": {}})
        for case in report["options"]["cases"]:
            row["summaries"].pop(case, None)
            row["failures"].pop(case, None)
            if case not in row["cases"]:
                row["cases"].append(case)
            # How fast a prompt was read does not depend on whether the answer
            # had the asked format, so a failed check still leaves its rate.
            rates = [trial["prefill"]["server_uncached_prefill_tokens_per_s"] for trial in report["trials"]
                     if trial["case"] == case and trial.get("prefill") and not trial["warmup"]
                     and not trial["audit"]["reasons"] and trial["prefill"]["cache_freshness_verified"]]
            rates = [rate for rate in rates if rate is not None]
            row["read"][case] = statistics.median(rates) if rates else None
        row["summaries"].update(report["summaries"])
        row["failures"].update(report["failures"])
        row["deep"] = report.get("deep") or row["deep"]
    return rows


def number(value, digits=1, unit=""):
    return "–" if value is None else f"{value:,.{digits}f}{unit}"


def metric(row, case, field):
    return row["summaries"].get(case, {}).get(field)


def table(rows):
    lines = ["| Model | Decode, short chat | 4 at once | 16 at once | Read, 4k prompt | Read, 24k prompt "
             "| Decode behind 24k | Cached 24k first token | Copying text | Checks passed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for (model, label), row in rows.items():
        asked = [case for case in QUALITY if case in row["cases"]]
        passed = [case for case in asked if case in row["summaries"]]
        failed = ", ".join(case for case in asked if case not in passed)
        lines.append("| " + " | ".join([
            f"{model} ({label})",
            number(metric(row, "text", "median_total_decode_tokens_per_s")),
            number(metric(row, "batch4", "aggregate_total_output_tokens_per_s")),
            number(metric(row, "batch16", "aggregate_total_output_tokens_per_s")),
            # Nemotron and Olmo fail the plain lookup's format; their schema and
            # acknowledgement variants read the same inventory.
            number(next((row["read"][case] for case in ("prefill", "prefill_json", "prefill_ack")
                         if row["read"].get(case) is not None), None), 0),
            number(row["read"].get("deep_prefill"), 0),
            number(metric(row, "deep_decode", "median_total_decode_tokens_per_s")),
            number(metric(row, "deep_decode", "median_ttft_s"), 2, " s"),
            number(metric(row, "copy", "median_total_decode_tokens_per_s")),
            f"{len(passed)}/{len(asked)}" + (f"; failed: {failed}" if failed else ""),
        ]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    print(table(merged(args.reports)))
    print("\nRates are tokens per second. Decode counts thinking tokens too; "
          "\"at once\" is the total over all concurrent requests.")


if __name__ == "__main__":
    main()
