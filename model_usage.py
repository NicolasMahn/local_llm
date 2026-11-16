"""Observed model use, persisted without prompts, answers or client identities.

Counters are baselined on first sight and after a backend restart. Daily totals
use UTC observation dates; active time is sampled wall time, not GPU time.
A benchmark's requests are not use: the suite names the model it measures in
a marker file, and the monitor leaves that model's requests out meanwhile.
"""

import argparse
import contextlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

COUNTERS = ("completed_requests", "input_tokens", "output_tokens")
FIELDS = (*COUNTERS, "sampled_active_seconds")
STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "local-llm"
BENCHMARK_MARKER = STATE / "benchmark.json"


@contextlib.contextmanager
def benchmark(model, marker=BENCHMARK_MARKER):
    """Tell the monitor that this process's requests to a model are a measurement."""
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"model": model, "pid": os.getpid()}))
    except OSError:
        pass  # a measurement must not fail because its requests get counted as use
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def benchmarked_model(marker=BENCHMARK_MARKER):
    """The model being measured right now, if the process that said so still runs."""
    try:
        entry = json.loads(marker.read_text())
        os.kill(entry["pid"], 0)
        return entry["model"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def timestamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def empty_totals():
    return {field: 0 for field in FIELDS}


def valid_counts(entry, fields):
    return isinstance(entry, dict) and all(
        isinstance(entry.get(key), (int, float)) and math.isfinite(entry[key]) and entry[key] >= 0
        for key in fields)


class ModelUsage:
    def __init__(self, path, boot_id):
        self.path = Path(path)
        self.boot_id = boot_id
        self.since = None
        self.days = {}
        self.last = {}
        self.last_used = {}
        self.error = None
        self._readable = True
        try:
            saved = json.loads(self.path.read_text())
            if (not isinstance(saved, dict) or saved.get("version") != 1
                    or not isinstance(saved.get("days"), dict)
                    or not isinstance(saved.get("last_used"), dict)
                    or not isinstance(saved.get("last"), dict)):
                raise ValueError("unrecognized usage ledger format")
            if any(not isinstance(models, dict) or any(not valid_counts(entry, FIELDS)
                                                      for entry in models.values())
                   for models in saved["days"].values()):
                raise ValueError("invalid daily usage counts")
            if any(not isinstance(entry, dict) or not valid_counts(entry.get("counters"), COUNTERS)
                   or not isinstance(entry.get("at"), (int, float)) or "active" not in entry
                   for entry in saved["last"].values()):
                raise ValueError("invalid usage counter baseline")
            self.since = saved["since"]
            self.days = saved["days"]
            self.last_used = saved["last_used"]
            if saved["boot_id"] == boot_id:
                self.last = saved["last"]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Keep serving/health measurements available without replacing
            # unreadable historical totals with an empty ledger.
            self.error = f"Cannot read model usage ledger: {exc}"
            self._readable = False

    def observe(self, name, counters, now, *, active=0, server_id=None, count=True):
        """count=False moves the baseline past requests that are not use."""
        if counters is None:
            self.last.pop(name, None)
            return
        if not valid_counts(counters, COUNTERS):
            self.last.pop(name, None)
            return
        self.since = self.since or timestamp(now)
        current = {key: int(counters[key]) for key in COUNTERS}
        previous = self.last.get(name)
        self.last[name] = {"counters": current, "at": now, "active": bool(active) and count,
                           "server_id": server_id}
        if not count:
            return
        day = timestamp(now)[:10]
        entry = self.days.setdefault(day, {}).setdefault(name, empty_totals())
        if active:
            self.last_used[name] = timestamp(now)
        if (previous is None or previous.get("server_id") != server_id
                or any(current[key] < previous["counters"][key] for key in COUNTERS)):
            return
        delta = {key: current[key] - previous["counters"][key] for key in COUNTERS}
        for key, value in delta.items():
            entry[key] += value
        if any(delta.values()):
            self.last_used[name] = timestamp(now)
        elapsed = now - previous["at"]
        # Long monitoring gaps cannot establish how long a model was active.
        if 0 < elapsed <= 30 and (active or previous["active"]):
            entry["sampled_active_seconds"] += elapsed

    def save(self):
        if not self._readable:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_suffix(".partial")
            partial.write_text(json.dumps({
                "version": 1, "since": self.since, "boot_id": self.boot_id,
                "days": self.days, "last": self.last, "last_used": self.last_used,
            }))
            os.replace(partial, self.path)
            self.error = None
        except OSError as exc:
            self.error = f"Cannot save model usage ledger: {exc}"

    def summary(self):
        totals = {}
        for models in self.days.values():
            for name, entry in models.items():
                target = totals.setdefault(name, empty_totals())
                for key in FIELDS:
                    target[key] += entry[key]
        for name, entry in totals.items():
            entry["sampled_active_seconds"] = round(entry["sampled_active_seconds"], 1)
            entry["last_used"] = self.last_used.get(name)
        return {
            "since": self.since, "day_timezone": "UTC", "models": totals,
            "days": [{"day": day, "models": self.days[day]} for day in sorted(self.days)],
            "error": self.error,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ledger = ModelUsage(STATE / "model-usage.json", Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    report = ledger.summary()
    if args.json:
        print(json.dumps(report))
        return
    if report["error"]:
        raise SystemExit(report["error"])
    print(f"Observed use since {report['since'] or 'not yet measured'} (UTC dates; active time is sampled)")
    for name, entry in sorted(report["models"].items(), key=lambda item: item[1]["completed_requests"], reverse=True):
        print(f"{name}: {entry['completed_requests']} completed requests, "
              f"{entry['input_tokens']} input / {entry['output_tokens']} output tokens, "
              f"{entry['sampled_active_seconds']:.0f} s active, last used {entry['last_used'] or 'not observed'}")


if __name__ == "__main__":
    main()
