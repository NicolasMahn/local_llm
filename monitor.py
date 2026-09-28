#!/usr/bin/env python3
"""Measures the machine and the models every few seconds, for the panel and the chat page.

Runs as a user service from login on (./llm monitor installs it) and writes one
JSON snapshot that both read. Energy and cost count from the last reboot: the
totals are saved with the boot's id, so a restarted monitor carries on and a
reboot starts from zero. Time before login, or while the monitor is stopped,
is not counted.

Energy comes from the chip's RAPL package counter (CPU, GPU and memory
controller together), which misses nothing between samples; it has to be made
readable once (README). Without it, the chip's time-filtered power reading is
sampled instead. Both come from the chip's own firmware, and the whole computer
draws more at the wall: memory chips, disk, fans and power-supply losses are
not in it, so the energy and cost here are a lower bound.
"""

import json
import os
import struct
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from health import Health

INTERVAL = 2  # seconds between samples
HISTORY_KEYS = ("time", "power_w", "gpu_w", "cpu_w", "rest_w", "mem_gpu_gb", "mem_other_gb",
                "gpu_busy", "tokens_per_second", "requests")
HISTORY = 300  # samples kept for the graphs: ten minutes
HERE = Path(__file__).resolve().parent
STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "local-llm"
STATS = STATE / "stats.json"
GATEWAY = "http://127.0.0.1:8400"
# The package domain. The "core" domain next to it counts CPU 0 only on AMD.
RAPL = Path("/sys/class/powercap/intel-rapl:0")

# Day-ahead prices for Germany (one market zone, so this is Esslingen's), €/MWh.
PRICE_URL = "https://api.awattar.de/v1/marketdata"
PRICE_REFRESH = 1800  # seconds; prices are fixed a day ahead, so this is plenty

# What a household in Esslingen pays per kWh on top of the market price, in
# ct/kWh before VAT, for 2026. Fixed monthly fees are left out: they do not
# grow with use. Sources are listed in the README.
SURCHARGES_CT = {
    "grid fee, Netze BW": 7.57,
    "concession fee, town of 25,000 to 100,000": 1.59,
    "electricity tax": 2.05,
    "CHP levy": 0.446,
    "offshore grid levy": 0.941,
    "special grid use levy (§19 StromNEV)": 1.559,
}
VAT = 1.19


def read_number(path):
    try:
        return float(Path(path).read_text())
    except (OSError, ValueError):
        return None


def system_memory():
    """Total and available system RAM in bytes; the VRAM carve-out is not in it."""
    info = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return info["MemTotal"], info["MemAvailable"]


def find_gpu():
    """The amdgpu device; its card number and hwmon index vary by boot."""
    for device in sorted(Path("/sys/class/drm").glob("card*/device")):
        if (device / "gpu_busy_percent").exists():
            hwmon = next((device / "hwmon").glob("hwmon*"), None)
            return device, hwmon
    return None, None


def metric(text, name):
    """Sum of a Prometheus metric over all its label sets, or None if absent."""
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith((name + "{", name + " "))
    ]
    return sum(values) if values else None


def fetch(url, timeout=1):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def boot():
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    btime = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime")
    )
    return boot_id, datetime.fromtimestamp(btime, timezone.utc).isoformat(timespec="seconds")


def models():
    output = subprocess.run([HERE / "llm", "list"], capture_output=True, text=True).stdout
    return {name: int(port) for name, port in (line.split() for line in output.splitlines())}


def power_split(device):
    """GPU and CPU-core watts from the chip firmware's metrics table, or None.

    Offsets are those of gpu_metrics_v3_0 (kernel: kgd_pp_interface.h); other
    table versions lay their fields out differently, so they get no split. Both
    values are the firmware's time-filtered figures; the core one is its
    activity-based estimate, not a measurement.
    """
    try:
        data = (device / "gpu_metrics").read_bytes()
    except (OSError, TypeError):
        return None
    size, major, minor = struct.unpack_from("<HBB", data)
    if (major, minor) != (3, 0) or size < 136:
        return None
    gpu_mw, = struct.unpack_from("<I", data, 124)  # average_gfx_power
    cores_mw, = struct.unpack_from("<I", data, 132)  # average_all_core_power
    return gpu_mw / 1000, cores_mw / 1000


class Rapl:
    """Joules the chip used since the last call, from a counter that wraps about every 18 Wh."""

    def __init__(self):
        self.range = read_number(RAPL / "max_energy_range_uj")
        self.last = read_number(RAPL / "energy_uj")

    def joules(self):
        now = read_number(RAPL / "energy_uj")
        last, self.last = self.last, now
        if now is None or last is None or not self.range:
            return None
        delta = now - last
        return (delta + self.range if delta < 0 else delta) / 1e6


class Prices:
    """Market price now, from a cached day-ahead list refreshed now and then."""

    def __init__(self):
        self.slots = []  # (start, end, €/MWh), times in epoch seconds
        self.fetched = 0

    def market_ct(self, now):
        if now - self.fetched > PRICE_REFRESH or not self._slot(now):
            try:
                data = json.loads(fetch(PRICE_URL, timeout=10))["data"]
                self.slots = [
                    (d["start_timestamp"] / 1000, d["end_timestamp"] / 1000, d["marketprice"])
                    for d in data
                ]
            except (OSError, ValueError, KeyError):
                pass  # Offline: keep the last list; a stale price beats none.
            self.fetched = now
        slot = self._slot(now) or (self.slots[-1] if self.slots else None)
        return slot[2] / 10 if slot else None  # €/MWh -> ct/kWh

    def _slot(self, now):
        return next((s for s in self.slots if s[0] <= now < s[1]), None)


def household_ct(market_ct):
    return (market_ct + sum(SURCHARGES_CT.values())) * VAT


class Monitor:
    def __init__(self):
        self.device, self.hwmon = find_gpu()
        self.models = models()
        self.rapl = Rapl()
        self.prices = Prices()
        self.boot_id, self.since = boot()
        self.tokens = {}  # model -> last generation token count, for rates
        self.energy_j = 0.0
        self.cost_eur = 0.0
        self.counting_since = None  # first sample of this boot; earlier time is not in the totals
        self.history = {key: [] for key in HISTORY_KEYS}
        STATE.mkdir(parents=True, exist_ok=True)
        self.health = Health(STATE)
        self._resume()

    def _resume(self):
        try:
            saved = json.loads(STATS.read_text())
        except (OSError, ValueError):
            return
        if saved.get("boot_id") == self.boot_id:
            self.energy_j = saved["energy_kwh"] * 3.6e6
            self.cost_eur = saved["cost_eur"]
            self.counting_since = saved.get("counting_since")
            saved_history = saved["history"]
            # Series added since the file was written start as zeros, so all stay aligned.
            self.history = {
                key: saved_history.get(key, [0] * len(saved_history["time"])) for key in HISTORY_KEYS
            }

    def sample(self, now, elapsed):
        device, hwmon = self.device, self.hwmon
        self.counting_since = self.counting_since or datetime.now(timezone.utc).isoformat(timespec="seconds")
        joules = self.rapl.joules()
        if joules is not None and elapsed:
            power_w, source = joules / elapsed, "RAPL package counter"
        else:
            power_w = (read_number(hwmon / "power1_average") or 0) / 1e6 if hwmon else 0
            joules, source = power_w * elapsed, "chip power reading"
        market = self.prices.market_ct(now)
        price = household_ct(market) if market is not None else None

        # "Rest" is whatever the total holds beyond GPU and cores: memory
        # controller, data links, I/O. Derived, so the parts add up to the total.
        split = power_split(device) if device else None
        if split:
            gpu_w, cpu_w = split
            parts = {"gpu": gpu_w, "cpu": cpu_w, "rest": max(0.0, power_w - gpu_w - cpu_w)}
        else:
            parts = None

        self.energy_j += joules
        if price is not None:
            self.cost_eur += joules / 3.6e6 * price / 100

        model_stats, total_rate, total_requests = {}, 0.0, 0
        for name, port in self.models.items():
            try:
                text = fetch(f"http://127.0.0.1:{port}/metrics")
            except OSError:
                model_stats[name] = {"up": False}
                self.tokens.pop(name, None)
                continue
            running = int(metric(text, "vllm:num_requests_running") or 0)
            waiting = int(metric(text, "vllm:num_requests_waiting") or 0)
            tokens = metric(text, "vllm:generation_tokens_total") or 0
            last = self.tokens.get(name)
            rate = max(0.0, tokens - last) / elapsed if last is not None and elapsed else 0.0
            self.tokens[name] = tokens
            total_rate += rate
            total_requests += running + waiting
            model_stats[name] = {
                "up": True,
                "running": running,
                "waiting": waiting,
                "cache": metric(text, "vllm:kv_cache_usage_perc") or 0,
                "tokens_per_second": round(rate, 1),
            }

        try:
            gateway_up = bool(fetch(GATEWAY + "/health"))
        except OSError:
            gateway_up = False

        gpu_busy = read_number(device / "gpu_busy_percent") if device else None
        # Strix Halo models live in both the VRAM carve-out and borrowed RAM (GTT).
        mem = {}
        for kind in ("vram", "gtt"):
            for end in ("used", "total"):
                mem[f"{kind}_{end}"] = (read_number(device / f"mem_info_{kind}_{end}") or 0) if device else 0
        used = mem["vram_used"] + mem["gtt_used"]

        # All installed memory: system RAM plus the VRAM carve-out. What the GPU
        # borrows from RAM (GTT) also shows as used RAM, so it is taken out of
        # "everything else" to count it once.
        ram_total, ram_available = system_memory()
        all_memory = ram_total + mem["vram_total"]
        other = max(0, ram_total - ram_available - mem["gtt_used"])
        celsius = read_number(hwmon / "temp1_input") if hwmon else None
        model_states, problems = self.health.check(
            now, {name: stats["up"] for name, stats in model_stats.items()}, total_requests,
            {"busy": gpu_busy, "gtt_total": mem["gtt_total"]}, celsius / 1000 if celsius else None)

        for key, value in (
            ("time", round(now)),
            ("power_w", round(power_w, 1)),
            ("gpu_w", round(parts["gpu"], 1) if parts else 0),
            ("cpu_w", round(parts["cpu"], 1) if parts else 0),
            ("rest_w", round(parts["rest"], 1) if parts else 0),
            ("mem_gpu_gb", round(used / 1024**3, 1)),
            ("mem_other_gb", round(other / 1024**3, 1)),
            ("gpu_busy", gpu_busy or 0),
            ("tokens_per_second", round(total_rate, 1)),
            ("requests", total_requests),
        ):
            self.history[key] = (self.history[key] + [value])[-HISTORY:]

        return {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "boot_id": self.boot_id,
            "since": self.since,
            "counting_since": self.counting_since,
            "interval_seconds": INTERVAL,
            "power_w": round(power_w, 1),
            "power_source": source,
            "power_parts_w": {k: round(v, 1) for k, v in parts.items()} if parts else None,
            "energy_kwh": self.energy_j / 3.6e6,
            "cost_eur": self.cost_eur,
            "market_ct_per_kwh": market,
            "price_ct_per_kwh": price,
            "gpu_busy": gpu_busy,
            "all_memory": {
                "total_gb": round(all_memory / 1024**3, 1),
                "gpu_gb": round(used / 1024**3, 1),
                "other_gb": round(other / 1024**3, 1),
            },
            "temperature_c": celsius / 1000 if celsius else None,
            "gateway_up": gateway_up,
            "tokens_per_second": round(total_rate, 1),
            "requests": total_requests,
            "models": model_stats,
            "model_states": model_states,
            "problems": problems,
            "history": self.history,
        }


def write(stats):
    # Replaced in one step, so readers never see half a file.
    STATE.mkdir(parents=True, exist_ok=True)
    partial = STATS.with_suffix(".partial")
    partial.write_text(json.dumps(stats))
    os.replace(partial, STATS)


def main():
    monitor = Monitor()
    last = time.monotonic()
    while True:
        time.sleep(INTERVAL)
        now = time.monotonic()
        write(monitor.sample(time.time(), now - last))
        last = now


if __name__ == "__main__":
    main()
