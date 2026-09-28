"""The failures the monitor looks out for, so they show before a request runs into them.

Each check turns something observed into a problem: a message, how bad it is
("error" or "warning") and since when. monitor.py writes them into its snapshot;
the gateway, the panel, the chat page and `./llm status` all show the same list,
and the gateway answers requests for a model that is not live with its state.
"""

import json
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024 ** 3
BUSY_WITHOUT_WORK = 300  # seconds of a busy GPU with nothing to do before it counts
RAM_LOW_GB = 4
HOT_C = 95
FIRST_START_LIMIT = 90 * 60  # seconds; a model with no start on record gets this long
TERMINATED = "terminated"  # a container's reason when a signal from outside ended it


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def duration(seconds):
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 90 else f"{minutes / 60:.1f} h"


def meminfo():
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }


class KernelLog:
    """GPU hangs the kernel logged this boot, followed as they happen.

    A hang ("ring ... timeout") makes the kernel reset the GPU, which kills
    whatever used it: models, the desktop, the browser.
    """

    def __init__(self):
        self.hangs = []  # epoch seconds
        threading.Thread(target=self._follow, daemon=True).start()

    def _follow(self):
        try:
            process = subprocess.Popen(
                ["journalctl", "-k", "-b", "-f", "-n", "all", "-o", "short-unix", "--grep", r"ring \S+ timeout"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return  # No journal access: this check stays quiet rather than wrong.
        for line in process.stdout:
            try:
                self.hangs.append(float(line.split()[0]))
            except (IndexError, ValueError):
                continue


class Containers:
    """The model containers as podman sees them, with the reason a crashed one gave."""

    def __init__(self):
        self._reasons = {}  # container id -> last error line

    def all(self):
        try:
            out = subprocess.run(
                ["podman", "ps", "-a", "--filter", "name=^llm-", "--format", "json"],
                capture_output=True, text=True, timeout=15).stdout
            listed = json.loads(out or "[]")
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {}
        return {entry["Names"][0].removeprefix("llm-"): entry for entry in listed}

    def reason(self, container):
        """Why a container ended: TERMINATED when it was stopped from outside,
        else the last "...Error: ..." line, skipping vLLM's generic wrapper
        that only points further up."""
        key = container["Id"]
        if key not in self._reasons:
            result = subprocess.run(["podman", "logs", "--tail", "300", key],
                                    capture_output=True, text=True)
            log = result.stdout + result.stderr
            if "KeyboardInterrupt: terminated" in log:
                self._reasons[key] = TERMINATED
                return TERMINATED
            errors = [line for line in log.splitlines()
                      if re.search(r"\w*Error: ", line) and "Engine core initialization failed" not in line]
            if errors:
                line = errors[-1]
                self._reasons[key] = line[re.search(r"\w*Error: ", line).start():].split(". ")[0][:160]
            else:
                self._reasons[key] = f"exit code {container['ExitCode']}, see ./llm logs"
        return self._reasons[key]


class Health:
    def __init__(self, state_dir):
        self.starts_file = state_dir / "starts.json"  # model -> seconds its last start took
        try:
            self.start_times = json.loads(self.starts_file.read_text())
        except (OSError, ValueError):
            self.start_times = {}
        self.kernel = KernelLog()
        self.containers = Containers()
        self.states = {}
        self.busy_since = None

    def check(self, now, models_up, requests, gpu, temperature_c):
        """Model states and problems for one sample.

        models_up maps each model to whether it answered; gpu holds busy percent
        and the VRAM/GTT sizes in bytes.
        """
        problems = []

        def problem(severity, message, since=None):
            problems.append({"severity": severity, "message": message, "since": since and iso(since)})

        containers = self.containers.all()
        states = {}
        for name, up in models_up.items():
            container = containers.get(name)
            if up:
                state = {"state": "live"}
                previous = self.states.get(name, {})
                # A start that was itself too slow must not become the yardstick.
                if previous.get("state") == "starting" and not previous.get("too_slow"):
                    self._remember_start(name, now - previous["started"])
            elif container and container["State"] == "running":
                state = {"state": "starting", "started": container["StartedAt"]}
                took = now - container["StartedAt"]
                last = self.start_times.get(name)
                limit = max(3 * last, 20 * 60) if last else FIRST_START_LIMIT
                if took > limit:
                    state["too_slow"] = True
                    usual = f"its last start took {duration(last)}" if last else "that is long even for a first start"
                    problem("warning", f"{name} has been starting for {duration(took)}; {usual}",
                            container["StartedAt"])
            elif container and container["State"] == "exited" and container["ExitCode"] != 0:
                reason = self.containers.reason(container)
                if reason == TERMINATED:
                    # A logout ends all of the user's processes unless lingering is on.
                    state = {"state": "off"}
                    problem("warning", f"{name} was stopped from outside while running, "
                            "e.g. by logging out; start it again with ./llm up", container.get("ExitedAt"))
                else:
                    state = {"state": "failed", "reason": reason}
                    problem("error", f"{name} crashed: {reason}", container.get("ExitedAt"))
            else:
                state = {"state": "off"}
            if container and container.get("Restarts"):
                problem("warning", f"{name} crashed and was restarted {container['Restarts']}×")
            states[name] = state
        self.states = states

        if self.kernel.hangs:
            last = self.kernel.hangs[-1]
            severity = "error" if now - last < 1800 else "warning"
            problem(severity, f"The GPU hung {len(self.kernel.hangs)}× this boot and was reset, last at "
                    f"{datetime.fromtimestamp(last).strftime('%H:%M')}; a reboot clears what it left behind",
                    last)

        starting = any(state["state"] == "starting" for state in states.values())
        if (gpu["busy"] or 0) >= 90 and not requests and not starting:
            self.busy_since = self.busy_since or now
            if now - self.busy_since > BUSY_WITHOUT_WORK:
                problem("warning", f"The GPU has been busy for {duration(now - self.busy_since)} "
                        "with nothing to do; a reboot clears this", self.busy_since)
        else:
            self.busy_since = None

        memory = meminfo()
        if memory["MemAvailable"] < RAM_LOW_GB * GIB:
            problem("error", f"RAM is almost full: {memory['MemAvailable'] / GIB:.1f} GB free")
        if memory["SwapTotal"] and memory["SwapFree"] < memory["SwapTotal"] / 2:
            problem("warning", f"{(memory['SwapTotal'] - memory['SwapFree']) / GIB:.1f} GB of swap in use; "
                    "the system is short of RAM")
        if gpu["gtt_total"] > memory["MemTotal"] + GIB:
            problem("error", f"The GPU may borrow {gpu['gtt_total'] / GIB:.0f} GB, but the system only has "
                    f"{memory['MemTotal'] / GIB:.0f} GB of RAM; is the BIOS still reserving VRAM?")
        if temperature_c and temperature_c >= HOT_C:
            problem("warning", f"The chip is at {temperature_c:.0f} °C")

        public = {name: {k: (iso(v) if k == "started" else v) for k, v in state.items() if k != "too_slow"}
                  for name, state in states.items()}
        return public, problems

    def _remember_start(self, name, seconds):
        self.start_times[name] = round(seconds)
        partial = self.starts_file.with_suffix(".partial")
        partial.write_text(json.dumps(self.start_times))
        partial.replace(self.starts_file)
