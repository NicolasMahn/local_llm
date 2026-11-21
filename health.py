"""The failures the monitor looks out for, so they show before a request runs into them.

Each check turns something observed into a problem: a message, how bad it is
("error" or "warning") and since when. monitor.py writes them into its snapshot;
the gateway, the panel, the chat page and `./llm status` all show the same list,
and the gateway answers requests for a model that is not live with its state.
"""

import ipaddress
import json
import os
import re
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024 ** 3
BUSY_WITHOUT_WORK = 300  # seconds of a busy GPU with nothing to do before it counts
NO_BREAK = 3600  # seconds a model may answer without a moment idle before it looks stuck
RAM_LOW_GB = 4
HOT_C = 95
DISK_LOW_GB = 25
DISK_CRITICAL_GB = 4
FIRST_START_LIMIT = 90 * 60  # seconds; a model with no start on record gets this long
TERMINATED = "terminated"  # a container's reason when a signal from outside ended it
ERROR_LINE = re.compile(r"\w*Error: ")  # Python's exceptions, from vLLM
# llama.cpp marks errors with " E "; the first of its last start is the cause,
# the ones after it only report that loading failed.
LLAMA_CPP_START = "llama_server: initializing"
LLAMA_CPP_LOADED = "server_run_i: model loaded"
KILLED = "killed (SIGKILL), most likely for lack of memory; journalctl -k shows it"
LLAMA_CPP_ERROR = re.compile(r"^\S+ E (?:\w+\s+)?(\w+: .*)$", re.M)
LLAMA_CPP_INFO = re.compile(r"^\S+ I .*$", re.M)
# ggml's abort, written straight to stderr; it can be the only line a GPU
# fault leaves, because llama.cpp's own log is written by a thread that dies
# with the process.
GGML_ABORT = re.compile(r"^\S*?([\w.-]+:\d+): (.*)$", re.M)
# Where podman keeps each running container's runtime files, including the
# one its monitor process writes the exit code to.
PODMAN_PERSIST = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "libpod/tmp/persist"
# The public name ./llm web serves; no file, no way in from outside.
WEB_NAME = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "local-llm/web-name"
NAME_CHECK_EVERY = 600  # seconds; DNS changes slowly and a lookup can hang for a while
STORAGE_CHECK_EVERY = 30
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "local-llm"
STORAGE_CONFIG = CONFIG_DIR / "storage.json"
HF_CACHE = Path.home() / ".cache/huggingface"


def _storage_paths(config):
    try:
        entries = json.loads(Path(config).read_text())["paths"]
        if not isinstance(entries, list) or any(
                not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) and entry[key]
                                                     for key in ("path", "mount", "uuid"))
                or not Path(entry["path"]).is_absolute() or not Path(entry["mount"]).is_absolute()
                for entry in entries):
            raise ValueError("expected absolute path, mount and UUID for each storage path")
        return entries, None
    except FileNotFoundError:
        return [], None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [], f"Cannot read storage configuration {config}: {exc}"


def _mount_error(entry):
    mount = Path(entry["mount"])
    try:
        # Looking at mountinfo first avoids walking into an absent systemd
        # automount, which could otherwise block every monitor sample.
        mounted = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields, filesystem = line.split(" - ", 1)
            fields = fields.split()
            target = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
            if Path(target) == mount and filesystem.split()[0] != "autofs":
                mounted.append(fields[2])
        if not mounted:
            return f"Required storage is not mounted at {mount} (UUID {entry['uuid']}); reconnect/mount the drive before starting models"
        device = (Path("/dev/disk/by-uuid") / entry["uuid"]).stat().st_rdev
        identity = f"{os.major(device)}:{os.minor(device)}"
        if identity not in mounted:
            return f"Wrong drive mounted at {mount}; expected UUID {entry['uuid']}"
    except (OSError, ValueError, IndexError) as exc:
        return f"Cannot verify storage at {mount} (UUID {entry['uuid']}): {exc}"
    return None


def storage_error(path, config=STORAGE_CONFIG):
    """Refuse access through a missing or wrong external drive before stat/open.

    Callers may check a not-yet-created download directory. Existing configured
    storage roots still have to exist, and broken symlinks are never writable
    fallback locations.
    """
    entries, error = _storage_paths(config)
    if error:
        return error
    path = Path(os.path.abspath(path))
    for entry in entries:
        root, mount = Path(entry["path"]), Path(entry["mount"])
        if path == root or root in path.parents or path == mount or mount in path.parents:
            if error := _mount_error(entry):
                return error
            if not root.is_dir():
                return f"Required storage directory {root} is missing"
    if path.is_symlink() and not path.exists():
        return f"Storage link {path} points to a missing directory; reconnect its drive"
    return None


def model_start_error(name):
    from model_catalog import MODELS, cache_for_model, moe_profile_error, runtime_error, snapshot

    try:
        path = snapshot(MODELS[name]["repo"]) if name in MODELS else (cache_for_model(name) or HF_CACHE)
        if reason := storage_error(path):
            return reason
        if name not in MODELS and path != HF_CACHE and not path.is_dir():
            return f"Configured model cache {path} is missing"
        if name == "nemotron-3.5-lightning-30b-a3b" and os.environ.get("LOCAL_LLM_MOE_PROFILE_DIR"):
            if reason := moe_profile_error(os.environ["LOCAL_LLM_MOE_PROFILE_DIR"]):
                return reason
        return runtime_error(name, os.environ.get("LOCAL_LLM_IMAGE", "localhost/local-llm:latest"))
    except (OSError, ValueError) as exc:
        return str(exc)


def storage_problems(config_dir=CONFIG_DIR, cache=HF_CACHE, root=Path("/")):
    config_dir, cache, root = Path(config_dir), Path(cache), Path(root)
    config = config_dir / "storage.json"
    entries, error = _storage_paths(config)
    problems = []
    seen_errors = set()
    paths = [root]

    def add_error(message):
        if message not in seen_errors:
            problems.append({"severity": "error", "message": message, "since": None})
            seen_errors.add(message)

    if error:
        add_error(error)
    for entry in entries:
        error = _mount_error(entry)
        if error:
            add_error(error)
        elif not Path(entry["path"]).is_dir():
            add_error(f"Required storage directory {entry['path']} is missing")
        else:
            paths.append(Path(entry["path"]))
    # Check the link itself before following it; a healthy configured archive
    # was already verified above, and an absent automount must not be touched.
    guarded_cache = any(cache == Path(entry["path"]) for entry in entries)
    if not guarded_cache:
        if cache.is_symlink() and not cache.exists():
            add_error(f"Storage link {cache} points to a missing directory; reconnect its drive")
        elif cache.is_dir():
            paths.append(cache)
    directory_file = config_dir / "model-dir"
    try:
        if directory_file.exists():
            value = directory_file.read_text().strip()
            if not value or not Path(value).is_absolute():
                add_error(f"Model storage configuration {directory_file} must contain an absolute directory")
            else:
                directory = Path(value)
                if error := storage_error(directory, config):
                    add_error(error)
                elif not directory.is_dir():
                    add_error(f"Configured model directory {directory} is missing; reconnect its drive before starting models")
                else:
                    paths.append(directory)
    except OSError as exc:
        add_error(f"Cannot read model storage configuration {directory_file}: {exc}")
    devices = set()
    for path in paths:
        try:
            device = path.stat().st_dev
            if device in devices:
                continue
            devices.add(device)
            stats = os.statvfs(path)
            free = stats.f_bavail * stats.f_frsize / GIB
            if free < DISK_LOW_GB:
                severity = "error" if free < DISK_CRITICAL_GB else "warning"
                problems.append({"severity": severity, "message": f"Storage at {path} is almost full: {free:.1f} GiB available; "
                                 "archive files before downloading more models", "since": None})
        except OSError as exc:
            add_error(f"Cannot check free space at {path}: {exc}")
    return problems


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def duration(seconds):
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 90 else f"{minutes / 60:.1f} h"


def outside_scope(pid):
    """The unit holding a container's monitor (conmon, the parent of its first
    process) when that is not the container's own scope in the local-llm slice,
    where `llm` puts it; elsewhere it ends with whatever started the container."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        conmon = re.search(r"^PPid:\s+(\d+)", status, re.M).group(1)
        cgroup = Path(f"/proc/{conmon}/cgroup").read_text().strip()
    except (OSError, AttributeError):
        return None  # Gone or unreadable: nothing to warn about.
    return None if "/local-llm.slice/" in cgroup else cgroup.rsplit("/", 1)[-1]


def exposed_model_ports():
    """Addresses outside this computer on which a model server listens.

    The model servers have no key; only the gateway may face the network.
    With host networking vLLM picked the network card's IPv6 link-local
    address for its engine's internal sockets (found 2026-10-29), which the
    firewall's default zone lets the whole network reach.
    """
    try:
        out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    exposed = []
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 6 or not any(server in fields[5].lower() for server in ("vllm", "llama-server")):
            continue
        host = fields[3].rsplit(":", 1)[0].strip("[]").split("%")[0]
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False  # "*": every address
        if not loopback:
            exposed.append(fields[3])
    return exposed


def foreign_ipv6(name):
    """The IPv6 addresses `name` resolves to that are not this computer's.

    Browsers try IPv6 first. The MyFRITZ name of the router itself resolves to
    the FRITZ!Box's own address, which answers with its login page instead of
    the gateway (2026-11-09); the device name must point here.
    """
    try:
        resolved = {info[4][0] for info in socket.getaddrinfo(name, 443, socket.AF_INET6, socket.SOCK_STREAM)}
        out = subprocess.run(["ip", "-6", "-o", "addr"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []  # No IPv6 for the name, or no answer: IPv4 still gets through.
    own = set(re.findall(r"inet6 ([0-9a-f:]+)/", out))
    return sorted(resolved - own)


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
        self._several_queues = {}  # container id -> started without one GPU queue

    def all(self):
        try:
            out = subprocess.run(
                ["podman", "ps", "-a", "--filter", "name=^llm-", "--format", "json"],
                capture_output=True, text=True, timeout=15).stdout
            listed = json.loads(out or "[]")
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return {}
        return {entry["Names"][0].removeprefix("llm-"): entry for entry in listed}

    def several_gpu_queues(self, container):
        """Whether a model runs without the one GPU queue ./llm starts it with.

        With the runtime's default of four, a loaded model keeps the GPU at
        100% without requests (2026-11-14)."""
        key = container["Id"]
        if key not in self._several_queues:
            try:
                out = subprocess.run(["podman", "inspect", "--format", "{{json .Config.Env}}", key],
                                     capture_output=True, text=True, timeout=15).stdout
                self._several_queues[key] = "GPU_MAX_HW_QUEUES=1" not in json.loads(out)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return False
        return self._several_queues[key]

    def reason(self, container):
        """Why a container ended: TERMINATED when it was stopped from outside,
        else the line that names the error, skipping vLLM's generic wrapper
        that only points further up."""
        key = container["Id"]
        if key not in self._reasons:
            result = subprocess.run(["podman", "logs", "--tail", "300", key],
                                    capture_output=True, text=True)
            log = result.stdout + result.stderr
            if "KeyboardInterrupt: terminated" in log:
                self._reasons[key] = TERMINATED
                return TERMINATED
            if LLAMA_CPP_START in log:
                self._reasons[key] = self.llama_cpp_reason(log, container["ExitCode"])
                return self._reasons[key]
            errors = [line for line in log.splitlines()
                      if ERROR_LINE.search(line) and "Engine core initialization failed" not in line]
            if errors:
                line = errors[-1]
                self._reasons[key] = line[ERROR_LINE.search(line).start():].split(". ")[0][:160]
            elif container["ExitCode"] == 137:
                # Killed without a word in the log: by the kernel for want of RAM,
                # or by a stop that timed out.
                self._reasons[key] = KILLED
            else:
                self._reasons[key] = f"exit code {container['ExitCode']}, see ./llm logs"
        return self._reasons[key]

    @staticmethod
    def llama_cpp_reason(log, exit_code):
        """While loading, the first error is the cause. While serving, llama.cpp
        also logs failed requests as errors, and goes on logging after them;
        a crash is what ends the log, after its last ordinary line. A silent
        kill outranks it."""
        loading, _, serving = log.rsplit(LLAMA_CPP_START, 1)[-1].partition(LLAMA_CPP_LOADED)
        if first := LLAMA_CPP_ERROR.search(loading):
            return first.group(1)[:160]
        if exit_code == 137:
            return KILLED
        ordinary = list(LLAMA_CPP_INFO.finditer(serving))
        serving = serving[ordinary[-1].end():] if ordinary else serving
        if abort := GGML_ABORT.findall(serving):
            place, message = abort[-1]
            return f"{message} ({place})"[:160]
        if crash := LLAMA_CPP_ERROR.findall(serving):
            return crash[-1][:160]
        return f"exit code {exit_code}, see ./llm logs"


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
        self.answering_since = {}  # model -> when it last went from idle to answering
        self.name_checked = 0
        self.name_problem = None
        self.storage_checked = None
        self.storage_problems = []
        self.model_start_errors = {}

    def check(self, now, models_up, requests, gpu, temperature_c, answering=None):
        """Model states and problems for one sample.

        models_up maps each model to whether it answered; answering to how many
        requests it holds; gpu holds busy percent and the VRAM/GTT sizes in bytes.
        """
        problems = []

        def problem(severity, message, since=None):
            problems.append({"severity": severity, "message": message, "since": since and iso(since)})

        if self.storage_checked is None or now - self.storage_checked >= STORAGE_CHECK_EVERY:
            self.storage_problems = storage_problems()
            self.model_start_errors = {name: model_start_error(name) for name in models_up}
            self.storage_checked = now
        problems.extend(self.storage_problems)

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
            if state["state"] == "off" and self.model_start_errors.get(name):
                state["reason"] = self.model_start_errors[name]
            if container and container.get("Restarts"):
                problem("warning", f"{name} crashed and was restarted {container['Restarts']}×")
            states[name] = state
        self.states = states

        # `podman system reset`, even one meant for other storage, deletes these
        # files under running containers (2026-09-28T14:14). They keep serving,
        # but podman no longer learns how they end, so a crash is not restarted.
        lost = [name for name, container in containers.items()
                if container["State"] == "running" and PODMAN_PERSIST.is_dir()
                and not (PODMAN_PERSIST / container["Id"]).exists()]
        if lost:
            problem("warning", f"podman lost track of {', '.join(sorted(lost))} (its runtime files were "
                    "deleted), so a crash there would not be restarted; restart with ./llm up or ./llm gateway")

        # Seen 2026-10-29: models started from the panel lived in GNOME Shell's
        # cgroup, the gateway in that of the app whose terminal started it.
        for name, container in sorted(containers.items()):
            unit = container["State"] == "running" and outside_scope(container["Pid"])
            if unit:
                problem("warning", f"{name} stops when {unit} ends (it started there, not in a scope of "
                        "its own); stop and start it once with ./llm or the website")

        exposed = exposed_model_ports()
        if exposed:
            problem("error", f"A model listens on the network without a key ({len(exposed)} port(s), e.g. "
                    f"{exposed[0]}); restart it with ./llm or the website so it binds to localhost only")

        # A request whose client is long gone can keep a model going for hours
        # (2026-10-29, before the gateway stopped those); from outside it only
        # shows as a model that never gets a moment idle.
        for name, count in (answering or {}).items():
            if not count:
                self.answering_since.pop(name, None)
                continue
            since = self.answering_since.setdefault(name, now)
            if now - since > NO_BREAK:
                problem("warning", f"{name} has been answering without a break for {duration(now - since)}; "
                        "a request may be stuck in a loop (./llm gateway ends those whose client is gone)", since)

        if now - self.name_checked > NAME_CHECK_EVERY:
            self.name_checked = now
            self.name_problem = None
            name = WEB_NAME.read_text().strip() if WEB_NAME.exists() else ""
            foreign = name and foreign_ipv6(name)
            if foreign:
                self.name_problem = (f"{name} points over IPv6 at {foreign[0]}, not at this computer, so "
                                     "browsers may get another machine (at home: the FRITZ!Box login); "
                                     "see \"From the internet\" in the README")
        if self.name_problem:
            problem("warning", self.name_problem)

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
                spinning = [name for name, container in containers.items()
                            if name in models_up and container["State"] == "running"
                            and self.containers.several_gpu_queues(container)]
                advice = (f"{', '.join(spinning)} was started without GPU_MAX_HW_QUEUES=1, which can keep an "
                          "idle model spinning; ./llm stop and ./llm up restart it with one queue"
                          if spinning else "inspect GPU clients and model logs before restarting anything")
                problem("warning", f"The GPU has been busy for {duration(now - self.busy_since)} "
                        f"with no model requests; {advice}", self.busy_since)
        else:
            self.busy_since = None

        memory = meminfo()
        if memory["MemAvailable"] < RAM_LOW_GB * GIB:
            problem("error", f"RAM is almost full: {memory['MemAvailable'] / GIB:.1f} GB free")
        if memory["SwapTotal"] and memory["SwapFree"] < memory["SwapTotal"] / 2:
            problem("warning", f"{(memory['SwapTotal'] - memory['SwapFree']) / GIB:.1f} GB of swap in use; "
                    f"{memory['MemAvailable'] / GIB:.1f} GB RAM available")
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
