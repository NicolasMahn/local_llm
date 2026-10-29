#!/usr/bin/env python3
"""Starts, stops and restarts models for the gateway, and resumes them at boot.

The gateway runs in a container and cannot start containers itself; handing
it podman would let anyone with a key run anything here. So this small service
runs on the host, knows only start, stop and restart for the models in `llm`,
and does each through `./llm`, which keeps the memory checks and the
one-start-at-a-time rule in one place. The gateway checks the admin key and
passes requests on over a Unix socket that only this user can open.

    GET  /models                  each model's share, state and what is being done to it
    POST /models/NAME/start       202, or 409 with the models that would have to stop;
                                  ?replace=true stops them first
    POST /models/NAME/stop        202
    POST /models/NAME/restart     202

Starting takes minutes (up to an hour on a first start), so the answer comes
right away and the state shows in GET /models as it goes.

./llm install runs it as a user service from boot on; it then starts the
gateway and the models that ran last (`./llm up`).
"""

import itertools
import json
import os
import re
import signal
import socketserver
import subprocess
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
LLM = str(HERE / "llm")
SOCKET = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "local-llm/control.sock"
STATS = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "local-llm/stats.json"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def config():
    return json.loads(subprocess.run([LLM, "config"], capture_output=True, text=True, check=True).stdout)


def model_states():
    """What monitor.py last saw, or {} when its snapshot is missing or stale."""
    try:
        stats = json.loads(STATS.read_text())
        if time.time() - datetime.fromisoformat(stats["updated"]).timestamp() < 15:
            return stats.get("model_states", {})
    except (OSError, ValueError, KeyError):
        pass
    return {}


def to_stop(name, models, states):
    """The fewest running models (then the least GPU memory) whose stopping lets name fit,
    [] when it fits already, or None when it cannot fit even alone."""
    shares = {model["name"]: model["share"] for model in models["models"]}
    running = [other for other, state in states.items()
               if other != name and other in shares and state["state"] in ("live", "starting")]
    for count in range(len(running) + 1):
        candidates = sorted(itertools.combinations(running, count),
                            key=lambda stopped: sum(shares[other] for other in stopped))
        for stopped in candidates:
            kept = sum(shares[other] for other in running if other not in stopped)
            if kept + shares[name] <= models["budget"] + 1e-9:
                return list(stopped)
    return None


def last_line(output):
    lines = [line for line in output.splitlines() if line.strip() and not line.startswith(("started ", "waiting "))]
    return lines[-1] if lines else "failed without saying why"


class Jobs:
    """What is being done to each model: one job at a time per model.

    A start can take an hour, so stop and restart interrupt one that is still
    going, rather than waiting for it or being refused.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.current = {}  # name -> {"action", "since", "process", "cancelled"}
        self.failed = {}  # name -> {"action", "message", "at"}: the last job that did not work

    def public(self, name):
        job = self.current.get(name)
        return job and {"action": job["action"], "since": job["since"]}

    def start(self, name, action, steps, interrupt=False):
        """Runs steps (each a list of llm arguments) in the background; False if busy."""
        with self.lock:
            running = self.current.get(name)
            if running and not interrupt:
                return False
            job = {"action": action, "since": now(), "process": None, "cancelled": False}
            self.current[name] = job
            self.failed.pop(name, None)
        if running:
            self._cancel(running)
        threading.Thread(target=self._run, args=(name, job, steps), daemon=True).start()
        return True

    def _cancel(self, job):
        with self.lock:
            job["cancelled"] = True
            process = job["process"]
        if process and process.poll() is None:
            # The whole group: llm and the curl or sleep it waits in.
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()

    def _run(self, name, job, steps):
        try:
            for step in steps:
                with self.lock:
                    if job["cancelled"]:
                        return
                    process = job["process"] = subprocess.Popen(
                        [LLM, *step], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                        start_new_session=True)
                stdout, stderr = process.communicate()
                if process.returncode and not job["cancelled"]:
                    with self.lock:
                        self.failed[name] = {"action": job["action"], "message": last_line(stderr or stdout),
                                             "at": now()}
                    return
        finally:
            with self.lock:
                if self.current.get(name) is job:
                    del self.current[name]


jobs = Jobs()


class Handler(BaseHTTPRequestHandler):
    server_version = "local-llm-control"

    def answer(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def refuse(self, status, message, **extra):
        self.answer(status, {"error": {"message": message, **extra}})

    def do_GET(self):
        if urlsplit(self.path).path != "/models":
            return self.refuse(HTTPStatus.NOT_FOUND, "Not found")
        models, states = config(), model_states()
        with jobs.lock:
            current = {name: jobs.public(name) for name in jobs.current}
            failed = dict(jobs.failed)
        self.answer(HTTPStatus.OK, {
            "budget": models["budget"],
            "monitor": bool(states),
            "models": [{
                **model,
                **states.get(model["name"], {"state": "unknown"}),
                "job": current.get(model["name"]),
                "failed_job": failed.get(model["name"]),
            } for model in models["models"]],
        })

    def do_POST(self):
        url = urlsplit(self.path)
        match = re.fullmatch(r"/models/([\w.-]+)/(start|stop|restart)", url.path)
        if not match:
            return self.refuse(HTTPStatus.NOT_FOUND, "Not found")
        name, action = match.groups()
        models = config()
        if name not in {model["name"] for model in models["models"]}:
            return self.refuse(HTTPStatus.NOT_FOUND, f"Unknown model {name!r}")

        stopping = []
        if action == "stop":
            steps = [["stop", name]]
        else:
            states = model_states()
            if not states:
                return self.refuse(HTTPStatus.SERVICE_UNAVAILABLE,
                                   "The monitor is not running, so it is unknown what runs; see ./llm install")
            stopping = to_stop(name, models, states)
            if stopping is None:
                return self.refuse(HTTPStatus.CONFLICT, f"{name} does not fit in the GPU memory budget even alone")
            replace = parse_qs(url.query).get("replace", [""])[0].lower() in ("1", "true", "yes")
            if stopping and not replace:
                return self.refuse(HTTPStatus.CONFLICT,
                                   f"{name} fits only after stopping {', '.join(stopping)}; ask again with ?replace=true",
                                   would_stop=stopping)
            steps = [["stop", other] for other in stopping]
            if action == "restart":
                steps.append(["stop", name])
            steps.append(["up", name])
        if not jobs.start(name, action, steps, interrupt=action in ("stop", "restart")):
            return self.refuse(HTTPStatus.CONFLICT, f"{name} is busy: {(jobs.public(name) or {}).get('action')}")
        self.answer(HTTPStatus.ACCEPTED, {"model": name, "action": action, "stopping": stopping})

    def log_message(self, format, *args):
        pass  # The journal gets errors only; requests are the gateway's to log.


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def get_request(self):
        # http.server expects a (host, port) peer; a Unix socket has none.
        request, _ = super().get_request()
        return request, ("local", 0)


def main():
    SOCKET.parent.mkdir(parents=True, exist_ok=True)
    SOCKET.unlink(missing_ok=True)
    os.umask(0o077)  # only this user (and its containers) may open the socket
    server = Server(str(SOCKET), Handler)
    # At boot: the gateway, and the models that ran when the computer went down.
    threading.Thread(target=subprocess.run, args=([LLM, "up"],), daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
