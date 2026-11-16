"""Run a component experiment serially and audit all serving request counters."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.added_models import Guard, HERE, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="An idle, reachable backend used for the request audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-idle-gpu-busy", type=float, default=20)
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        parser.error("A component command is required")
    config = json.loads(subprocess.check_output([str(HERE / "llm"), "config"], text=True))
    backends = {m["name"]: f"http://127.0.0.1:{m['port']}/v1" for m in config["models"]}
    lock_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "local-llm-benchmark.lock"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard = Guard(args.model, backends, args.max_idle_gpu_busy)
        started = timestamp()

        def run():
            with args.output.open("w") as log:
                return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode

        idle_rejections = []
        for _ in range(64):
            code, audit = guard.trial(run, 0)
            if code is not None or 'after' in audit:
                break
            idle_rejections.append(audit)
            time.sleep(0.5)
        report = {"started": started, "finished": timestamp(), "command": command,
                  "returncode": code, "max_idle_gpu_busy": args.max_idle_gpu_busy,
                  "preflight_rejections": idle_rejections,
                  "audit": audit, "accepted": code == 0 and not audit["reasons"]}
        args.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["accepted"]:
            raise SystemExit(f"Experiment rejected; inspect {args.output.with_suffix('.audit.json')}")


if __name__ == "__main__":
    main()
