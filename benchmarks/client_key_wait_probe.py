"""Reproduce whisper-local's modifier wait without audio, HTTP or typing."""

import argparse
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace


def run_probe(client_project, held_seconds=60):
    sys.path.insert(0, str(Path(client_project) / "src"))
    from whisper_local.app import Dictation

    class Recorder:
        def start(self):
            pass

        def stop(self):
            return b"synthetic recording"

    clock = SimpleNamespace(now=0.0)
    first_started, allow_finish, first_finished, all_pasted = (threading.Event() for _ in range(4))
    model_calls, pasted = [], []

    def transcribe(_wav, _captured):
        index = len(model_calls)
        model_calls.append({"job": index, "started_s": clock.now})
        if index == 0:
            first_started.set()
            if not allow_finish.wait(2):
                raise RuntimeError("Probe did not release its fake model")
            model_calls[index]["finished_s"] = clock.now
            first_finished.set()
        else:
            model_calls[index]["finished_s"] = clock.now
        return f"transcript {index}"

    def insert(text):
        pasted.append({"text": text, "at_s": clock.now})
        if len(pasted) == 2:
            all_pasted.set()

    app = Dictation(Recorder(), transcribe, insert, clock=lambda: clock.now)
    try:
        app.pressed()
        clock.now = 1
        app.released()
        if not first_started.wait(2):
            raise RuntimeError("Probe worker did not start")
        clock.now = 2
        app.pressed()
        allow_finish.set()
        if not first_finished.wait(2):
            raise RuntimeError("Fake model did not finish")
        before_release = {"model_finished": True, "inserted_count": len(pasted),
                          "model_call_count": len(model_calls)}
        if pasted:
            raise AssertionError("Transcript was inserted with the trigger key held")
        clock.now = 2 + held_seconds
        app.released()
        if not all_pasted.wait(2):
            raise RuntimeError("Probe transcripts did not finish after release")
        return {"synthetic_clock": True, "real_model_calls": 0, "real_keyboard_actions": 0,
                "held_seconds": held_seconds, "while_key_held": before_release,
                "model_calls": model_calls, "insertions": pasted,
                "first_result_wait_after_model_s": pasted[0]["at_s"] - model_calls[0]["finished_s"]}
    finally:
        allow_finish.set()
        app.released()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-project", type=Path, default=Path.home() / "Projects/whisper-local")
    args = parser.parse_args()
    print(json.dumps(run_probe(args.client_project), indent=2))
