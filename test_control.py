"""Which models the control service stops to make room (run with test_gateway.py)."""

from http import HTTPStatus
from unittest.mock import Mock

import pytest

import control
from control import to_stop

MODELS = {"budget": 0.95, "models": [
    {"name": "qwen", "share": 0.30}, {"name": "gemma-26b", "share": 0.25},
    {"name": "gemma-12b", "share": 0.18}, {"name": "gemma-31b", "share": 0.32},
]}


def running(*names):
    return {name: {"state": "live"} for name in names}


def test_a_model_that_fits_stops_nothing():
    assert to_stop("gemma-12b", MODELS, running("qwen", "gemma-26b")) == []


def test_the_fewest_and_then_smallest_models_make_room():
    # 0.73 runs; 0.32 more needs 0.10 freed: one model will do, the smallest is enough.
    assert to_stop("gemma-31b", MODELS, running("qwen", "gemma-26b", "gemma-12b")) == ["gemma-12b"]


def test_crashed_and_stopped_models_take_no_room():
    states = running("qwen", "gemma-26b") | {"gemma-12b": {"state": "failed"}}
    assert to_stop("gemma-31b", MODELS, states) == []


def test_a_model_larger_than_the_budget_cannot_start():
    assert to_stop("huge", {"budget": 0.95, "models": [{"name": "huge", "share": 1.2}]}, {}) is None


def test_kernel_startup_policy_retains_only_the_measured_serving_win():
    import json
    import os
    from pathlib import Path
    import subprocess

    script = Path(__file__).with_name("llm")
    environment = os.environ.copy()
    environment.pop("LOCAL_LLM_FAST_KERNELS", None)

    def modes():
        result = subprocess.run([str(script), "config"], env=environment,
                                check=True, capture_output=True, text=True)
        return {model["name"]: model["fast_kernels"] for model in json.loads(result.stdout)["models"]}

    defaults = modes()
    assert defaults["qwen3-omni-30b"] == 1
    assert defaults["gemma-4-26b"] == 0
    assert defaults["nemotron-3.5-lightning-30b-a3b"] == 1
    for override in ("0", "1"):
        environment["LOCAL_LLM_FAST_KERNELS"] = override
        assert set(modes().values()) == {int(override)}
    environment["LOCAL_LLM_FAST_KERNELS"] = "enabled"
    assert set(modes().values()) == {1}


def request_model_action(monkeypatch, *, reason=None, downloaded=True, action="start"):
    models = {"budget": 0.95, "models": [
        {"name": "granite", "share": 0.32, "downloaded": downloaded},
        *MODELS["models"][:3],
    ]}
    states = running("qwen", "gemma-26b", "gemma-12b")
    states["granite"] = {"state": "live" if action == "restart" else "off"}
    monkeypatch.setattr(control, "config", lambda: models)
    monkeypatch.setattr(control, "model_states", lambda: states)
    monkeypatch.setattr(control, "model_start_error", lambda name: reason)
    jobs = Mock()
    jobs.start.return_value = True
    monkeypatch.setattr(control, "jobs", jobs)
    handler = object.__new__(control.Handler)
    handler.path = f"/models/granite/{action}?replace=true"
    handler.answer = Mock()
    handler.refuse = Mock()
    handler.do_POST()
    return handler, jobs


@pytest.mark.parametrize("action", ["start", "restart"])
@pytest.mark.parametrize("reason", ["Required storage is not mounted", "Required runtime feature is missing"])
def test_hard_blockers_cannot_stop_a_running_model_or_its_neighbours(monkeypatch, action, reason):
    handler, jobs = request_model_action(monkeypatch, reason=reason, action=action)
    jobs.start.assert_not_called()
    handler.refuse.assert_called_once_with(HTTPStatus.CONFLICT, f"granite cannot start: {reason}")


def test_incomplete_download_is_refused_before_replacement_stops(monkeypatch):
    handler, jobs = request_model_action(monkeypatch, downloaded=False)
    jobs.start.assert_not_called()
    assert "not (fully) downloaded" in handler.refuse.call_args.args[1]


def test_preflight_allows_memory_to_be_freed_by_the_requested_replacement(monkeypatch):
    handler, jobs = request_model_action(monkeypatch)
    jobs.start.assert_called_once_with("granite", "start", [["stop", "gemma-12b"], ["up", "granite"]], interrupt=False)
    handler.answer.assert_called_once_with(HTTPStatus.ACCEPTED, {
        "model": "granite", "action": "start", "stopping": ["gemma-12b"]})


def test_stopping_does_not_depend_on_checkpoint_availability(monkeypatch):
    handler, jobs = request_model_action(monkeypatch, reason="archive missing", downloaded=False, action="stop")
    jobs.start.assert_called_once_with("granite", "stop", [["stop", "granite"]], interrupt=True)
    handler.refuse.assert_not_called()


def test_cli_status_shows_actionable_off_reason_without_warning_for_unused_models(tmp_path):
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    import subprocess
    import sys

    source = Path(__file__).with_name("llm").read_text()
    marker = 'python3 - "$STATE_DIR/stats.json" "$(config)" <<\'PY\'\n'
    program = source.split(marker, 1)[1].split("\nPY", 1)[0]
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(), "problems": [],
        "model_states": {"blocked": {"state": "off", "reason": "Required runtime fix is missing"},
                         "unused": {"state": "off"}, "running": {"state": "live"}},
    }))
    config = {"models": [{"name": name} for name in ("blocked", "unused", "running")]}
    result = subprocess.run([sys.executable, "-c", program, str(path), json.dumps(config)],
                            check=True, text=True, capture_output=True)
    assert result.stdout.splitlines() == ["blocked: off: Required runtime fix is missing", "unused: off", "running: live"]
