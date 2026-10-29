"""Which models the control service stops to make room (run with test_gateway.py)."""

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
