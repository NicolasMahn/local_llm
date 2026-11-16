import json
import os

from model_usage import ModelUsage


def counters(requests=10, prompt=100, output=50):
    return {"completed_requests": requests, "input_tokens": prompt, "output_tokens": output}


def test_initial_cumulative_counters_are_baselined_not_retroactive_use(tmp_path):
    usage = ModelUsage(tmp_path / "usage.json", "boot-a")
    usage.observe("model", counters(), 100)
    assert usage.summary()["models"]["model"] == {
        "completed_requests": 0, "input_tokens": 0, "output_tokens": 0,
        "sampled_active_seconds": 0, "last_used": None,
    }
    usage.observe("model", counters(11, 130, 55), 102, active=1)
    assert usage.summary()["models"]["model"]["completed_requests"] == 1
    assert usage.summary()["models"]["model"]["input_tokens"] == 30
    assert usage.summary()["models"]["model"]["output_tokens"] == 5
    assert usage.summary()["models"]["model"]["sampled_active_seconds"] == 2


def test_benchmark_requests_are_not_counted_and_use_after_them_is(tmp_path):
    usage = ModelUsage(tmp_path / "usage.json", "boot-a")
    usage.observe("model", counters(), 100)
    usage.observe("model", counters(40, 9000, 700), 102, active=4, count=False)
    usage.observe("model", counters(60, 12000, 900), 104, count=False)
    assert usage.summary()["models"].get("model", {}).get("completed_requests", 0) == 0
    assert usage.summary()["models"].get("model", {}).get("last_used") is None
    usage.observe("model", counters(61, 12030, 905), 106, active=1)
    entry = usage.summary()["models"]["model"]
    assert (entry["completed_requests"], entry["input_tokens"], entry["output_tokens"]) == (1, 30, 5)
    # The benchmark's last sample was not use, so the gap after it adds only its own time.
    assert entry["sampled_active_seconds"] == 2


def test_benchmark_marker_names_the_model_only_while_its_process_runs(tmp_path):
    from model_usage import benchmark, benchmarked_model

    marker = tmp_path / "benchmark.json"
    assert benchmarked_model(marker) is None
    with benchmark("granite-4.2-30b", marker):
        assert benchmarked_model(marker) == "granite-4.2-30b"
    assert benchmarked_model(marker) is None
    # A killed suite leaves its marker behind; a dead process must not hide real use.
    marker.write_text('{"model": "granite-4.2-30b", "pid": 2147483646}')
    assert benchmarked_model(marker) is None
    marker.write_text("not json")
    assert benchmarked_model(marker) is None


def test_monitor_restart_keeps_totals_and_continues_same_backend_baseline(tmp_path):
    path = tmp_path / "usage.json"
    usage = ModelUsage(path, "boot-a")
    usage.observe("model", counters(), 100, server_id=10)
    usage.observe("model", counters(11, 120, 60), 102, active=1, server_id=10)
    usage.save()
    usage = ModelUsage(path, "boot-a")
    usage.observe("model", counters(12, 125, 62), 104, server_id=10)
    result = usage.summary()["models"]["model"]
    assert result["completed_requests"] == 2
    assert result["input_tokens"] == 25
    assert result["output_tokens"] == 12
    assert result["sampled_active_seconds"] == 4
    assert result["last_used"] == "1970-01-01T00:01:44+00:00"


def test_backend_counter_reset_and_identity_change_never_erase_history(tmp_path):
    usage = ModelUsage(tmp_path / "usage.json", "boot")
    usage.observe("model", counters(), 100, server_id=10)
    usage.observe("model", counters(11, 120, 60), 102, server_id=10)
    usage.observe("model", counters(1, 10, 5), 104, server_id=10)
    usage.observe("model", counters(2, 20, 10), 106, server_id=10)
    usage.observe("model", counters(100, 1000, 500), 108, server_id=108)
    usage.observe("model", counters(101, 1010, 505), 110, server_id=108)
    assert usage.summary()["models"]["model"]["completed_requests"] == 3
    assert usage.summary()["models"]["model"]["input_tokens"] == 40


def test_reboot_and_offline_gaps_start_fresh_baselines(tmp_path):
    path = tmp_path / "usage.json"
    usage = ModelUsage(path, "boot-a")
    usage.observe("model", counters(), 100)
    usage.observe("model", counters(11), 102)
    usage.save()
    usage = ModelUsage(path, "boot-b")
    usage.observe("model", counters(200), 104)
    usage.observe("model", None, 106)
    usage.observe("model", counters(300), 108)
    assert usage.summary()["models"]["model"]["completed_requests"] == 1


def test_mid_reset_metrics_never_subtract_previous_totals(tmp_path):
    usage = ModelUsage(tmp_path / "usage.json", "boot")
    usage.observe("model", counters(), 100)
    usage.observe("model", counters(11, 120, 60), 102)
    # Multiprocess metrics can expose a mixture of old and new counters while
    # workers restart. Every drop resets the whole baseline, preserving totals.
    usage.observe("model", counters(1, 140, 70), 104)
    usage.observe("model", counters(2, 10, 5), 106)
    usage.observe("model", counters(3, 20, 10), 108)
    totals = usage.summary()["models"]["model"]
    assert totals["completed_requests"] == 2
    assert totals["input_tokens"] == 30
    assert totals["output_tokens"] == 15


def test_unavailable_metrics_and_long_gaps_cannot_invent_usage_time(tmp_path):
    usage = ModelUsage(tmp_path / "usage.json", "boot")
    usage.observe("model", counters(), 100, active=1)
    usage.observe("model", counters(11), 200, active=1)
    assert usage.summary()["models"]["model"]["sampled_active_seconds"] == 0
    usage.observe("model", {**counters(), "input_tokens": None}, 202)
    usage.observe("model", counters(15), 204)
    assert usage.summary()["models"]["model"]["completed_requests"] == 1


def test_daily_totals_use_utc_observation_date_and_contain_only_counts(tmp_path):
    path = tmp_path / "usage.json"
    usage = ModelUsage(path, "boot")
    usage.observe("model", counters(), 86399)
    usage.observe("model", counters(11, 120, 60), 86401)
    usage.save()
    report = usage.summary()
    assert [row["day"] for row in report["days"]] == ["1970-01-01", "1970-01-02"]
    assert report["days"][1]["models"]["model"]["completed_requests"] == 1
    assert set(json.loads(path.read_text())["last"]["model"]) == {"counters", "at", "active", "server_id"}


def test_unreadable_history_is_reported_and_not_overwritten(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("broken")
    usage = ModelUsage(path, "boot")
    usage.observe("model", counters(), 100)
    usage.save()
    assert usage.summary()["error"]
    assert path.read_text() == "broken"


def test_temporary_persistence_failure_is_visible_and_recovers(tmp_path, monkeypatch):
    usage = ModelUsage(tmp_path / "usage.json", "boot")
    usage.observe("model", counters(), 100)
    replace = os.replace

    def read_only(*args):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("model_usage.os.replace", read_only)
    usage.save()
    assert "Cannot save" in usage.summary()["error"]
    monkeypatch.setattr("model_usage.os.replace", replace)
    usage.save()
    assert usage.summary()["error"] is None
    assert usage.path.is_file()


def test_valid_json_with_invalid_ledger_structure_is_preserved(tmp_path):
    path = tmp_path / "usage.json"
    for contents in ["[]", json.dumps({"version": 1, "days": {"day": {"model": {}}},
                                      "last_used": {}, "last": {}})]:
        path.write_text(contents)
        usage = ModelUsage(path, "boot")
        assert usage.summary()["error"]
        usage.save()
        assert path.read_text() == contents
