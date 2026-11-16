import json
import os
from pathlib import Path
from types import SimpleNamespace

import health


def config(tmp_path, path, mount, uuid="drive-uuid"):
    location = tmp_path / "storage.json"
    location.write_text(json.dumps({"paths": [{"path": str(path), "mount": str(mount), "uuid": uuid}]}))
    return location


def fake_mount(monkeypatch, mount, *, device="8:18", filesystem="ext4", uuid_device=None):
    read_text, stat = Path.read_text, Path.stat

    def text(path, *args, **kwargs):
        if path == Path("/proc/self/mountinfo"):
            return f"50 20 {device} / {mount} rw - {filesystem} /dev/sdb2 rw\n"
        return read_text(path, *args, **kwargs)

    def metadata(path, *args, **kwargs):
        if path == Path("/dev/disk/by-uuid/drive-uuid"):
            return SimpleNamespace(st_rdev=os.makedev(*(uuid_device or (8, 18))))
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(Path, "stat", metadata)


def test_absent_drive_is_rejected_before_following_its_path(tmp_path, monkeypatch):
    mount = tmp_path / "archive"
    root = tmp_path / "huggingface"
    root.symlink_to(mount / "cache")
    location = config(tmp_path, root, mount)
    fake_mount(monkeypatch, mount, filesystem="autofs")
    original = Path.is_dir

    def no_follow(path):
        assert path != root, "must not trigger the absent automount"
        return original(path)

    monkeypatch.setattr(Path, "is_dir", no_follow)
    assert "not mounted" in health.storage_error(root / "hub/model", location)


def test_wrong_drive_is_refused_and_correct_drive_allows_future_download_leaf(tmp_path, monkeypatch):
    mount = tmp_path / "archive"
    mount.mkdir()
    root = mount / "cache"
    root.mkdir()
    location = config(tmp_path, root, mount)
    fake_mount(monkeypatch, mount, uuid_device=(8, 19))
    assert "Wrong drive" in health.storage_error(root / "future-model", location)
    fake_mount(monkeypatch, mount)
    assert health.storage_error(root / "future-model", location) is None


def test_broken_cache_link_and_missing_model_directory_are_reported(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    cache = tmp_path / "cache"
    cache.symlink_to(tmp_path / "absent")
    (tmp_path / "model-dir").write_text(str(tmp_path / "models"))
    messages = [row["message"] for row in health.storage_problems(tmp_path, cache, root)]
    assert any("Storage link" in message for message in messages)
    assert any("Configured model directory" in message for message in messages)


def test_low_disk_space_is_visible_before_downloads_fill_the_filesystem(tmp_path, monkeypatch):
    for free_gib, severity in [(20, "warning"), (3, "error")]:
        monkeypatch.setattr(health.os, "statvfs", lambda path: SimpleNamespace(f_bavail=free_gib, f_frsize=health.GIB))
        problems = health.storage_problems(tmp_path, tmp_path / "missing", tmp_path)
        assert len(problems) == 1
        assert problems[0]["severity"] == severity
        assert "almost full" in problems[0]["message"]


def test_invalid_storage_config_fails_closed(tmp_path):
    (tmp_path / "storage.json").write_text('{"paths": [{"path": "relative"}]}')
    assert "Cannot read storage configuration" in health.storage_error(tmp_path, tmp_path / "storage.json")


def test_off_models_keep_their_storage_reason_but_live_models_keep_serving(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "KernelLog", lambda: SimpleNamespace(hangs=[]))
    monkeypatch.setattr(health.Containers, "all", lambda self: {})
    monkeypatch.setattr(health, "storage_problems", lambda: [])
    monkeypatch.setattr(health, "model_start_error", lambda name: "Required archive is not mounted")
    monkeypatch.setattr(health, "exposed_model_ports", lambda: [])
    monkeypatch.setattr(health, "WEB_NAME", tmp_path / "no-web-name")
    monkeypatch.setattr(health, "meminfo", lambda: {
        "MemAvailable": 20 * health.GIB, "MemTotal": 124 * health.GIB,
        "SwapTotal": 0, "SwapFree": 0})
    states, _ = health.Health(tmp_path).check(
        1000, {"off-model": False, "live-model": True}, 0, {"busy": 0, "gtt_total": 110 * health.GIB}, 30)
    assert states["off-model"] == {"state": "off", "reason": "Required archive is not mounted"}
    assert states["live-model"] == {"state": "live"}


def test_hot_cache_mount_is_checked_before_following_an_absent_automount(tmp_path, monkeypatch):
    import model_catalog

    hot_cache = tmp_path / "absent-automount/hot-cache"
    monkeypatch.setattr(model_catalog, "cache_for_model", lambda name: hot_cache)
    checked = []

    def unavailable(path):
        checked.append(path)
        return "Required storage is not mounted"

    def must_not_follow(path):
        raise AssertionError("must not follow a target on an absent automount")

    monkeypatch.setattr(health, "storage_error", unavailable)
    monkeypatch.setattr(Path, "is_dir", must_not_follow)
    assert health.model_start_error("qwen3-omni-30b") == "Required storage is not mounted"
    assert checked == [hot_cache]


def test_hot_cache_preflight_uses_the_same_runtime_image_as_llm(tmp_path, monkeypatch):
    import model_catalog

    monkeypatch.setattr(model_catalog, "cache_for_model", lambda name: tmp_path)
    monkeypatch.setattr(health, "storage_error", lambda path: None)
    observed = []
    monkeypatch.setattr(model_catalog, "runtime_error", lambda name, image: observed.append((name, image)))
    monkeypatch.setenv("LOCAL_LLM_IMAGE", "localhost/local-llm:tested")
    assert health.model_start_error("qwen3-omni-30b") is None
    assert observed == [("qwen3-omni-30b", "localhost/local-llm:tested")]


def test_invalid_configured_nemotron_profile_is_visible_in_model_health(tmp_path, monkeypatch):
    import model_catalog

    monkeypatch.setattr(model_catalog, "snapshot", lambda repo: tmp_path / "checkpoint")
    monkeypatch.setattr(health, "storage_error", lambda path: None)
    monkeypatch.setenv("LOCAL_LLM_MOE_PROFILE_DIR", str(tmp_path))
    assert "Invalid Nemotron MoE profile" in health.model_start_error("nemotron-3.5-lightning-30b-a3b")


def test_persistent_idle_gpu_activity_warns_and_clears_after_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "KernelLog", lambda: SimpleNamespace(hangs=[]))
    monkeypatch.setattr(health.Containers, "all", lambda self: {})
    monkeypatch.setattr(health, "storage_problems", lambda: [])
    monkeypatch.setattr(health, "model_start_error", lambda name: None)
    monkeypatch.setattr(health, "exposed_model_ports", lambda: [])
    monkeypatch.setattr(health, "WEB_NAME", tmp_path / "no-web-name")
    monkeypatch.setattr(health, "meminfo", lambda: {
        "MemAvailable": 20 * health.GIB, "MemTotal": 124 * health.GIB,
        "SwapTotal": 0, "SwapFree": 0})
    monitor = health.Health(tmp_path)
    gpu = {"busy": 100, "gtt_total": 110 * health.GIB}
    assert not monitor.check(1000, {"granite": True}, 0, gpu, 30)[1]
    _, problems = monitor.check(1001 + health.BUSY_WITHOUT_WORK, {"granite": True}, 0, gpu, 30)
    assert any("no model requests" in p["message"] for p in problems)
    assert not monitor.check(1002 + health.BUSY_WITHOUT_WORK, {"granite": True}, 0,
                             gpu | {"busy": 0}, 30)[1]


def test_idle_gpu_warning_names_the_model_that_runs_with_several_gpu_queues(tmp_path, monkeypatch):
    running = {name: {"Id": name, "State": "running", "StartedAt": 0, "Pid": 1}
               for name in ("gemma-4-26b", "granite-4.2-30b", "gateway")}
    environments = {"gemma-4-26b": ["LOCAL_LLM_FAST_KERNELS=0"],
                    "granite-4.2-30b": ["GPU_MAX_HW_QUEUES=1"], "gateway": []}
    monkeypatch.setattr(health, "KernelLog", lambda: SimpleNamespace(hangs=[]))
    monkeypatch.setattr(health.Containers, "all", lambda self: running)
    monkeypatch.setattr(health.subprocess, "run", lambda command, **_: SimpleNamespace(
        stdout=json.dumps(environments[command[-1]])))
    monkeypatch.setattr(health, "storage_problems", lambda: [])
    monkeypatch.setattr(health, "model_start_error", lambda name: None)
    monkeypatch.setattr(health, "exposed_model_ports", lambda: [])
    monkeypatch.setattr(health, "outside_scope", lambda pid: None)
    monkeypatch.setattr(health, "PODMAN_PERSIST", tmp_path / "no-persist")
    monkeypatch.setattr(health, "WEB_NAME", tmp_path / "no-web-name")
    monkeypatch.setattr(health, "meminfo", lambda: {
        "MemAvailable": 20 * health.GIB, "MemTotal": 124 * health.GIB, "SwapTotal": 0, "SwapFree": 0})
    monitor = health.Health(tmp_path)
    models = {"gemma-4-26b": True, "granite-4.2-30b": True}
    gpu = {"busy": 100, "gtt_total": 110 * health.GIB}
    monitor.check(1000, models, 0, gpu, 30)
    _, problems = monitor.check(1001 + health.BUSY_WITHOUT_WORK, models, 0, gpu, 30)
    message = next(p["message"] for p in problems if "no model requests" in p["message"])
    assert "gemma-4-26b was started without GPU_MAX_HW_QUEUES=1" in message
    assert "granite" not in message and "gateway" not in message
