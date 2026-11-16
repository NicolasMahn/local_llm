"""Resumable model downloads with pinned revisions and disk-space admission."""

import argparse
import fcntl
import json
import os
import shutil
from fnmatch import fnmatch
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from health import storage_error
from model_catalog import GIB, MODELS, cache_error, cache_root, model_dir, snapshot

HERE = Path(__file__).resolve().parent
RESERVE_BYTES = 25 * GIB


def configured_models():
    result = subprocess.run([str(HERE / "llm"), "config"], check=True,
                            capture_output=True, text=True)
    return json.loads(result.stdout)["models"]


def select_models(models, names):
    by_name = {model["name"]: model for model in models}
    unknown = set(names) - by_name.keys()
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(sorted(unknown))}; ./llm list shows them")
    return [by_name[name] for name in dict.fromkeys(names)] if names else models


def remaining_bytes(files, target):
    """Finished files cost no extra space; partial transfers stay conservative."""
    missing = 0
    for filename, size in files:
        local = target / filename
        try:
            finished = local.is_file() and local.stat().st_size == size
        except OSError:
            finished = False
        if not finished:
            missing += size
    return missing


def require_space(target, missing, *, reserve=RESERVE_BYTES):
    reason = storage_error(target)
    if reason:
        raise ValueError(reason)
    parent = target
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    available = shutil.disk_usage(parent).free
    if available < missing + reserve:
        raise ValueError(
            f"not enough space for {target}: {missing / GIB:.1f} GiB still to download, "
            f"{reserve / GIB:.0f} GiB reserve required, {available / GIB:.1f} GiB available")


def download(model, *, check=False):
    # Imported only here so selection and admission tests need no Hub/network.
    from huggingface_hub import HfApi, snapshot_download

    name, repo = model["name"], model["repo"]
    pinned = MODELS.get(name)
    root = model_dir() if pinned else None
    cache = None
    if root is None:
        if reason := cache_error(repo):
            raise ValueError(reason)
        cache = cache_root(repo) / "hub"
    guard_path = root if root is not None else cache
    reason = storage_error(guard_path)
    if reason:
        raise ValueError(reason)
    # These public additions must not inherit an expired OAuth login from some
    # other project. Existing gated checkpoints still use the user's login.
    authentication = {"token": False} if pinned else {}
    info = HfApi().model_info(repo, revision=pinned["revision"] if pinned else None,
                             files_metadata=True, **authentication)
    if pinned and info.sha != pinned["revision"]:
        raise ValueError(f"{name}: the Hub resolved a different revision than requested")
    target = snapshot(repo) if pinned else cache / ("models--" + repo.replace("/", "--")) / "snapshots" / info.sha
    ignore = pinned.get("ignore", []) if pinned else []
    files = [(item.rfilename, item.size) for item in info.siblings
             if not any(fnmatch(item.rfilename, pattern) for pattern in ignore)]
    if any(size is None or size < 0 or Path(filename).is_absolute()
           or ".." in Path(filename).parts for filename, size in files):
        raise ValueError(f"{name}: missing file sizes or unsafe paths in Hub metadata")
    reason = storage_error(target)
    if reason:
        raise ValueError(reason)
    missing = remaining_bytes(files, target)
    require_space(target, missing)
    print(f"{'CHECK' if check else 'START'} {name} {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
          f"revision={info.sha} remaining_gib={missing / GIB:.2f} target={target}", flush=True)
    if check:
        return
    options = {"local_dir": target} if root is not None else {"cache_dir": cache}
    snapshot_download(repo, revision=info.sha, ignore_patterns=ignore or None, **options, **authentication)
    if remaining_bytes(files, target):
        raise ValueError(f"{name}: download ended with missing or incomplete files")
    print(f"DONE {name} {datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)


def main():
    from huggingface_hub.errors import HfHubHTTPError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Names from ./llm list; omit for every configured model")
    parser.add_argument("--check", action="store_true", help="Check sizes, revisions and space without downloading")
    args = parser.parse_args()
    models = select_models(configured_models(), args.models)
    state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "local-llm"
    state.mkdir(parents=True, exist_ok=True)
    failures = []
    with (state / "download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for model in models:
            try:
                download(model, check=args.check)
            except HfHubHTTPError as exc:
                failures.append(model["name"])
                status = exc.response.status_code if exc.response is not None else "unknown"
                reason = ("authentication refused; a gated checkpoint needs a valid local Hugging Face login"
                          if status == 401 else "request refused")
                # Do not dump request/response objects or cached credentials.
                print(f"FAIL {model['name']}: Hub HTTP {status}: {reason}", flush=True)
            except (OSError, ValueError) as exc:
                failures.append(model["name"])
                print(f"FAIL {model['name']}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"Download failed for {', '.join(failures)}")


if __name__ == "__main__":
    main()
