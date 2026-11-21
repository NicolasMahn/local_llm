"""Pinned checkpoint selection, disk admission and CLI discovery."""

import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

import model_catalog as catalog


def test_author_recommended_benchmark_sampling_applies_only_to_olmo():
    assert catalog.MODELS["olmo-3.1-32b-think"]["benchmark_sampling"] == {
        "temperature": 0.6, "top_p": 0.95, "seed": 42}
    assert "benchmark_sampling" not in catalog.MODELS["granite-4.2-30b"]
    assert "benchmark_sampling" not in catalog.MODELS["nemotron-3.5-lightning-30b-a3b"]


def test_missing_or_incomplete_moe_profile_cannot_silently_use_native_defaults(tmp_path):
    assert "Invalid Nemotron MoE profile" in catalog.moe_profile_error(tmp_path)
    path = tmp_path / catalog.NEMOTRON_MOE_PROFILE
    path.write_text(json.dumps({"128": {"BLOCK_SIZE_M": 32}}))
    assert "every batch size" in catalog.moe_profile_error(tmp_path)
    profile = {str(m): {"BLOCK_SIZE_M": 16 if m <= 20 else 32 if m <= 40 else 64,
                       "GROUP_SIZE_M": 1, "SPLIT_K": 1} for m in range(1, 513)}
    path.write_text(json.dumps(profile))
    assert catalog.moe_profile_error(tmp_path) is None
    profile["128"]["SPLIT_K"] = 2
    path.write_text(json.dumps(profile))
    assert "batch 128" in catalog.moe_profile_error(tmp_path)


def test_gemma_with_a_quantized_kv_cache_is_refused_before_it_can_outlast_the_gpu_watchdog():
    quantized = ["--max-model-len", "262144", "--kv-cache-dtype", "int8_per_token_head"]
    assert "unquantized KV cache" in catalog.flags_error("gemma-4-31b", quantized)
    # Models whose prefill the attention plugin does not take over may try it.
    assert catalog.flags_error("qwen3-omni-30b", quantized) is None
    assert catalog.flags_error("gemma-4-31b", ["--max-model-len", "262144"]) is None
    # A repeated flag: vLLM uses the last value, and so does the check.
    assert catalog.flags_error("gemma-4-12b", [*quantized, "--kv-cache-dtype", "auto"]) is None
    script = Path(catalog.__file__)
    refused = subprocess.run(["python3", str(script), "check-flags", "gemma-4-12b", "--kv-cache-dtype", "fp8"],
                             capture_output=True, text=True)
    assert refused.returncode == 1 and "gemma-4-12b needs an unquantized KV cache" in refused.stderr


def test_qwen_with_its_draft_head_is_refused_without_the_interval_the_engine_then_requires():
    usual = ["--max-model-len", "262144", "--prefix-cache-retention-interval", "1568"]
    draft = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3}']
    assert catalog.flags_error("qwen3.8-27b", usual) is None
    assert "--prefix-cache-retention-interval 1600" in catalog.flags_error("qwen3.8-27b", usual + draft)
    assert catalog.flags_error("qwen3.8-27b", usual + draft + ["--prefix-cache-retention-interval", "1600"]) is None
    # Other models have their own block sizes; nothing is known to refuse there.
    assert catalog.flags_error("granite-4.2-30b", draft) is None
    # Switching speculation off above four requests failed while capturing graphs.
    scheduled = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":3,'
                 '"num_speculative_tokens_per_batch_size":[[1,4,3],[5,32,0]]}',
                 "--prefix-cache-retention-interval", "1600"]
    assert "zero speculative tokens" in catalog.flags_error("qwen3.8-27b", usual + scheduled)
    assert catalog.flags_error("qwen3.8-27b", usual + [
        "--speculative-config", '{"method":"mtp","num_speculative_tokens_per_batch_size":[[1,32,3]]}',
        "--prefix-cache-retention-interval", "1600"]) is None
    assert catalog.flags_error("granite-4.2-30b", ["--speculative-config", "not json"]) is None


def test_granite_refuses_an_image_without_the_symmetric_gptq_runtime_fix():
    assert "zero-point fix" in catalog.runtime_features_error("granite-4.2-30b", {})
    assert catalog.runtime_features_error("granite-4.2-30b", {"local-llm.rdna-symmetric-gptq": "1"}) is None
    assert catalog.runtime_features_error("nemotron-3.5-lightning-30b-a3b", {}) is None


def test_olmo_refuses_an_image_without_reasoning_separation_and_counting():
    assert "reasoning separation" in catalog.runtime_features_error("olmo-3.1-32b-think", {})
    assert catalog.runtime_features_error("olmo-3.1-32b-think", {"local-llm.olmo-reasoning": "1"})
    assert catalog.runtime_features_error("olmo-3.1-32b-think", {"local-llm.olmo-reasoning": "2"})
    assert catalog.runtime_features_error("olmo-3.1-32b-think", {"local-llm.olmo-reasoning": "3"}) is None


def test_postload_runtime_changes_get_a_separate_compile_cache(monkeypatch):
    first = catalog.cache_namespace("granite-4.2-30b")
    assert first.startswith("runtime-")
    assert catalog.cache_namespace("qwen3-omni-30b") == ""
    monkeypatch.setitem(catalog.MODELS["granite-4.2-30b"], "required_runtime_features",
                        {"local-llm.rdna-symmetric-gptq": "2"})
    assert catalog.cache_namespace("granite-4.2-30b") != first


@pytest.fixture(autouse=True)
def isolated_cache_routes(tmp_path, monkeypatch):
    settings = tmp_path / "model-caches.json"
    monkeypatch.setattr(catalog, "MODEL_CACHES_FILE", settings)
    return settings


@pytest.fixture
def configured_dir(tmp_path, monkeypatch):
    settings = tmp_path / "model-dir"
    monkeypatch.setattr(catalog, "MODEL_DIR_FILE", settings)
    return settings


def test_pinned_snapshot_never_follows_a_moving_main_revision(configured_dir, tmp_path):
    root = tmp_path / "model weights"
    configured_dir.write_text(str(root))
    name, model = next(iter(catalog.MODELS.items()))
    assert catalog.snapshot(model["repo"]) == root / name / model["revision"]


def test_without_a_separate_store_snapshot_is_still_pinned(configured_dir):
    model = catalog.MODELS["olmo-3.1-32b-think"]
    assert catalog.snapshot(model["repo"]) == (
        Path.home() / ".cache/huggingface/hub" / ("models--" + model["repo"].replace("/", "--"))
        / "snapshots" / model["revision"])


def test_cache_routes_select_only_the_named_repository(isolated_cache_routes, tmp_path):
    root = tmp_path / "hot-huggingface"
    isolated_cache_routes.write_text(json.dumps({"qwen3-omni-30b": {
        "repo": "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit", "cache": str(root)}}))
    assert catalog.cache_for_model("qwen3-omni-30b") == root
    assert catalog.cache_for_model("gemma-4-26b") is None
    assert catalog.cache_root("cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit") == root
    assert catalog.cache_root("google/gemma-4-31B-it-qat-w4a16-ct") == catalog.HF_CACHE


@pytest.mark.parametrize("value", [[], {"one": "wrong-shape"}, {"one": {"repo": "owner/name"}},
                                  {"one": {"repo": "owner/name", "cache": "relative"}},
                                  {"one": {"repo": "owner/name", "cache": "/archive/../other"}},
                                  {"one": {"repo": "owner/name", "cache": "/absolute", "typo": True}},
                                  {"one": {"repo": "owner/name", "cache": "/one"},
                                   "two": {"repo": "owner/name", "cache": "/two"}}])
def test_invalid_cache_routing_never_falls_back_to_the_default(isolated_cache_routes, value):
    isolated_cache_routes.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        catalog.cache_root("owner/name")
    with pytest.raises(ValueError):
        catalog.cache_for_model("one")


def test_missing_configured_cache_is_refused_without_falling_back(isolated_cache_routes, tmp_path, monkeypatch):
    root = tmp_path / "missing-hot-cache"
    isolated_cache_routes.write_text(json.dumps({"one": {"repo": "owner/name", "cache": str(root)}}))
    # A native hot cache needs no external-mount entry, but must still exist.
    monkeypatch.setattr("health.storage_error", lambda path: None)
    assert catalog.cache_root("owner/name") == root
    assert "Configured model cache" in catalog.cache_error("owner/name")
    root.mkdir()
    assert catalog.cache_error("owner/name") is None


def test_pinned_cache_mount_keeps_blob_symlinks_inside_the_bind(configured_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "HF_CACHE", tmp_path)
    model = catalog.MODELS["olmo-3.1-32b-think"]
    checkpoint = catalog.snapshot(model["repo"])
    repo = checkpoint.parent.parent
    (repo / "blobs").mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (repo / "blobs/weight").write_bytes(b"test")
    (checkpoint / "model.safetensors").symlink_to("../../blobs/weight")
    source, target, served = catalog.serving_layout(model["repo"])
    assert source == repo
    assert target == "/checkpoint"
    assert served == f"/checkpoint/snapshots/{model['revision']}"
    assert (checkpoint / "model.safetensors").resolve().is_relative_to(source)
    assert (source / Path(served).relative_to(target) / "model.safetensors").read_bytes() == b"test"


def test_plain_model_directory_keeps_its_existing_mount(configured_dir, tmp_path):
    configured_dir.write_text(str(tmp_path / "models"))
    model = catalog.MODELS["granite-4.2-30b"]
    assert catalog.serving_layout(model["repo"]) == (catalog.snapshot(model["repo"]), "/model", "/model")


@pytest.mark.parametrize("value", ["", "relative/path"])
def test_invalid_store_cannot_fall_through_to_the_working_directory(configured_dir, value):
    configured_dir.write_text(value)
    with pytest.raises(ValueError, match="absolute directory"):
        catalog.model_dir()


def test_incomplete_or_unsafe_shards_are_not_admitted(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}))
    assert not catalog.complete(tmp_path)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.touch()
    assert not catalog.complete(tmp_path)
    shard.write_bytes(b"test")
    assert catalog.complete(tmp_path)
    index.write_text(json.dumps({"weight_map": {"weight": "../other.safetensors"}}))
    assert not catalog.complete(tmp_path)
    index.write_text("broken json")
    assert not catalog.complete(tmp_path)
    index.write_text(json.dumps({"weight_map": []}))
    assert not catalog.complete(tmp_path)


def test_capabilities_and_launch_parsers_match_the_exact_model_generation():
    granite = catalog.MODELS["granite-4.2-30b"]
    assert granite["reasoning_parser"] == "qwen3"
    assert granite["tool_parser"] == "qwen3_coder"
    olmo = catalog.MODELS["olmo-3.1-32b-think"]
    assert not olmo["capabilities"]["tools"]
    assert "--enable-auto-tool-choice" not in catalog.flags(olmo)
    mistral = catalog.MODELS["mistral-small-3.2-24b"]
    assert "--reasoning-parser" not in catalog.flags(mistral)
    for model in catalog.MODELS.values():
        assert model["capabilities"]["input"][0] == "text"
        flags = catalog.flags(model)
        # One GPU call per 512 tokens keeps prefill under the GPU watchdog (README).
        steps = ("--batch-size", "--ubatch-size") if catalog.engine(model) == "llama.cpp" else ("--max-num-batched-tokens",)
        assert all(flags[flags.index(step) + 1] == "512" for step in steps)


def test_a_gguf_checkpoint_is_complete_only_with_its_chosen_files(tmp_path):
    files = catalog.checkpoint_files(catalog.MODELS["inkling-2-mini"])
    assert files == ["Inkling-2-Mini-IQ4_NL.gguf", "mmproj-Inkling-2-Mini-Q8_0.gguf"]
    (tmp_path / files[0]).write_bytes(b"GGUF")
    assert not catalog.complete(tmp_path, files)
    (tmp_path / files[1]).touch()
    assert not catalog.complete(tmp_path, files)
    (tmp_path / files[1]).write_bytes(b"GGUF")
    assert catalog.complete(tmp_path, files)
    assert catalog.checkpoint_files(catalog.MODELS["granite-4.2-30b"]) is None


def test_llama_cpp_models_get_their_files_and_their_own_image(configured_dir, tmp_path):
    configured_dir.write_text(str(tmp_path / "models"))
    args = catalog.serve_args("inkling-2-mini")
    assert args == ["--model", "/model/Inkling-2-Mini-IQ4_NL.gguf",
                    "--mmproj", "/model/mmproj-Inkling-2-Mini-Q8_0.gguf"]
    rows = subprocess.run(["python3", str(Path(catalog.__file__)), "rows"], check=True,
                          capture_output=True, text=True).stdout.splitlines()
    row = next(line.split("|") for line in rows if line.startswith("inkling-2-mini|"))
    assert row[5] == "llama.cpp" and row[6] == catalog.ENGINE_IMAGES["llama.cpp"]
    granite = next(line.split("|") for line in rows if line.startswith("granite-4.2-30b|"))
    assert granite[5:] == ["vllm", ""]


def test_a_copy_is_listed_ten_ports_up_only_while_asked_for():
    script = Path(__file__).with_name("llm")
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "LOCAL_LLM_COPY": "qwen3-omni-30b"}
    listed = subprocess.run([str(script), "list"], check=True, text=True, capture_output=True, env=environment)
    entries = dict(line.split() for line in listed.stdout.splitlines())
    assert entries["qwen3-omni-30b"] == "8006" and entries["qwen3-omni-30b-copy"] == "8016"
    plain = subprocess.run([str(script), "list"], check=True, text=True, capture_output=True,
                           env=environment | {"LOCAL_LLM_COPY": ""})
    assert "copy" not in plain.stdout


def test_cli_discovers_new_models_without_removing_the_existing_ones():
    script = Path(__file__).with_name("llm")
    result = subprocess.run([str(script), "list"], check=True, text=True, capture_output=True)
    entries = dict(line.split() for line in result.stdout.splitlines())
    assert entries["qwen3-omni-30b"] == "8006"
    assert entries["gemma-4-26b"] == "8002"
    assert {name: entries[name] for name in catalog.MODELS} == {
        name: str(model["port"]) for name, model in catalog.MODELS.items()}


def test_download_selection_is_explicit_and_deduplicated():
    from download_models import select_models

    models = [{"name": "one"}, {"name": "two"}]
    assert select_models(models, []) == models
    assert select_models(models, ["two", "two"]) == [models[1]]
    with pytest.raises(ValueError, match="unknown model"):
        select_models(models, ["typo"])


def test_finished_files_are_reused_but_partial_files_need_space(tmp_path):
    from download_models import remaining_bytes

    (tmp_path / "done").write_bytes(b"1234")
    (tmp_path / "partial").write_bytes(b"12")
    assert remaining_bytes([("done", 4), ("partial", 4), ("missing", 8)], tmp_path) == 12


def test_download_admission_keeps_reserve_and_refuses_missing_mount_before_stat(tmp_path, monkeypatch):
    import download_models as downloads

    monkeypatch.setattr(downloads, "storage_error", lambda path: None)
    monkeypatch.setattr(downloads.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    downloads.require_space(tmp_path / "future", 60, reserve=40)
    with pytest.raises(ValueError, match="reserve required"):
        downloads.require_space(tmp_path / "future", 61, reserve=40)
    monkeypatch.setattr(downloads, "storage_error", lambda path: "archive is not mounted")
    monkeypatch.setattr(downloads.shutil, "disk_usage", lambda path: pytest.fail("must not stat unavailable storage"))
    with pytest.raises(ValueError, match="not mounted"):
        downloads.require_space(tmp_path / "future", 1)


@pytest.mark.parametrize("public", [True, False])
def test_public_additions_ignore_stale_login_but_existing_gated_downloads_keep_it(tmp_path, monkeypatch, public):
    import huggingface_hub
    import download_models as downloads

    model = ({"name": "granite-4.2-30b", "repo": catalog.MODELS["granite-4.2-30b"]["repo"]}
             if public else {"name": "gemma-4-12b", "repo": "google/gemma-4-12B-it-qat-w4a16-ct"})
    revision = catalog.MODELS[model["name"]]["revision"] if public else "moving-main-revision"
    info_calls, download_calls = [], []

    class Api:
        def model_info(self, repo, **kwargs):
            info_calls.append((repo, kwargs))
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename="config.json", size=2)])

    target = tmp_path / "weights"
    monkeypatch.setattr(downloads, "cache_root", lambda repo: tmp_path)
    monkeypatch.setattr(downloads, "cache_error", lambda repo: None)
    if not public:
        target = tmp_path / "hub" / ("models--" + model["repo"].replace("/", "--")) / "snapshots" / revision
    monkeypatch.setattr(downloads, "model_dir", lambda: tmp_path if public else None)
    monkeypatch.setattr(downloads, "snapshot", lambda repo: target)
    monkeypatch.setattr(downloads, "require_space", lambda *args, **kwargs: None)
    monkeypatch.setattr(downloads, "storage_error", lambda path: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)

    def transfer(repo, **kwargs):
        download_calls.append((repo, kwargs))
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", transfer)
    downloads.download(model)
    for _, options in info_calls + download_calls:
        if public:
            assert options["token"] is False
        else:
            assert "token" not in options
    assert download_calls[0][1]["revision"] == revision
    if public:
        assert download_calls[0][1]["local_dir"] == target
    else:
        assert download_calls[0][1]["cache_dir"] == tmp_path / "hub"


def test_ignored_duplicate_weights_are_neither_counted_nor_fetched(tmp_path, monkeypatch):
    import huggingface_hub
    import download_models as downloads

    name = "ministral-3-3b"
    model = {"name": name, "repo": catalog.MODELS[name]["repo"]}
    siblings = [SimpleNamespace(rfilename="model.safetensors", size=4),
                SimpleNamespace(rfilename="consolidated.safetensors", size=4)]
    counted, fetched = [], []

    class Api:
        def model_info(self, repo, **kwargs):
            return SimpleNamespace(sha=catalog.MODELS[name]["revision"], siblings=siblings)

    def transfer(repo, **kwargs):
        fetched.append(kwargs)
        (tmp_path / "model.safetensors").write_bytes(b"1234")

    monkeypatch.setattr(downloads, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(downloads, "snapshot", lambda repo: tmp_path)
    monkeypatch.setattr(downloads, "storage_error", lambda path: None)
    monkeypatch.setattr(downloads, "require_space", lambda target, missing: counted.append(missing))
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", transfer)
    downloads.download(model)
    assert counted == [4]
    assert fetched[0]["ignore_patterns"] == ["consolidated.safetensors", "tekken.json"]


def test_a_gguf_repository_fetches_only_the_chosen_quantization(tmp_path, monkeypatch):
    import huggingface_hub
    import download_models as downloads

    name = "inkling-2-mini"
    pinned = catalog.MODELS[name]
    model = {"name": name, "repo": pinned["repo"]}
    siblings = [SimpleNamespace(rfilename=pinned["gguf"], size=4),
                SimpleNamespace(rfilename=pinned["mmproj"], size=2),
                SimpleNamespace(rfilename="Inkling-2-Mini-BF16.gguf", size=52)]
    counted, fetched = [], []

    class Api:
        def model_info(self, repo, **kwargs):
            return SimpleNamespace(sha=pinned["revision"], siblings=siblings)

    def transfer(repo, **kwargs):
        fetched.append(kwargs)
        (tmp_path / pinned["gguf"]).write_bytes(b"1234")
        (tmp_path / pinned["mmproj"]).write_bytes(b"12")

    monkeypatch.setattr(downloads, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(downloads, "snapshot", lambda repo: tmp_path)
    monkeypatch.setattr(downloads, "storage_error", lambda path: None)
    monkeypatch.setattr(downloads, "require_space", lambda target, missing: counted.append(missing))
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", transfer)
    downloads.download(model)
    assert counted == [6]
    assert fetched[0]["allow_patterns"] == [pinned["gguf"], pinned["mmproj"]]
    siblings.pop(1)
    with pytest.raises(ValueError, match="lacks mmproj"):
        downloads.download(model)


def test_missing_selected_cache_refuses_download_before_hub_access(monkeypatch):
    import huggingface_hub
    import download_models as downloads

    monkeypatch.setattr(downloads, "cache_error", lambda repo: "Configured model cache is missing")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: pytest.fail("must not contact Hub for unavailable cache"))
    with pytest.raises(ValueError, match="Configured model cache is missing"):
        downloads.download({"name": "one", "repo": "owner/name"})


def test_broken_routing_link_is_not_treated_as_an_absent_optional_config(isolated_cache_routes, tmp_path):
    isolated_cache_routes.symlink_to(tmp_path / "missing-settings")
    with pytest.raises(ValueError, match="routing link.*broken"):
        catalog.cache_root("owner/name")


def test_hub_failure_reports_status_without_dumping_credentials(tmp_path, monkeypatch, capsys):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError
    import download_models as downloads

    model = {"name": "one", "repo": "example/one"}
    monkeypatch.setattr(downloads, "configured_models", lambda: [model])
    monkeypatch.setattr("sys.argv", ["download_models.py"])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    def refuse(*args, **kwargs):
        response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/api/models/example/one"))
        raise HfHubHTTPError("opaque-test-credential-must-stay-private", response=response)

    monkeypatch.setattr(downloads, "download", refuse)
    with pytest.raises(SystemExit, match="Download failed for one"):
        downloads.main()
    output = capsys.readouterr().out
    assert "Hub HTTP 401" in output
    assert "valid local Hugging Face login" in output
    assert "opaque-test-credential" not in output


def test_a_parser_the_image_lacks_is_refused_before_start(monkeypatch):
    available = {"tool": ["minicpm5", "mistral", "qwen3_coder"], "reasoning": ["qwen3"]}
    assert catalog.parsers_error("minicpm5-2b", available) is None
    # No reasoning parser at all is valid: Mistral models do not think.
    assert catalog.parsers_error("mistral-small-3.2-24b", available) is None
    monkeypatch.setitem(catalog.MODELS["minicpm5-2b"], "tool_parser", "minicpm5xml")
    assert "'minicpm5xml'" in catalog.parsers_error("minicpm5-2b", available)


def test_a_leftover_ignored_file_is_refused_before_start(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "snapshot", lambda repo: tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"1")
    assert catalog.ignored_files_error("ministral-3-3b") is None
    (tmp_path / "tekken.json").write_text("{}")
    assert "tekken.json" in catalog.ignored_files_error("ministral-3-3b")
    assert catalog.ignored_files_error("granite-4.2-3b") is None


def test_kv_cache_estimate_counts_full_attention_and_skips_hybrids(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "snapshot", lambda repo: tmp_path)
    config = {"num_hidden_layers": 52, "num_key_value_heads": 8, "head_dim": 128,
              "hidden_size": 5120, "num_attention_heads": 40}
    (tmp_path / "config.json").write_text(json.dumps(config))
    # vLLM asked Nanbeige for 6.5 GiB at 32k and refused to start without it.
    assert catalog.kv_cache_bytes("nanbeige-4.3-33b") == 52 * 2 * 8 * 128 * 2 * 32768
    (tmp_path / "config.json").write_text(json.dumps(config | {"hybrid_override_pattern": "M-M*"}))
    assert catalog.kv_cache_bytes("nanbeige-4.3-33b") == 0
    assert catalog.kv_cache_bytes("qwen3.8-27b") == 0


def test_a_llama_cpp_model_refuses_settings_whose_memory_was_not_measured():
    measured = catalog.flags(catalog.MODELS["inkling-2-mini"])
    assert catalog.flags_error("inkling-2-mini", measured) is None
    # 262k took over 44 GB where llama.cpp projected 32, and the kernel killed it.
    assert "--ctx-size 65536, not 262144" in catalog.flags_error("inkling-2-mini", [*measured, "--ctx-size", "262144"])
    assert "--parallel" in catalog.flags_error("inkling-2-mini-copy", [*measured, "--parallel", "8"])
    assert "not 262144" in catalog.flags_error("inkling-2-mini", [*measured, "-c", "262144"])
    assert catalog.memory_bytes("inkling-2-mini") > 26 * catalog.GIB
    assert catalog.memory_bytes("granite-4.2-30b") == 0
