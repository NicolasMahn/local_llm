"""Pinned additions to the serving catalog and their checkpoint locations.

Launch values are conservative baselines until a recorded serving comparison
justifies changing them. Quantization names describe the stored weights, not
the quantizer's repository naming convention.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "local-llm"
MODEL_DIR_FILE = CONFIG / "model-dir"
MODEL_CACHES_FILE = CONFIG / "model-caches.json"
HF_CACHE = Path.home() / ".cache/huggingface"
GIB = 1024**3
NEMOTRON_MOE_PROFILE = "E=128,N=1856,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json"
MODELS = {
    "granite-4.2-30b": {
        "port": 8007, "share": 0.23,
        "source": "ibm-granite/granite-4.2-30b",
        "repo": "derikn/Granite-4.2-30B-GPTQ-INT4",
        "revision": "fc0411d102bae7bd3446b0d477ecaca9f8a49961",
        "quantization": "AutoRound 0.15.1 GPTQ INT4 symmetric group128, BF16 activations",
        "reasoning_parser": "qwen3", "tool_parser": "qwen3_coder",
        "required_runtime_features": {"local-llm.rdna-symmetric-gptq": "1"},
        "context": 32768, "max_seqs": 16,
        "tuning": "measured on gfx1151; 512-token prefill, selected dense kernels, 0.23 share; benchmarks/added-models-2026-11-12.md",
        "capabilities": {"input": ["text"], "reasoning": True, "thinking_toggle": True,
                         "tools": True, "structured_output": True},
    },
    "nemotron-3.5-lightning-30b-a3b": {
        "port": 8008, "share": 0.22,
        "source": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "repo": "useful-quants/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-W4A16",
        "revision": "cc977f002c7a396da87dc34c7bb4e5cdba710e24",
        "quantization": "compressed-tensors INT4 symmetric group64, BF16 activations",
        "reasoning_parser": "nemotron_v3", "tool_parser": "qwen3_coder",
        "context": 32768, "max_seqs": 16,
        "tuning": "measured on gfx1151; single-token vector experts, FP32 Mamba cache, 0.22 share; benchmarks/all-models-2026-11-14.md",
        "capabilities": {"input": ["text"], "reasoning": True, "thinking_toggle": True,
                         "tools": True, "structured_output": True},
    },
    "olmo-3.1-32b-think": {
        "port": 8009, "share": 0.24,
        "source": "allenai/Olmo-3.1-32B-Think",
        "repo": "cyankiwi/Olmo-3.1-32B-Think-AWQ-4bit",
        "revision": "0aebbdfaa8ba57bada66e87eb1c21c1a30ac2faa",
        "quantization": "compressed-tensors INT4 symmetric group32, BF16 activations",
        "reasoning_parser": "olmo3", "tool_parser": None,
        "required_runtime_features": {"local-llm.olmo-reasoning": "3"},
        "context": 32768, "max_seqs": 16,
        "tuning": "measured on gfx1151; selected group32 dense kernels, 0.24 share; benchmarks/added-models-2026-11-12.md",
        # Ai2 recommends sampling; a fixed seed makes variant comparisons repeatable.
        "benchmark_sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 42},
        "capabilities": {"input": ["text"], "reasoning": True, "thinking_toggle": False,
                         "tools": False, "structured_output": True},
    },
    # Additions of 2026-11-16, picked from Artificial Analysis' small (4-40B)
    # and tiny (<4B) open-weights rankings; benchmarks/small-models-2026-11-16.md.
    "nanbeige-4.3-33b": {
        "port": 8020, "share": 0.31,
        "source": "Nanbeige/Nanbeige4.3-33B",
        "repo": "Nanbeige/Nanbeige4.3-33B-INT4",
        "revision": "d834b414cf1c47e42640de33c147b7cf8a9321ef",
        "quantization": "compressed-tensors INT4 asymmetric group32, BF16 activations",
        "reasoning_parser": "qwen3", "tool_parser": "qwen3_coder",
        "context": 32768, "max_seqs": 16,
        "capabilities": {"input": ["text"], "reasoning": True, "thinking_toggle": True,
                         "tools": True, "structured_output": True},
    },
    "mistral-small-3.2-24b": {
        "port": 8021, "share": 0.19,
        "source": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "repo": "Intel/Mistral-Small-3.2-24B-Instruct-2506-int4-AutoRound",
        "revision": "21907bf9a442ec05c91e059ef17b08a8b44983f5",
        "ignore": ["tekken.json"],  # see ministral-3-3b
        "quantization": "AutoRound GPTQ INT4 symmetric group128 (language model only), BF16 activations",
        "reasoning_parser": None, "tool_parser": "mistral",
        "required_runtime_features": {"local-llm.rdna-symmetric-gptq": "1",
                                      "local-llm.pixtral-positions": "1"},
        "context": 32768, "max_seqs": 16,
        "capabilities": {"input": ["text", "image"], "reasoning": False, "thinking_toggle": False,
                         "tools": True, "structured_output": True},
    },
    "ministral-3-3b": {
        "port": 8022, "share": 0.08,
        "source": "mistralai/Ministral-3-3B-Instruct-2512",
        "repo": "cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-4bit",
        "revision": "61379da9054d715381033d9916619f8e4f03fe94",
        # The same weights again in Mistral's own layout. Without tekken.json,
        # Transformers 5 loads the standard tokenizer, whose ids match
        # Mistral's; with it, vLLM's image processor misses every [IMG] token.
        "ignore": ["consolidated.safetensors", "tekken.json"],
        "quantization": "compressed-tensors INT4 symmetric group32 (language model only), BF16 activations",
        "reasoning_parser": None, "tool_parser": "mistral",
        "required_runtime_features": {"local-llm.pixtral-positions": "1"},
        "context": 32768, "max_seqs": 16,
        "capabilities": {"input": ["text", "image"], "reasoning": False, "thinking_toggle": False,
                         "tools": True, "structured_output": True},
    },
    "nemotron-3-nano-4b": {
        "port": 8023, "share": 0.12,
        "source": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "repo": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "quantization": "none (BF16); no trusted INT4 build exists",
        "reasoning_parser": "nemotron_v3", "tool_parser": "qwen3_coder",
        "context": 32768, "max_seqs": 16,
        "capabilities": {"input": ["text"], "reasoning": True, "thinking_toggle": True,
                         "tools": True, "structured_output": True},
    },
    "granite-4.2-3b": {
        "port": 8024, "share": 0.10,
        "source": "ibm-granite/granite-4.2-3b",
        "repo": "ibm-granite/granite-4.2-3b",
        "revision": "4e9c275e232f10211ec88644d106a1f6a1b60e06",
        "quantization": "none (BF16); no trusted INT4 build exists",
        "reasoning_parser": "qwen3", "tool_parser": "qwen3_coder",
        "context": 32768, "max_seqs": 16,
        "capabilities": {"input": ["text"], "reasoning": True, "thinking_toggle": True,
                         "tools": True, "structured_output": True},
    },
    "minicpm5-2b": {
        "port": 8025, "share": 0.06,
        "source": "openbmb/MiniCPM5-2B",
        "repo": "cyankiwi/MiniCPM5-2B-AWQ-INT4",
        "revision": "fd622ce683e142b56a8ed78fa8a649a7254041f5",
        "quantization": "compressed-tensors INT4 asymmetric group32, BF16 activations",
        "reasoning_parser": "qwen3", "tool_parser": "minicpm5",
        "context": 32768, "max_seqs": 16,
        "capabilities": {"input": ["text"], "reasoning": True, "thinking_toggle": True,
                         "tools": True, "structured_output": True},
    },
    # vLLM runs Inkling 2 only from its NVFP4/MXFP8 weights on Hopper or newer
    # (2026-11-21), so this one is served by llama.cpp, which added the looped
    # architecture on 2026-11-20 (ggml-org/llama.cpp#32694).
    "inkling-2-mini": {
        "port": 8026, "share": 0.25,
        "engine": "llama.cpp",
        "source": "thinkingmachines/Inkling-2-Mini",
        "repo": "ggml-org/Inkling-2-Mini-GGUF",
        "revision": "e8a2cecf0e78914f07cc27b9be711a9b2c7561d9",
        "gguf": "Inkling-2-Mini-IQ4_NL.gguf",
        "mmproj": "mmproj-Inkling-2-Mini-Q8_0.gguf",
        "quantization": "GGUF IQ4_NL experts in layers 0-19 (imatrix), Q8_0 elsewhere",
        # llama.cpp takes what this needs, whatever the share says, and its own
        # projection undercounts: at 262k it promised 32 GB, took over 44 and
        # the kernel killed it for want of RAM (2026-11-21). So the context
        # stays where its memory was measured: 26.3 GB loaded, 26.6 GB at peak
        # with a 60k-token prompt. Each step's working memory is 6.5 GB at any
        # context; upstream's ggml-org/llama.cpp#32752 should make longer ones cheaper.
        "context": 65536, "max_seqs": 4, "measured_gib": 26.6,
        "tuning": "measured on gfx1151 next to a busy gemma-4-31b; fused glue kernels on ROCm; benchmarks/inkling-2-mini-2026-11-21.md",
        # Its template sets thinking with chat_template_kwargs.reasoning_effort,
        # 0.0 to 0.99 (default 0.9); even 0.0 thinks a few sentences.
        "capabilities": {"input": ["text", "image"], "reasoning": True, "thinking_toggle": False,
                         "tools": True, "structured_output": True},
    },
}

# llama.cpp is not in the vLLM image. Containerfile.llama-cpp builds this one,
# and ./llm up runs that build when the tag is missing; change the tag
# whenever the Containerfile's pinned sources change.
ENGINE_IMAGES = {
    "llama.cpp": "localhost/local-llm-llama-cpp:inkling2-speed-20261121",
}


def engine(model):
    return model.get("engine", "vllm")


def checkpoint_files(model):
    """The files a GGUF model needs, or None for a full Hugging Face snapshot."""
    if engine(model) != "llama.cpp":
        return None
    return [model["gguf"]] + ([model["mmproj"]] if model.get("mmproj") else [])


def for_repo(repo):
    return next(((name, model) for name, model in MODELS.items() if model["repo"] == repo), None)


def model_caches():
    """Validate the whole routing map so a typo cannot silently choose an archive."""
    try:
        entries = json.loads(MODEL_CACHES_FILE.read_text())
    except FileNotFoundError:
        if MODEL_CACHES_FILE.is_symlink():
            raise ValueError(f"Model cache routing link {MODEL_CACHES_FILE} is broken")
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read model cache routing {MODEL_CACHES_FILE}: {exc}") from exc
    if not isinstance(entries, dict):
        raise ValueError(f"{MODEL_CACHES_FILE} must contain an object keyed by model name")
    repos = set()
    for name, entry in entries.items():
        if (not isinstance(name, str) or not name or not isinstance(entry, dict)
                or set(entry) != {"repo", "cache"}
                or not isinstance(entry["repo"], str) or not entry["repo"]
                or not isinstance(entry["cache"], str) or not entry["cache"]
                or not Path(entry["cache"]).is_absolute()
                or ".." in Path(entry["cache"]).parts
                or entry["repo"] in repos):
            raise ValueError(f"Invalid cache route in {MODEL_CACHES_FILE}: each model needs a unique repo and absolute cache directory")
        repos.add(entry["repo"])
    return entries


def cache_for_model(name):
    entry = model_caches().get(name)
    return Path(entry["cache"]) if entry else None


def cache_root(repo):
    entry = next((entry for entry in model_caches().values() if entry["repo"] == repo), None)
    return Path(entry["cache"]) if entry else HF_CACHE


def cache_error(repo):
    """A configured hot cache is required; falling back could break boot serving."""
    from health import storage_error

    root = cache_root(repo)
    if reason := storage_error(root):
        return reason
    if any(entry["repo"] == repo for entry in model_caches().values()) and not root.is_dir():
        return f"Configured model cache {root} is missing; restore it before starting or downloading this model"
    return None


def model_dir():
    """A separate, optional local-dir store; existing HF cache stays untouched."""
    try:
        value = MODEL_DIR_FILE.read_text().strip()
    except FileNotFoundError:
        if MODEL_DIR_FILE.is_symlink():
            raise ValueError(f"Model storage link {MODEL_DIR_FILE} is broken")
        return None
    root = Path(value).expanduser()
    if not value or not root.is_absolute():
        raise ValueError(f"{MODEL_DIR_FILE} must contain an absolute directory")
    return root


def snapshot(repo):
    name, model = for_repo(repo)
    root = model_dir()
    if root is not None:
        return root / name / model["revision"]
    return cache_root(repo) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / model["revision"]


def serving_layout(repo):
    checkpoint = snapshot(repo)
    if model_dir() is not None:
        return checkpoint, "/model", "/model"
    # A cache snapshot links ../../blobs. Binding only that snapshot breaks
    # those links inside the container; keep its entire repository visible.
    return checkpoint.parent.parent, "/checkpoint", f"/checkpoint/snapshots/{checkpoint.name}"


def complete(path, files=None):
    """Do not admit a checkpoint while an indexed shard is still downloading.

    A GGUF checkpoint has no config.json; `files` names what it needs instead.
    """
    path = Path(path)
    if files is not None:
        try:
            return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in files)
        except OSError:
            return False
    try:
        if not isinstance(json.loads((path / "config.json").read_text()), dict):
            return False
        index = path / "model.safetensors.index.json"
        if index.exists():
            indexed = json.loads(index.read_text())
            if not isinstance(indexed, dict) or not isinstance(indexed.get("weight_map"), dict):
                return False
            shards = set(indexed["weight_map"].values())
            if not shards:
                return False
        else:
            shards = {"model.safetensors"}
        return all(isinstance(shard, str) and Path(shard).name == shard and (path / shard).is_file()
                   and (path / shard).stat().st_size > 0 for shard in shards)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def flags(model):
    if engine(model) == "llama.cpp":
        return llama_flags(model)
    result = ["--max-model-len", str(model["context"]), "--max-num-batched-tokens", str(model.get("prefill", 512)),
              "--max-num-seqs", str(model["max_seqs"]), "--dtype", "bfloat16",
              "--attention-backend", "TRITON_ATTN", "--enable-prompt-tokens-details"]
    if model["reasoning_parser"]:
        result += ["--reasoning-parser", model["reasoning_parser"]]
    if model["tool_parser"]:
        result += ["--tool-call-parser", model["tool_parser"], "--enable-auto-tool-choice",
                   "--exclude-tools-when-tool-choice-none"]
    return result


def llama_flags(model):
    # Batches of 512 keep each GPU call short, as for vLLM (README). One KV
    # pool shared by all slots lets a single request use the whole context.
    # --jinja applies the model's own chat template, which handles tools and
    # thinking; --metrics feeds the monitor. --cache-ram 0: llama-server
    # otherwise keeps up to 8 GiB of old prompts in RAM, outside the measured
    # memory. A slot still reuses its conversation's prompt until another
    # conversation needs the room; then that prompt is read again.
    return ["--ctx-size", str(model["context"]), "--parallel", str(model["max_seqs"]), "--kv-unified",
            "--batch-size", "512", "--ubatch-size", "512", "--n-gpu-layers", "999",
            "--flash-attn", "on", "--cache-ram", "0", "--jinja", "--metrics", "--no-webui"]


def serve_args(name):
    """The model files for llama-server, as paths inside the container."""
    model = MODELS[name]
    target = serving_layout(model["repo"])[2]
    result = ["--model", f"{target}/{model['gguf']}"]
    if model.get("mmproj"):
        result += ["--mmproj", f"{target}/{model['mmproj']}"]
    return result


def runtime_features_error(name, labels):
    descriptions = {
        "local-llm.rdna-symmetric-gptq": "the RDNA symmetric GPTQ zero-point fix",
        "local-llm.olmo-reasoning": "the Olmo reasoning separation and token-count fix",
        "local-llm.pixtral-positions": "the Pixtral image-position fix for Transformers 5",
    }
    for feature, version in MODELS.get(name, {}).get("required_runtime_features", {}).items():
        if (labels or {}).get(feature) != version:
            return (f"{name} requires {descriptions.get(feature, feature)}; "
                    "rebuild the local model image before starting it (Containerfile)")
    return None


def cache_namespace(name):
    features = MODELS.get(name, {}).get("required_runtime_features")
    if not features:
        return ""
    import hashlib

    signature = json.dumps(features, sort_keys=True).encode()
    return "runtime-" + hashlib.sha256(signature).hexdigest()[:12]


def runtime_error(name, image="localhost/local-llm:latest"):
    if not MODELS.get(name, {}).get("required_runtime_features"):
        return None
    import subprocess

    try:
        result = subprocess.run(["podman", "image", "inspect", image], capture_output=True,
                                text=True, check=True, timeout=10)
        metadata = json.loads(result.stdout)[0]
        labels = metadata.get("Labels") or metadata.get("Config", {}).get("Labels") or {}
        return runtime_features_error(name, labels)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return f"Cannot verify model runtime {image}: {type(exc).__name__}"


# Reads vLLM's parser tables as source: importing vLLM needs a GPU.
_LIST_PARSERS = """
import ast, importlib.util, json, pathlib
root = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
names = {}
for kind, table in (("tool", "tool_parsers"), ("reasoning", "reasoning")):
    tree = ast.parse((root / table / "__init__.py").read_text())
    names[kind] = sorted(key.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                         for key in node.keys if isinstance(key, ast.Constant))
print(json.dumps(names))
"""


def parsers_error(name, available):
    """A misspelt parser lets vLLM start, then crash-loop on its first config check."""
    model = MODELS.get(name, {})
    for kind, chosen in (("tool", model.get("tool_parser")), ("reasoning", model.get("reasoning_parser"))):
        if chosen and chosen not in available.get(kind, []):
            return (f"{name} names {kind} parser {chosen!r}, which the model image does not have; "
                    f"it has {', '.join(available.get(kind, []))}")
    return None


def kv_cache_bytes(name):
    """The KV cache one request at full context needs, or 0 where unknown.

    vLLM refuses to start below it, after loading the weights; Nanbeige did
    on 2026-11-16 with weights that fit its share. Hybrid and sliding-window
    models keep far less per token than this formula says, so they get 0
    rather than a false refusal.
    """
    model = MODELS.get(name)
    if model is None:
        return 0
    try:
        config = json.loads((snapshot(model["repo"]) / "config.json").read_text())
    except (OSError, ValueError):
        return 0
    text = config.get("text_config", config)
    if (text.get("sliding_window") or any(key in text for key in
            ("layer_types", "hybrid_override_pattern", "layers_block_type"))):
        return 0
    try:
        head_dim = text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]
        per_token = 2 * text["num_hidden_layers"] * text["num_key_value_heads"] * head_dim * 2  # K and V, BF16
    except (KeyError, TypeError, ZeroDivisionError):
        return 0
    return per_token * model["context"]


def memory_bytes(name):
    """What a llama.cpp model was measured to take at its settings, or 0."""
    return int(MODELS.get(name, {}).get("measured_gib", 0) * GIB)


def ignored_files_error(name):
    """An ignored file left in the folder still loads: a tekken.json from an
    earlier copy brings back the image crash ministral-3-3b documents."""
    from fnmatch import fnmatch

    model = MODELS.get(name, {})
    folder = snapshot(model["repo"]) if model else None
    if not model.get("ignore") or not folder.is_dir():
        return None
    found = sorted(path.name for path in folder.iterdir()
                   if any(fnmatch(path.name, pattern) for pattern in model["ignore"]))
    if found:
        return f"{name} must not have {', '.join(found)} in {folder}; remove it before starting"
    return None


def image_parsers(image):
    import subprocess

    result = subprocess.run(["podman", "run", "--rm", "--network", "none", "--entrypoint", "python3",
                             image, "-c", _LIST_PARSERS], capture_output=True, text=True, check=True, timeout=60)
    return json.loads(result.stdout)


def flags_error(name, flags):
    """Refuse launch flags known to end badly, before any weights load.

    Gemma's long-context prefill only runs on an unquantized cache:
    vllm_plugins/local_llm_attention.py takes over that prefill, and steps
    aside for a quantized cache. vLLM's own kernel then needs more than the
    GPU watchdog's two seconds for one call past 64k tokens (README).

    Qwen3.8 with speculative decoding has a scheduler block of 1600 tokens
    and refuses its usual cache interval of 1568 after loading the weights;
    on 2026-11-15 that crashed three starts in a row. So did a schedule that
    switches speculation off for some batch sizes: the engine fails an
    assertion while capturing graphs for zero speculative tokens.
    """
    def last(flag):
        # vLLM takes the last one when a flag is repeated.
        values = [value for name_, value in zip(flags, flags[1:]) if name_ == flag]
        return values[-1] if values else None

    model = MODELS.get(name.removesuffix("-copy"), {})
    if engine(model) == "llama.cpp":
        # llama.cpp also takes the last value, under the short name or with underscores.
        short = {"-c": "--ctx-size", "-np": "--parallel", "-b": "--batch-size",
                 "-ub": "--ubatch-size", "-cram": "--cache-ram"}
        flags = [short.get(flag, flag.replace("_", "-") if flag.startswith("--") else flag) for flag in flags]
        measured = dict(zip(llama_flags(model), llama_flags(model)[1:]))
        for flag in ("--ctx-size", "--parallel", "--batch-size", "--ubatch-size", "--cache-ram"):
            if last(flag) != measured[flag]:
                return (f"{name} was measured with {flag} {measured[flag]}, not {last(flag)}; "
                        "llama.cpp ignores the memory share and undercounts what other "
                        "settings take, so measure them before changing the catalog")
        return None
    dtype = last("--kv-cache-dtype")
    if dtype not in (None, "auto") and name.startswith("gemma-"):
        return (f"{name} needs an unquantized KV cache: its long-context attention "
                f"protection does not cover {dtype}")
    try:
        schedule = json.loads(last("--speculative-config") or "{}").get("num_speculative_tokens_per_batch_size")
        if any(entry[2] == 0 for entry in schedule or []):
            return (f"{name} cannot switch speculative decoding off for some batch sizes: "
                    "the engine fails while capturing graphs for zero speculative tokens")
    except (ValueError, TypeError, IndexError, AttributeError):
        pass  # vLLM reports a malformed value itself, before it loads anything
    if (name.removesuffix("-copy") == "qwen3.8-27b" and "--speculative-config" in flags
            and last("--prefix-cache-retention-interval") != "1600"):
        return (f"{name} with speculative decoding needs --prefix-cache-retention-interval 1600; "
                "the engine refuses any other interval after loading the weights")
    return None


def moe_profile_error(folder):
    """Never label a missing profile experiment as a native-profile comparison."""
    path = Path(folder) / NEMOTRON_MOE_PROFILE
    try:
        profile = json.loads(path.read_text())
        if not isinstance(profile, dict) or not all(str(m) in profile for m in range(1, 513)):
            raise ValueError("profile must preserve every batch size from 1 through 512")
        allowed = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                   "SPLIT_K", "num_warps", "num_stages"}
        for step, config in profile.items():
            if step == "triton_version":
                continue
            if (not step.isdecimal() or not isinstance(config, dict) or not config
                    or set(config) - allowed or any(type(v) is not int or v <= 0 for v in config.values())
                    or config.get("BLOCK_SIZE_M") not in (16, 32, 64)
                    or config.get("GROUP_SIZE_M") != 1 or config.get("SPLIT_K") != 1):
                raise ValueError(f"unsupported profile configuration at batch {step}")
    except (OSError, ValueError, TypeError) as exc:
        return f"Invalid Nemotron MoE profile {path}: {exc}"
    return None


def main():
    command = sys.argv[1]
    if command == "rows":
        for name, model in MODELS.items():
            print(f"{name}|{model['port']}|{model['share']}|{model['repo']}|{' '.join(flags(model))}"
                  f"|{engine(model)}|{ENGINE_IMAGES.get(engine(model), '')}")
    elif command == "snapshot":
        if for_repo(sys.argv[2]) is None:
            sys.exit(1)
        print(snapshot(sys.argv[2]))
    elif command == "enrich":
        config = json.load(sys.stdin)
        for model in config["models"]:
            if model["name"] in MODELS:
                model.update(MODELS[model["name"]])
                model.setdefault("tuning", "baseline; serving measurements pending")
        print(json.dumps(config))
    elif command == "model-dir":
        root = model_dir()
        if root is not None:
            print(root)
    elif command == "memory-bytes":
        print(memory_bytes(sys.argv[2]))
    elif command == "kv-bytes":
        found = for_repo(sys.argv[2])
        print(kv_cache_bytes(found[0]) if found else 0)
    elif command == "cache-root":
        print(cache_root(sys.argv[2]))
    elif command == "check-cache":
        if reason := cache_error(sys.argv[2]):
            raise SystemExit(reason)
    elif command == "mount":
        source, target, _ = serving_layout(sys.argv[2])
        print(f"{source}:{target}:ro")
    elif command == "serve-args":
        print(" ".join(serve_args(sys.argv[2])))
    elif command == "serve-path":
        print(serving_layout(sys.argv[2])[2])
    elif command == "contains":
        sys.exit(sys.argv[2] not in MODELS)
    elif command == "complete":
        from health import storage_error

        if storage_error(sys.argv[2]):
            sys.exit(1)
        found = next((model for model in MODELS.values()
                      if snapshot(model["repo"]) == Path(sys.argv[2])), None)
        sys.exit(not complete(sys.argv[2], checkpoint_files(found) if found else None))
    elif command == "check-storage":
        from health import storage_error

        reason = storage_error(sys.argv[2])
        if reason:
            raise SystemExit(reason)
    elif command == "check-runtime":
        if reason := runtime_error(sys.argv[2], sys.argv[3]):
            raise SystemExit(reason)
        if reason := ignored_files_error(sys.argv[2]):
            raise SystemExit(reason)
        if sys.argv[2] in MODELS and engine(MODELS[sys.argv[2]]) == "vllm":
            try:
                available = image_parsers(sys.argv[3])
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                raise SystemExit(f"Cannot list the parsers in {sys.argv[3]}: {type(exc).__name__}")
            if reason := parsers_error(sys.argv[2], available):
                raise SystemExit(reason)
    elif command == "check-moe-profile":
        if reason := moe_profile_error(sys.argv[2]):
            raise SystemExit(reason)
    elif command == "check-flags":
        if reason := flags_error(sys.argv[2], sys.argv[3:]):
            raise SystemExit(reason)
    elif command == "cache-namespace":
        print(cache_namespace(sys.argv[2]))
    else:
        raise SystemExit(f"unknown catalog command: {command}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
