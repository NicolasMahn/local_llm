"""Compare native and host-metadata Omni audio on one real encoder tower.

Run inside the model image with vllm_plugins on PYTHONPATH. This loads only
the audio tower, not a second language model or KV cache. Native timing keeps
the existing once-per-stack host attention bound improvement enabled.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from local_llm_audio import _prepare_audio, audio_layout


def checkpoint_weights(directory):
    from safetensors import safe_open

    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "thinker.audio_tower."
    with ExitStack() as stack:
        opened = {}
        for name, shard in index.items():
            if not name.startswith(prefix):
                continue
            if shard not in opened:
                opened[shard] = stack.enter_context(safe_open(directory / shard, framework="pt", device="cpu"))
            yield name.removeprefix(prefix), opened[shard].get_tensor(name)


def elapsed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000, result


def synchronization_ops(call):
    # CPU activity records scalar reads and dynamic boolean packing without
    # enabling a separate GPU tracing backend or changing attention kernels.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        call()
        torch.cuda.synchronize()
    watched = {"aten::_local_scalar_dense", "aten::nonzero", "aten::item"}
    return {event.key: event.count for event in profile.key_averages() if event.key in watched}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", default=["100", "801", "3107", "100,101,399"],
                        help="Mel-frame lengths; commas put recordings in one batch")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Count CPU-side scalar and dynamic-index operations")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least three alternating repetitions")
    cases = [tuple(int(n) for n in value.split(",")) for value in args.lengths]
    for case in cases:
        audio_layout(case)
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1151":
        raise SystemExit("This experiment targets gfx1151")

    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeConditionalGenerationMixin,
    )
    from vllm.utils.network_utils import get_open_port
    from local_llm_encoder import register as register_host_bound

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    config = json.loads((args.checkpoint / "config.json").read_text())["thinker_config"]["audio_config"]
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}", backend="gloo")
        initialize_model_parallel(backend="gloo")
        register_host_bound()
        tower = Qwen3OmniMoeAudioEncoder(Qwen3OmniMoeAudioEncoderConfig(**config)).eval()
        loaded = tower.load_weights(checkpoint_weights(args.checkpoint))
        expected = {name for name, _ in tower.named_parameters()}
        if expected - loaded:
            raise RuntimeError(f"Audio weights were not loaded: {sorted(expected - loaded)}")
        # Serving restores the floating default after constructing BF16
        # weights. Integer length division must use its native FP32 default.
        torch.set_default_dtype(torch.float32)
        native = Qwen3OmniMoeConditionalGenerationMixin._process_audio_input
        if getattr(native, "_local_llm_host_audio", False):
            native = native.__wrapped__
        optimized = _prepare_audio(native, Qwen3OmniMoeAudioEncoder)
        owner = SimpleNamespace(audio_tower=tower)
        print(json.dumps({"dtype": str(tower.dtype), "weights": "real audio tower", "seed": args.seed,
                          "native_host_attention_bound": True, "loaded_parameters": len(loaded)}), flush=True)
        for lengths in cases:
            features = torch.randn((128, sum(lengths)), dtype=torch.float32, device="cuda") * 0.5
            data = {"input_features": features,
                    "audio_feature_lengths": torch.tensor(lengths, dtype=torch.int64, device="cpu")}
            calls = {"native": lambda: native(owner, data), "host_metadata": lambda: optimized(owner, data)}
            for _ in range(2):
                for call in calls.values():
                    call()
            reference, candidate = calls["native"](), calls["host_metadata"]()
            torch.cuda.synchronize()
            assert len(reference) == len(candidate) == len(lengths)
            for expected_output, actual in zip(reference, candidate):
                assert torch.isfinite(expected_output).all() and torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected_output, rtol=0, atol=0)
            samples = {key: [] for key in calls}
            for repeat in range(args.repeats):
                order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
                for key in order:
                    milliseconds, _ = elapsed(calls[key])
                    samples[key].append(milliseconds)
            result = {"feature_lengths": lengths, "output_lengths": audio_layout(lengths).output_lengths,
                      "bitwise_equal": True, "median_ms": {k: statistics.median(v) for k, v in samples.items()},
                      "samples_ms": samples}
            if args.profile:
                result["synchronization_ops"] = {key: synchronization_ops(call) for key, call in calls.items()}
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
