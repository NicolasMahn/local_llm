"""Compare one real dense INT4 projection without loading a model instance.

Run inside the Strix Halo image. Packing uses the installed vLLM conversion
helpers; the reference independently expands the original checkpoint nibbles.
This script never registers a serving hook or changes a checkpoint.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm_plugins"))
from local_llm_dense import splitk_w4a16_gemm
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as native
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import permute_param_layout_
from vllm.scalar_type import scalar_types

from dense_compare import check_reference, dequantize


MODELS = {
    "granite-4.2-30b": {"architecture": "GraniteForCausalLM", "method": "gptq",
                         "hidden": 4096, "intermediate": 32768, "group": 128},
    "olmo-3.1-32b-think": {"architecture": "Olmo3ForCausalLM", "method": "compressed-tensors",
                           "hidden": 5120, "intermediate": 27648, "group": 32},
}
CANDIDATE = {"split_k": 1, "block_m": 64, "block_n": 64, "block_k": 128,
             "num_stages": 1, "num_warps": 4}


def emit(value):
    print(json.dumps(value), flush=True)


def quantization(checkpoint, model):
    expected = MODELS[model]
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("architectures") != [expected["architecture"]]:
        raise ValueError("Checkpoint architecture does not match the selected model generation")
    if (config.get("hidden_size"), config.get("intermediate_size")) != (
            expected["hidden"], expected["intermediate"]):
        raise ValueError("Unexpected model projection dimensions")
    quant = config["quantization_config"]
    if quant.get("quant_method") != expected["method"]:
        raise ValueError("Unsupported checkpoint quantization format")
    if expected["method"] == "gptq":
        supported = (quant.get("bits") == 4 and quant.get("sym") is True
                     and quant.get("desc_act") is False
                     and quant.get("group_size") == expected["group"])
    else:
        groups = list(quant.get("config_groups", {}).values())
        if len(groups) != 1:
            raise ValueError("The probe requires one uniform compressed-tensors group")
        weights = groups[0]["weights"]
        supported = (quant.get("format") == "pack-quantized"
                     and weights.get("num_bits") == 4 and weights.get("symmetric") is True
                     and weights.get("group_size") == expected["group"]
                     and weights.get("actorder") is None
                     and weights.get("type") == "int")
    if not supported:
        raise ValueError("Only the measured symmetric INT4 layout without activation reordering is supported")
    return expected


class Checkpoint:
    def __init__(self, path):
        self.path = path
        index = path / "model.safetensors.index.json"
        self.weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
        self.loaded = []

    def contains(self, key):
        if self.weight_map is not None:
            return key in self.weight_map
        with safe_open(self.path / "model.safetensors", framework="pt", device="cpu") as source:
            return key in source.keys()

    def tensor(self, key):
        filename = self.weight_map[key] if self.weight_map is not None else "model.safetensors"
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe checkpoint shard path for {key}")
        shard = self.path / filename
        if not shard.is_file():
            raise ValueError(f"Checkpoint shard is not fully downloaded: {filename}")
        with safe_open(shard, framework="pt", device="cpu") as source:
            tensor = source.get_tensor(key)
        self.loaded.append({"name": key, "shard": filename, "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype)})
        return tensor


def native_layout(raw, scales, method):
    """The same unpack, shuffle and scale permutation used by the native layer."""
    packed_dim = 0 if method == "gptq" else 1
    unpacked = unpack_quantized_values_into_int32(raw, scalar_types.uint4b8, packed_dim=packed_dim)
    if method == "gptq":
        unpacked = unpacked.t().contiguous()
    packed = native.pack_int4_exllama_shuffle(unpacked).contiguous()
    parameter = torch.nn.Parameter(scales, requires_grad=False)
    # The helpers need layout metadata, not a distributed linear layer.
    parameter.input_dim = 0 if method == "gptq" else 1
    parameter.output_dim = 1 if method == "gptq" else 0
    permute_param_layout_(parameter, input_dim=1, output_dim=0)
    return packed, parameter.data.contiguous()


def source_dequantize(raw, scales, method, group, columns):
    """Independently decode selected output rows directly from stored nibbles."""
    shifts = torch.arange(8, device="cpu", dtype=torch.int64) * 4
    if method == "gptq":
        words = raw[:, columns].t().contiguous()
        selected_scales = scales[:, columns].t().contiguous()
    else:
        words = raw[columns].contiguous()
        selected_scales = scales[columns].contiguous()
    values = ((words.to(torch.int64)[..., None] >> shifts) & 15).flatten(1)
    return ((values - 8).to(selected_scales.dtype)
            * selected_scales.repeat_interleave(group, dim=1)).to(selected_scales.dtype)


def load_projection(checkpoint, model, projection, layer, reference_rows):
    expected = quantization(checkpoint, model)
    method, group = expected["method"], expected["group"]
    parts = ("gate_proj", "up_proj") if projection == "gate" else ("down_proj",)
    inputs = expected["hidden"] if projection == "gate" else expected["intermediate"]
    outputs = expected["intermediate"] if projection == "gate" else expected["hidden"]
    source = Checkpoint(checkpoint)
    packed_parts, scale_parts, reference_parts = [], [], []
    reference_columns = []
    for part_index, part in enumerate(parts):
        prefix = f"model.layers.{layer}.mlp.{part}"
        weight_name = "qweight" if method == "gptq" else "weight_packed"
        scale_name = "scales" if method == "gptq" else "weight_scale"
        raw, scales = source.tensor(f"{prefix}.{weight_name}"), source.tensor(f"{prefix}.{scale_name}")
        expected_raw = (inputs // 8, outputs) if method == "gptq" else (outputs, inputs // 8)
        expected_scales = (inputs // group, outputs) if method == "gptq" else (outputs, inputs // group)
        if raw.dtype != torch.int32 or tuple(raw.shape) != expected_raw:
            raise ValueError(f"Unexpected packed layout for {prefix}: {raw.dtype} {tuple(raw.shape)}")
        allowed_scale_types = (torch.float16, torch.bfloat16) if method == "gptq" else (torch.bfloat16,)
        if scales.dtype not in allowed_scale_types or tuple(scales.shape) != expected_scales:
            raise ValueError(f"Unexpected per-group scales for {prefix}: {scales.dtype} {tuple(scales.shape)}")
        # GPTQ parameters are allocated in the serving activation dtype. Granite's
        # stored FP16 scales are converted by the BF16 serving loader too.
        scales = scales.to(torch.bfloat16)
        if not torch.isfinite(scales).all() or not (scales != 0).any():
            raise ValueError(f"Invalid scales in {prefix}")
        if source.contains(f"{prefix}.bias"):
            raise ValueError("The measured projections have no bias; refusing an unexpected checkpoint")
        if method == "gptq" and source.contains(f"{prefix}.g_idx"):
            g_idx = source.tensor(f"{prefix}.g_idx")
            if not torch.equal(g_idx, torch.arange(inputs, dtype=g_idx.dtype) // group):
                raise ValueError(f"Unexpected GPTQ group reordering in {prefix}")
        columns = torch.linspace(0, outputs - 1, min(outputs, reference_rows), dtype=torch.float64).round().long()
        independent = source_dequantize(raw, scales, method, group, columns)
        packed, converted_scales = native_layout(raw, scales, method)
        # This comparison validates the installed conversion against the stored
        # layout before either matrix implementation can share a packing error.
        torch.testing.assert_close(dequantize(packed[columns], converted_scales[columns], group,
                                             scales.dtype), independent, rtol=0, atol=0)
        packed_parts.append(packed)
        scale_parts.append(converted_scales)
        reference_parts.append(independent)
        reference_columns.extend((columns + part_index * outputs).tolist())
    packed = torch.cat(packed_parts, dim=0).contiguous() if len(parts) > 1 else packed_parts[0]
    scales = torch.cat(scale_parts, dim=0).contiguous() if len(parts) > 1 else scale_parts[0]
    reference = torch.cat(reference_parts, dim=0)
    return packed, scales, reference, reference_columns, {
        "model": model, "projection": projection, "layer": layer, "group_size": group,
        "k": inputs, "n": outputs * len(parts), "source_method": method,
        "loaded_tensors": source.loaded, "independent_output_columns": reference_columns,
        "scale_min": scales.min().item(), "scale_max": scales.max().item(),
        "packing_reference_bitwise_equal": True,
        "conversion_helpers": ["unpack_quantized_values_into_int32", "pack_int4_exllama_shuffle",
                               "permute_param_layout_"],
    }


def graph_for(call, operations):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(operations):
            output = call()
    # Capture records work without executing it. Initialise the captured output
    # before comparing it, including the single-operation eviction graph.
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def graph_ms(graph, operations, eviction=None):
    samples = []
    for _ in range(7):
        if eviction is not None:
            # Queued before the start event, so eviction is excluded from timing.
            eviction.add_(1)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / operations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "operations_per_replay": operations}


def cpu_self_test():
    for method, group in (("gptq", 128), ("compressed-tensors", 32)):
        n, k = 10, 256
        values = (torch.arange(n * k, dtype=torch.int64).reshape(n, k) * 7 + 3) % 16
        words = torch.zeros((n, k // 8), dtype=torch.int64)
        for i in range(8):
            words |= values[:, i::8] << (4 * i)
        raw = words.to(torch.int32)
        scales = (0.01 + torch.arange(n * (k // group)).reshape(n, k // group) * 0.0001).to(torch.bfloat16)
        if method == "gptq":
            raw, scales = raw.t().contiguous(), scales.t().contiguous()
        converted, converted_scales = native_layout(raw, scales, method)
        columns = torch.arange(n)
        independent = source_dequantize(raw, scales, method, group, columns)
        decoded = dequantize(converted, converted_scales, group, torch.bfloat16)
        torch.testing.assert_close(decoded, independent, rtol=0, atol=0)
        original_scales = scales.t() if method == "gptq" else scales
        direct = ((values - 8).to(torch.bfloat16)
                  * original_scales.repeat_interleave(group, 1)).to(torch.bfloat16)
        torch.testing.assert_close(decoded, direct, rtol=0, atol=0)
    emit({"cpu_packing_self_test": "passed", "formats": ["gptq", "compressed-tensors"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--projection", choices=("gate", "down"), default="down",
                        help="gate loads gate and up together, as vLLM's fused GLU projection")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 128, 512])
    parser.add_argument("--reference-rows", type=int, default=64, help="Independent output rows checked per source projection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64), default=CANDIDATE["block_m"],
                        help="Row tile of the candidate; small steps were measured with 16")
    parser.add_argument("--evict-cache-mib", type=int, default=0,
                        help="Also time single-operation graphs after this many MiB of GPU cache eviction")
    parser.add_argument("--inspect-only", action="store_true", help="CPU-only checkpoint loading and packing verification")
    parser.add_argument("--self-test", action="store_true", help="CPU-only known-value tests, no checkpoint needed")
    parser.add_argument("--keep-going", action="store_true", help="Record numerical rejections without timing them")
    args = parser.parse_args()
    if args.self_test:
        cpu_self_test()
        return
    if args.checkpoint is None or args.model is None:
        parser.error("--checkpoint and --model are required unless using --self-test")
    if (any(not 1 <= m <= 512 for m in args.batches) or args.reference_rows < 1
            or args.layer < 0 or not 0 <= args.evict_cache_mib <= 1024):
        parser.error("Use batches 1–512, positive reference rows, nonnegative layer, and at most 1024 MiB eviction")
    torch.set_num_threads(1)
    packed, scales, independent, columns, metadata = load_projection(
        args.checkpoint, args.model, args.projection, args.layer, args.reference_rows)
    emit({"checkpoint": str(args.checkpoint), "packing": metadata})
    if args.inspect_only:
        return
    device = torch.cuda.get_device_properties(0)
    if device.gcnArchName.split(":")[0] != "gfx1151":
        raise ValueError("These measurements are restricted to the validated gfx1151 machine")
    torch.manual_seed(args.seed)
    packed, scales = packed.cuda(), scales.cuda()
    independent = independent.cuda()
    columns = torch.tensor(columns, device="cuda", dtype=torch.long)
    eviction = (torch.zeros(args.evict_cache_mib * 1024**2, device="cuda", dtype=torch.uint8)
                if args.evict_cache_mib else None)
    cu_count = device.multi_processor_count
    group = metadata["group_size"]
    rejected = 0
    candidate_config = CANDIDATE | {"block_m": args.block_m}
    for batch in args.batches:
        x = torch.randn((batch, metadata["k"]), device="cuda", dtype=torch.bfloat16)
        timing_input = x.clone()
        changed_inputs = [torch.randn_like(x) for _ in range(2)]

        def stock():
            return native._rdna_hybrid_w4a16_apply_impl(
                x, packed.view(torch.int8), scales, None, None, cu_count, group)

        def candidate():
            return splitk_w4a16_gemm(x, packed, scales, group, **candidate_config)

        baseline = stock()
        reference = x.float() @ independent.float().t()
        baseline_error = check_reference(baseline[:, columns], reference)
        choices = [("stock", stock), ("candidate", candidate)]
        if args.reverse:
            choices.reverse()
        for name, call in choices:
            try:
                x.copy_(timing_input)
                output = call()
                torch.testing.assert_close(output, baseline, rtol=0.01, atol=0.005)
                errors = check_reference(output[:, columns], reference)
                graph, captured = graph_for(call, 10)
                # A captured pointer must read new activations, not a frozen
                # result from the first request or warmup.
                changed_errors = []
                for changed in changed_inputs:
                    x.copy_(changed)
                    graph.replay()
                    native_changed = stock()
                    torch.testing.assert_close(captured, native_changed, rtol=0.01, atol=0.005)
                    changed_errors.append(check_reference(captured[:, columns], x.float() @ independent.float().t()))
                x.copy_(timing_input)
                warm = graph_ms(graph, 10)
                cold = None
                if eviction is not None:
                    single_graph, single_output = graph_for(call, 1)
                    torch.testing.assert_close(single_output, stock(), rtol=0.01, atol=0.005)
                    cold = graph_ms(single_graph, 1, eviction)
                emit({"model": args.model, "projection": args.projection, "layer": args.layer,
                      "batch": batch, "seed": args.seed, "kernel": name, "accepted": True,
                      "config": candidate_config if name == "candidate" else {}, "warm": warm,
                      "evicted": cold, "evict_cache_mib": args.evict_cache_mib,
                      "reference_error": errors, "native_reference_error": baseline_error,
                      "changed_activation_errors": changed_errors, "graph_replay_checked": True})
            except AssertionError as exc:
                if not args.keep_going:
                    raise
                rejected += 1
                emit({"model": args.model, "projection": args.projection, "batch": batch,
                      "kernel": name, "accepted": False, "rejection": str(exc)})
    emit({"completed": True, "rejected_candidates": rejected,
          "serving_hooks_registered": False, "checkpoint_changed": False})


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc))
