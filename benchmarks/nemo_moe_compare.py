"""Bounded MoE experiments for Nemotron 3.5 Lightning on gfx1151.

Tile candidates compare vLLM configurations of the native kernel. Vector
candidates (--vector) compare the local single-token kernels in
vllm_plugins/local_llm_moe.py against the native path instead.
It loads at most one expert layer, never installs a tuning profile, and records
rejected candidates without using them for timing or serving.
"""

import argparse
from contextlib import ExitStack, contextmanager
import inspect
import json
from pathlib import Path
import statistics


MODEL = "nemotron-3.5-lightning-30b-a3b"
EXPERTS, TOPK, HIDDEN, INTERMEDIATE, GROUP = 128, 6, 2688, 1856, 64
ROUTED_SCALE = 2.5
TOLERANCE = {"rtol": 0.03, "atol": 0.03}
RELATIVE_L2_LIMIT = 0.01


def candidates():
    tiles = [(16, 32, 64), (16, 64, 128), (16, 64, 256),
             (32, 32, 128), (32, 64, 128), (32, 64, 256),
             (64, 32, 64), (64, 32, 128)]
    return [(f"m{m}-n{n}-k{k}", {"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n,
             "BLOCK_SIZE_K": k, "GROUP_SIZE_M": 1, "SPLIT_K": 1,
             "num_warps": 4, "num_stages": 1}) for m, n, k in tiles]


def vector_candidates():
    return [(f"vector-r{rows}-w{warps}", {"rows": rows, "warps": warps})
            for rows, warps in ((1, 4), (2, 2), (2, 4), (2, 8), (4, 4), (4, 8), (8, 4))]


def validate_checkpoint(config, layer):
    expected = {"hidden_size": HIDDEN, "moe_intermediate_size": INTERMEDIATE,
                "n_routed_experts": EXPERTS, "num_experts_per_tok": TOPK,
                "mlp_hidden_act": "relu2", "routed_scaling_factor": ROUTED_SCALE}
    if (config.get("architectures") != ["NemotronHForCausalLM"]
            or any(config.get(key) != value for key, value in expected.items())):
        raise ValueError("Checkpoint does not match the Nemotron 3.5 Lightning expert contract")
    if config.get("layers_block_type", [])[layer] != "moe":
        raise ValueError(f"Layer {layer} is not an MoE layer")
    quant = config.get("quantization_config", {})
    groups = list(quant.get("config_groups", {}).values())
    if quant.get("quant_method") != "compressed-tensors" or len(groups) != 1:
        raise ValueError("Expected the selected compressed-tensors checkpoint")
    weights = groups[0]["weights"]
    if not (weights.get("num_bits") == 4 and weights.get("symmetric") is True
            and weights.get("group_size") == GROUP and weights.get("type") == "int"
            and weights.get("actorder") in (None, "static")
            and quant.get("transform_config") in (None, {})):
        raise ValueError("Expected symmetric INT4 group64 without dynamic group reordering")


def dequantize(packed, scales, group=GROUP):
    import torch

    # Native expert buffers pack adjacent K values into low/high nibbles;
    # this is not the ExLlama layout used by dense RDNA matrices.
    values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).float() - 8
    return (values * scales.float().repeat_interleave(group, dim=-1)).to(scales.dtype)


def reference(x, w1, w2, s1, s2, routing, ids, group=GROUP):
    """Sparse FP32 arithmetic with the native BF16 intermediate boundaries."""
    import torch
    import torch.nn.functional as functional

    result = torch.empty_like(x)
    for token in range(x.shape[0]):
        parts = []
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            up = functional.linear(x[token].float(), dequantize(w1[expert], s1[expert], group).float()).to(x.dtype)
            activated = up.float().clamp_min(0).square().to(x.dtype)
            down = functional.linear(activated.float(), dequantize(w2[expert], s2[expert], group).float())
            parts.append((down * routing[token, slot]).to(x.dtype).float())
        result[token] = torch.stack(parts).sum(0).to(x.dtype)
    return result


def check(actual, expected):
    import torch

    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Non-finite expert output")
    torch.testing.assert_close(actual, expected, **TOLERANCE)
    delta = actual.float() - expected.float()
    relative_l2 = (delta.square().sum() / expected.float().square().sum().clamp_min(1e-30)).sqrt().item()
    if relative_l2 > RELATIVE_L2_LIMIT:
        raise AssertionError(f"Expert relative L2 error {relative_l2} exceeds {RELATIVE_L2_LIMIT}")
    return {"max_abs_error": delta.abs().max().item(),
            "relative_l2_error": relative_l2, "relative_l2_limit": RELATIVE_L2_LIMIT}


def routing_inputs(torch, batch, shared=False):
    x = torch.randn((batch, HIDDEN), device="cuda", dtype=torch.bfloat16)
    scores = torch.randn((batch, EXPERTS), device="cuda", dtype=torch.float32)
    values, ids = scores.topk(TOPK)
    if shared:
        ids[1:].copy_(ids[:1].expand_as(ids[1:]))
    return x, (values.softmax(-1) * ROUTED_SCALE).contiguous(), ids.to(torch.int32).contiguous()


@contextmanager
def checkpoint_reader(checkpoint, opener=None):
    if opener is None:
        from safetensors import safe_open

        opener = safe_open
    index_file = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())["weight_map"] if index_file.is_file() else None
    if index is not None and not isinstance(index, dict):
        raise ValueError("Invalid checkpoint weight map")
    with ExitStack() as stack:
        opened = {}

        def tensor(name):
            filename = index[name] if index is not None else "model.safetensors"
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Unsafe shard path in checkpoint index")
            if filename not in opened:
                shard = checkpoint / filename
                if not shard.is_file():
                    raise ValueError(f"Checkpoint file is not fully downloaded: {filename}")
                opened[filename] = stack.enter_context(opener(shard, framework="pt", device="cpu"))
            return opened[filename].get_tensor(name)

        yield tensor


def weights(torch, checkpoint, layer):
    dtype = torch.bfloat16
    shapes = [(EXPERTS, INTERMEDIATE, HIDDEN // 2), (EXPERTS, HIDDEN, INTERMEDIATE // 2),
              (EXPERTS, INTERMEDIATE, HIDDEN // GROUP), (EXPERTS, HIDDEN, INTERMEDIATE // GROUP)]
    if checkpoint is None:
        w1, w2 = [torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8) for shape in shapes[:2]]
        s1, s2 = [(0.003 + torch.rand(shape, device="cuda") * 0.015).to(dtype) for shape in shapes[2:]]
        return w1, w2, s1, s2
    validate_checkpoint(json.loads((checkpoint / "config.json").read_text()), layer)
    tensors = [torch.empty(shape, dtype=torch.uint8 if i < 2 else dtype, device="cpu")
               for i, shape in enumerate(shapes)]
    w1, w2, s1, s2 = tensors
    with checkpoint_reader(checkpoint) as tensor:
        for expert in range(EXPERTS):
            prefix = f"backbone.layers.{layer}.mixer.experts.{expert}."
            for name, packed, scale in (("up_proj", w1[expert], s1[expert]), ("down_proj", w2[expert], s2[expert])):
                raw, raw_scale = tensor(prefix + name + ".weight_packed"), tensor(prefix + name + ".weight_scale")
                if (raw.dtype != torch.int32 or raw_scale.dtype != dtype
                        or raw.view(torch.uint8).shape != packed.shape or raw_scale.shape != scale.shape):
                    raise ValueError(f"Unexpected native expert layout: {prefix}{name}")
                packed.copy_(raw.view(torch.uint8))
                scale.copy_(raw_scale)
    if any(not torch.isfinite(scale).all() or (scale < 0).any() for scale in (s1, s2)):
        raise ValueError("Non-finite or negative checkpoint scales")
    return tuple(tensor.cuda() for tensor in tensors)


def capture(torch, override, call, config):
    with override(config):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
    graph.replay()
    return graph, output


def timings(torch, baseline, candidate, repeats, eviction, reverse):
    samples = {mode: {name: [] for name in ("baseline", "candidate")} for mode in ("warm", "evicted")}
    graphs = {"baseline": baseline, "candidate": candidate}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for repeat in range(repeats):
        order = ("candidate", "baseline") if bool(repeat % 2) != reverse else ("baseline", "candidate")
        for mode in (("warm", "evicted") if repeat % 2 == 0 else ("evicted", "warm")):
            for name in order:
                if mode == "evicted":
                    eviction.add_(1)
                else:
                    graphs[name].replay()
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[mode][name].append(start.elapsed_time(end))
    return {mode: {"baseline_ms": statistics.median(values["baseline"]),
                   "candidate_ms": statistics.median(values["candidate"]),
                   "speedup": statistics.median(values["baseline"]) / statistics.median(values["candidate"]),
                   "samples_ms": values} for mode, values in samples.items()}


def cpu_self_test():
    import torch

    # One positive and one negative expert distinguish ReLU squared from a
    # gated activation and prove that routing is applied after projection.
    hidden, intermediate = 128, 64
    x = torch.ones((1, hidden), dtype=torch.bfloat16)
    w1 = torch.stack([torch.full((intermediate, hidden // 2), n, dtype=torch.uint8) for n in (0x99, 0x77)])
    w2 = torch.full((2, hidden, intermediate // 2), 0x99, dtype=torch.uint8)
    s1 = torch.full((2, intermediate, hidden // GROUP), 0.125, dtype=x.dtype)
    s2 = torch.full((2, hidden, intermediate // GROUP), 0.125, dtype=x.dtype)
    output = reference(x, w1, w2, s1, s2, torch.tensor([[0.5, 0.5]]), torch.tensor([[0, 1]]))
    torch.testing.assert_close(output, torch.full_like(x, 1024), rtol=0, atol=0)
    scales = torch.tensor([[0.125, 0.25, 0.5]], dtype=x.dtype)
    decoded = dequantize(torch.full((1, 96), 0xA9, dtype=torch.uint8), scales)
    assert decoded.shape == (1, 192)
    torch.testing.assert_close(decoded[0, [0, 1, 64, 65, 128, 129]],
                               torch.tensor([0.125, 0.25, 0.25, 0.5, 0.5, 1], dtype=x.dtype), rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional selected Nemotron snapshot; load one real MoE layer")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 128, 512])
    parser.add_argument("--candidate", action="append",
                        choices=[name for name, _ in (*candidates(), *vector_candidates())])
    parser.add_argument("--vector", action="store_true",
                        help="Compare the local vector kernels instead of native tile configurations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--shared-routing", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--reference-tokens", type=int, default=2)
    parser.add_argument("--evict-cache-mib", type=int, default=256)
    parser.add_argument("--max-allocation-mib", type=int, default=1536)
    parser.add_argument("--output", type=Path, help="Write clean JSONL as well as stdout")
    parser.add_argument("--self-test", action="store_true", help="CPU-only known-value quantization/activation checks")
    args = parser.parse_args()
    if args.self_test:
        cpu_self_test()
        print(json.dumps({"cpu_reference_self_test": "passed"}))
        return
    if (any(not 1 <= batch <= 512 for batch in args.batches) or args.repeats < 3
            or not 1 <= args.reference_tokens <= 4 or not 1 <= args.evict_cache_mib <= 256
            or args.layer < 0 or not 1024 <= args.max_allocation_mib <= 2048):
        parser.error("Use batches 1–512, repeats >=3, reference tokens 1–4, eviction 1–256 MiB, allocation 1024–2048 MiB")
    import torch
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    import vllm
    from vllm.model_executor.layers.fused_moe import override_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation_supported
    from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, try_get_optimal_moe_config

    activation = getattr(MoEActivation, "RELU2_NO_MUL", None)
    if (activation is None or activation.is_gated or not apply_moe_activation_supported(activation)
            or not {"activation", "quant_config"} <= inspect.signature(fused_experts).parameters.keys()):
        raise SystemExit("Installed native fused_experts does not support the required non-gated ReLU squared interface")
    device = torch.cuda.get_device_properties(0)
    if device.gcnArchName.split(":")[0] != "gfx1151":
        raise SystemExit("This bounded experiment requires gfx1151")
    if torch.cuda.mem_get_info()[0] < 3 * 1024**3:
        raise SystemExit("At least 3 GiB free GPU memory is required before allocating experiment tensors")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open("w") if args.output else None

    def emit(value):
        line = json.dumps(value)
        print(line, flush=True)
        if stream:
            stream.write(line + "\n")
            stream.flush()

    try:
        w1, w2, s1, s2 = weights(torch, args.checkpoint, args.layer)
        quant = int4_w4a16_moe_quant_config(s1, s2, block_shape=[0, GROUP])
        eviction = torch.zeros(args.evict_cache_mib * 1024**2, device="cuda", dtype=torch.uint8)
        emit({"model": MODEL, "vllm": vllm.__version__, "torch": torch.__version__,
              "device": device.name, "architecture": device.gcnArchName, "experts": EXPERTS,
              "topk": TOPK, "hidden": HIDDEN, "intermediate": INTERMEDIATE, "group_size": GROUP,
              "activation": activation.value, "precision": "bfloat16", "gated": False,
              "checkpoint": str(args.checkpoint) if args.checkpoint else None, "layer": args.layer,
              "seed": args.seed, "shared_routing": args.shared_routing, "reverse": args.reverse,
              "tolerance": TOLERANCE, "relative_l2_limit": RELATIVE_L2_LIMIT,
              "evict_cache_mib": args.evict_cache_mib,
              "profile_installed": False, "serving_validation_required": True})
        if args.vector:
            from local_llm_moe import vector_experts
        selected = [(name, config) for name, config in (vector_candidates() if args.vector else candidates())
                    if not args.candidate or name in args.candidate]
        if args.reverse:
            selected.reverse()
        for batch in args.batches:
            x, routing, ids = routing_inputs(torch, batch, args.shared_routing)
            original = (x.clone(), routing.clone(), ids.clone())
            changed = [routing_inputs(torch, batch, args.shared_routing) for _ in range(2)]

            def restore(values):
                for target, value in zip((x, routing, ids), values):
                    target.copy_(value)

            def call():
                return fused_experts(x, w1, w2, routing, ids, activation=activation, quant_config=quant)

            baseline_config = dict(try_get_optimal_moe_config(w1.shape, w2.shape, TOPK,
                                   "int4_w4a16", batch, block_shape=[0, GROUP]))
            baseline_graph, baseline_output = capture(torch, override_config, call, baseline_config)
            baseline_copy = baseline_output.clone()
            tokens = min(batch, args.reference_tokens)
            independent = reference(x[:tokens], w1, w2, s1, s2, routing[:tokens], ids[:tokens])
            baseline_error = check(baseline_copy[:tokens], independent)
            emit({"batch": batch, "status": "baseline_checked", "baseline_config": baseline_config,
                  "independent_reference_tokens": tokens, "reference_error": baseline_error})
            for name, config in selected:
                graph, output = None, None
                row = {"model": MODEL, "batch": batch, "candidate": name,
                       "config": config, "baseline_config": baseline_config}
                try:
                    restore(original)
                    if args.vector:
                        def vector(config=config):
                            return vector_experts(x, w1, w2, s1, s2, routing, ids,
                                                  activation=activation.value, group=GROUP, **config)

                        graph, output = capture(torch, override_config, vector, baseline_config)
                    else:
                        graph, output = capture(torch, override_config, call, config)
                    row["native_error"] = check(output, baseline_copy)
                    row["reference_error"] = check(output[:tokens], independent)
                    graph_checks = []
                    for values in changed:
                        restore(values)
                        graph.replay()
                        baseline_graph.replay()
                        errors = check(output, baseline_output)
                        checked = reference(x[:tokens], w1, w2, s1, s2, routing[:tokens], ids[:tokens])
                        errors["independent"] = check(output[:tokens], checked)
                        graph_checks.append(errors)
                    restore(original)
                    row["changed_input_routing_graph_checks"] = graph_checks
                    row["allocated_mib"] = torch.cuda.memory_allocated() / 1024**2
                    if row["allocated_mib"] > args.max_allocation_mib:
                        row["status"] = "allocation_budget_exceeded"
                        raise RuntimeError("Experiment tensor allocation exceeded the configured budget")
                    row["timings"] = timings(torch, baseline_graph, graph, args.repeats, eviction, args.reverse)
                    row["status"] = "passed"
                except (AssertionError, CompilationError, OutOfResources) as exc:
                    row.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
                finally:
                    emit(row)
                    del graph, output
                    restore(original)
            del baseline_graph, baseline_output, baseline_copy, independent, x, routing, ids, original, changed
            torch.cuda.synchronize()
        emit({"completed": True, "profile_installed": False, "serving_validation_required": True})
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise SystemExit(str(exc))
