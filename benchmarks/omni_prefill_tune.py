"""Sweep native Omni expert prefill tiles without changing serving settings.

The native WNA16 kernel already indexes every group32 scale within a K tile.
This experiment changes its configuration, not its dequantization or routing.
Run serialized with all other CPU/GPU timing work inside the serving image.
"""

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import statistics


def nearest_config(profile, batch):
    nearest = min((int(key) for key in profile if key != "triton_version"),
                  key=lambda size: abs(size - batch))
    return profile[str(nearest)]


def guarded_profile(profile, winners, limit=512):
    """Suggest changes only at measured steps; preserve every other step."""
    result = {key: dict(value) if isinstance(value, dict) else value
              for key, value in profile.items()}
    for batch, config in sorted(winners.items()):
        for neighbor in (batch - 1, batch + 1):
            result[str(neighbor)] = dict(nearest_config(profile, neighbor))
        result[str(batch)] = dict(config)
    for batch in range(1, limit + 1):
        expected = winners.get(batch, nearest_config(profile, batch))
        if nearest_config(result, batch) != expected:
            raise ValueError(f"Suggested profile would alter unmeasured step {batch}")
    return result


def paired_timings(torch, baseline, candidate, repeats, eviction):
    samples = {mode: {name: [] for name in ("baseline", "candidate")}
               for mode in ("warm", "evicted")}
    graphs = {"baseline": baseline, "candidate": candidate}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for repeat in range(repeats):
        names = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        modes = ("warm", "evicted") if repeat % 2 == 0 else ("evicted", "warm")
        for mode in modes:
            for name in names:
                if mode == "evicted":
                    # Stream ordering excludes eviction time from the events.
                    eviction.add_(1)
                else:
                    graphs[name].replay()
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[mode][name].append(start.elapsed_time(end))
    return {mode: {
        "baseline_ms": statistics.median(values["baseline"]),
        "candidate_ms": statistics.median(values["candidate"]),
        "speedup": statistics.median(values["baseline"]) / statistics.median(values["candidate"]),
        "samples_ms": values,
    } for mode, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional cached Omni snapshot; loads one expert layer only")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batches", type=int, nargs="+", default=(128, 512))
    parser.add_argument("--block-m", type=int, nargs="+", default=(16, 32))
    parser.add_argument("--block-n", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--block-k", type=int, nargs="+", default=(64, 128, 256))
    parser.add_argument("--num-warps", type=int, help="Otherwise retain the native profile setting")
    parser.add_argument("--num-stages", type=int, help="Otherwise retain the native profile setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-routing", action="store_true",
                        help="Every token selects the same experts, stressing expert reuse")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--evict-cache-mib", type=int, default=256)
    parser.add_argument("--min-speedup", type=float, default=1.05,
                        help="Required evicted gain before suggesting a profile entry")
    parser.add_argument("--profile", type=Path, help="Baseline profile; otherwise use installed current-device file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(batch not in (128, 512) for batch in args.batches):
        parser.error("This bounded experiment supports only 128- and 512-token prefill steps")
    if args.repeats < 3 or not 1 <= args.evict_cache_mib <= 256:
        parser.error("Use at least three repetitions and 1–256 MiB of eviction storage")
    if (any(v not in (16, 32) for v in args.block_m)
            or any(v not in (16, 32, 64) for v in args.block_n)
            or any(v not in (64, 128, 256) for v in args.block_k)):
        parser.error("Tiles exceed the bounded M16/32, N16/32/64, K64/128/256 sweep")
    if args.num_warps not in (None, 1, 2, 4, 8) or args.num_stages not in (None, 1, 2):
        parser.error("Use one/two stages and one/two/four/eight warps")
    if args.min_speedup < 1:
        parser.error("A suggested profile must improve evicted timing")

    import torch
    import vllm
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    from vllm.model_executor.layers.fused_moe import override_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from moe_vector_compare import checkpoint_tensors, reference, tensors

    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1151":
        raise SystemExit("This experiment is limited to the Radeon 8060S gfx1151")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    profile_path = args.profile or (Path(vllm.__file__).parent / "model_executor/layers/fused_moe/configs"
                                   / "E=128,N=768,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json")
    profile = json.loads(profile_path.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"at": datetime.now(timezone.utc).isoformat(),
              "model": "qwen3-omni-30b", "seed": args.seed, "real_weights": bool(args.checkpoint),
              "checkpoint": str(args.checkpoint) if args.checkpoint else None, "layer": args.layer,
              "profile": str(profile_path), "eviction_mib": args.evict_cache_mib,
              "shared_routing": args.shared_routing, "repeats": args.repeats, "cases": [],
              "software": {"torch": torch.__version__, "vllm": vllm.__version__},
              "hardware": torch.cuda.get_device_properties(0).name,
              "geometry": {"hidden": 2048, "intermediate": 768, "experts": 128,
                           "top_k": 8, "group_size": 32, "dtype": "bfloat16"},
              "scope": "One expert layer; native tiled kernels; synthetic activations/routing. No serving-model settings changed."}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    eviction = torch.empty(args.evict_cache_mib * 1024**2 // 4, dtype=torch.float32, device="cuda")
    eviction.zero_()
    winners = {}
    for batch in args.batches:
        values = (checkpoint_tensors(batch, 2048, 768, torch.bfloat16, args.checkpoint, args.layer, False)
                  if args.checkpoint else tensors(batch, 2048, 768, 128, 8, torch.bfloat16))
        x, w1, w2, s1, s2, weights, ids = values
        if args.shared_routing:
            ids.copy_(ids[:1].clone().expand_as(ids))
        quant = int4_w4a16_moe_quant_config(s1, s2, block_shape=[0, 32])
        baseline_config = dict(nearest_config(profile, batch))

        def call():
            return fused_experts(x, w1, w2, weights, ids,
                                 activation=MoEActivation.SILU, quant_config=quant)

        def capture(config):
            with override_config(config):
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = call()
                graph.replay()
                torch.cuda.synchronize()
            return graph, output

        baseline_graph, baseline_output = capture(baseline_config)
        if not torch.isfinite(baseline_output).all():
            raise RuntimeError("Native baseline produced non-finite outputs")
        # A small independent dequantized reference keeps the check bounded;
        # all candidate rows must additionally match the complete native result.
        independent = reference(x[:2], w1, w2, s1, s2, weights[:2], ids[:2], "silu")
        torch.testing.assert_close(baseline_output[:2], independent, rtol=.03, atol=.03)
        baseline_copy = baseline_output.clone()
        best = None
        for m, n, k in product(args.block_m, args.block_n, args.block_k):
            config = {**baseline_config, "BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k}
            if args.num_warps is not None:
                config["num_warps"] = args.num_warps
            if args.num_stages is not None:
                config["num_stages"] = args.num_stages
            row = {"batch": batch, "config": config, "baseline_config": baseline_config}
            report["cases"].append(row)
            graph, actual = None, None
            try:
                graph, actual = capture(config)
                if not torch.isfinite(actual).all():
                    raise AssertionError("Candidate produced non-finite outputs")
                torch.testing.assert_close(actual, baseline_copy, rtol=.03, atol=.03)
                torch.testing.assert_close(actual[:2], independent, rtol=.03, atol=.03)
                row["max_abs_difference_from_native"] = float((actual.float() - baseline_copy.float()).abs().max())
                row["allocated_mib"] = torch.cuda.memory_allocated() / 1024**2
                if row["allocated_mib"] > 1024:
                    row["status"] = "allocation_budget_exceeded"
                    raise RuntimeError("Experiment exceeded its 1 GiB tensor allocation budget")
                row["timings"] = paired_timings(torch, baseline_graph, graph, args.repeats, eviction)
                row["status"] = "passed"
                gain = row["timings"]["evicted"]["speedup"]
                if (config != baseline_config and gain >= args.min_speedup
                        and row["timings"]["warm"]["speedup"] >= 1
                        and (best is None or gain > best[0])):
                    best = gain, dict(config)
            except (CompilationError, OutOfResources, AssertionError) as exc:
                row.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
            finally:
                del graph, actual
                save()
            print(json.dumps({key: value for key, value in row.items() if key != "timings"}
                             | ({"timings": {mode: {key: value for key, value in timing.items() if key != "samples_ms"}
                                              for mode, timing in row["timings"].items()}}
                                if "timings" in row else {})), flush=True)
        if best is not None:
            winners[batch] = best[1]
        del baseline_graph, baseline_output, baseline_copy, independent, values
        del x, w1, w2, s1, s2, weights, ids, quant
        torch.cuda.synchronize()
    report["suggested_profile"] = guarded_profile(profile, winners) if winners else None
    report["requires_serving_validation"] = True
    save()
    print(json.dumps({"suggested_batches": sorted(winners), "requires_serving_validation": True}), flush=True)


if __name__ == "__main__":
    main()
