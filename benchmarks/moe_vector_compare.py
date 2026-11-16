"""Check fused vector experts against independent dequantization and vLLM."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import vllm
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from local_llm_moe import register, vector_config, vector_experts
from moe_compare import measure


def measure_after_eviction(call, cache_mib):
    """Time graph replay after eviction, excluding the eviction kernel itself.

    Replaying one expert layer alone otherwise keeps its selected weights hot;
    a serving model touches many other layers before returning to those weights.
    """
    for _ in range(5):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    eviction = torch.empty(cache_mib * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    eviction.zero_()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(50):
        eviction.add_(1)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def dequant(packed, scales, dtype):
    # Independent scalar nibble expansion, without Triton or vLLM helpers.
    values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).float() - 8
    return (values * scales.float().repeat_interleave(32, dim=-1)).to(dtype)


def reference(x, w1, w2, s1, s2, routing, ids, activation):
    result = torch.empty_like(x)
    for token in range(x.shape[0]):
        parts = []
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate_up = F.linear(x[token].float(), dequant(w1[expert], s1[expert], x.dtype).float()).to(x.dtype)
            gate, up = gate_up.float().chunk(2)
            activated = F.silu(gate) if activation == "silu" else F.gelu(gate, approximate="tanh")
            value = (activated * up).to(x.dtype)
            down = F.linear(value.float(), dequant(w2[expert], s2[expert], x.dtype).float())
            parts.append((down * routing[token, slot]).to(x.dtype).float())
        result[token] = torch.stack(parts).sum(0).to(x.dtype)
    return result


def tensors(batch, hidden, intermediate, experts, topk, dtype):
    x = torch.randn((batch, hidden), device="cuda", dtype=dtype)
    w1 = torch.randint(0, 256, (experts, intermediate * 2, hidden // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), device="cuda", dtype=torch.uint8)
    # Changing group scales catch indexing errors hidden by constant scales.
    s1 = (0.003 + torch.rand((experts, intermediate * 2, hidden // 32), device="cuda") * 0.015).to(dtype)
    s2 = (0.003 + torch.rand((experts, hidden, intermediate // 32), device="cuda") * 0.015).to(dtype)
    weights, ids = torch.randn((batch, experts), device="cuda").topk(topk)
    return x, w1, w2, s1, s2, weights.softmax(-1), ids.to(torch.int32)


def checkpoint_tensors(batch, hidden, intermediate, dtype, directory, layer, gemma):
    """Read one real expert layer, not a model instance or its KV caches."""
    from safetensors import safe_open

    values = tensors(batch, hidden, intermediate, 128, 8, dtype)
    x, w1, w2, s1, s2, routing, ids = values
    index_file = directory / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())["weight_map"] if index_file.exists() else None
    with ExitStack() as stack:
        opened = {}

        def tensor(name):
            shard = index[name] if index is not None else "model.safetensors"
            if shard not in opened:
                opened[shard] = stack.enter_context(safe_open(directory / shard, framework="pt", device="cpu"))
            return opened[shard].get_tensor(name)

        for expert in range(128):
            prefix = (f"model.language_model.layers.{layer}.experts.{expert}." if gemma else
                      f"thinker.model.layers.{layer}.mlp.experts.{expert}.")
            for name, weights, scales in (("gate_proj", w1[expert, :intermediate], s1[expert, :intermediate]),
                                          ("up_proj", w1[expert, intermediate:], s1[expert, intermediate:]),
                                          ("down_proj", w2[expert], s2[expert])):
                weights.copy_(tensor(prefix + name + ".weight_packed").view(torch.uint8))
                scales.copy_(tensor(prefix + name + ".weight_scale").to(dtype))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen3-omni-30b", "gemma-4-26b"), default="qwen3-omni-30b")
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--rows", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--warps", type=int, nargs="+", default=(4, 8))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--registered", action="store_true", help="Also exercise the real vLLM expert apply hook")
    parser.add_argument("--checkpoint", type=Path, help="Cached model snapshot: validate using one real expert layer")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--shared-routing", action="store_true", help="Stress expert reuse between different tokens")
    parser.add_argument("--evict-cache-mib", type=int, default=0,
                        help="Evict this many MiB before each timed replay; eviction time is excluded")
    args = parser.parse_args()
    if args.evict_cache_mib < 0:
        parser.error("Cache eviction size must be nonnegative")
    timing = (lambda call: measure_after_eviction(call, args.evict_cache_mib)) if args.evict_cache_mib else measure
    print(json.dumps({"cache_eviction_mib": args.evict_cache_mib, "seed": args.seed}), flush=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    gemma = args.model == "gemma-4-26b"
    hidden, intermediate = (2816, 704) if gemma else (2048, 768)
    dtype = torch.float16 if gemma else torch.bfloat16
    activation = "gelu_tanh" if gemma else "silu"
    activation_enum = MoEActivation.GELU_TANH if gemma else MoEActivation.SILU
    tolerance = 0.005 if gemma else 0.03
    small = tensors(3, 160, 96, 5, 2, dtype)
    expected = reference(*small, activation)
    actual = vector_experts(*small, activation=activation)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    configs = Path(vllm.__file__).parent / "model_executor/layers/fused_moe/configs"
    filename = ("E=128,N=704,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json" if gemma else
                "E=128,N=768,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json")
    profile = json.loads((args.profile or configs / filename).read_text())
    profile.pop("triton_version", None)
    for batch in args.batches:
        values = (checkpoint_tensors(batch, hidden, intermediate, dtype, args.checkpoint, args.layer, gemma)
                  if args.checkpoint else tensors(batch, hidden, intermediate, 128, 8, dtype))
        x, w1, w2, s1, s2, routing, ids = values
        if args.shared_routing:
            ids[1:].copy_(ids[:1].expand_as(ids[1:]))
        expected = reference(*values, activation)
        quant = int4_w4a16_moe_quant_config(s1, s2, block_shape=[0, 32])
        nearest = min(map(int, profile), key=lambda size: abs(size - batch))
        with override_config(profile[str(nearest)]):
            def stock():
                return fused_experts(x, w1, w2, routing, ids, activation=activation_enum, quant_config=quant)

            baseline = stock()
            torch.testing.assert_close(baseline, expected, rtol=tolerance, atol=tolerance)
            ms = timing(stock)
            print(json.dumps({"model": args.model, "batch": batch, "kernel": "tuned_vllm", "ms": ms,
                              "real_weights": bool(args.checkpoint), "shared_routing": args.shared_routing,
                              "max_abs_error": float((baseline.float() - expected.float()).abs().max())}), flush=True)
        for rows in args.rows:
            for warps in args.warps:
                def candidate():
                    return vector_experts(*values, activation=activation, rows=rows, warps=warps)

                actual = candidate()
                if not torch.isfinite(actual).all():
                    raise RuntimeError("Vector experts produced non-finite values")
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                error = float((actual.float() - expected.float()).abs().max())
                ms = timing(candidate)
                print(json.dumps({"model": args.model, "batch": batch, "kernel": "vector",
                                  "rows": rows, "warps": warps, "ms": ms, "max_abs_error": error}), flush=True)
        if args.registered:
            if vector_config(batch, hidden, intermediate, dtype, activation) is None:
                print(json.dumps({"model": args.model, "batch": batch,
                                  "registration_policy": "native_fallback"}), flush=True)
                continue
            from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts
            register()
            context = SimpleNamespace(quant_config=quant, block_shape=[0, 32],
                                      w1_scale=s1, w2_scale=s2, _lora_context=None)
            output, workspace = torch.empty_like(x), torch.empty(0, device=x.device)

            def through_vllm():
                returned = TritonWNA16Experts.apply(
                    context, output, x, w1, w2, routing, ids, activation_enum,
                    128, None, None, None, workspace, workspace, None, False,
                )
                assert returned is None
                return output

            actual = through_vllm()
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            ms = timing(through_vllm)
            print(json.dumps({"model": args.model, "batch": batch, "kernel": "registered_vector",
                              "ms": ms, "max_abs_error": float((actual.float() - expected.float()).abs().max())}), flush=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = through_vllm()
            for _ in range(2):
                # Routing and activations change every token, while a decode
                # graph retains its addresses. Test that replay reads new data.
                previous = replayed.clone()
                x.add_(0.01)
                ids.copy_((ids + 17) % 128)
                routing.copy_(torch.randn_like(routing).softmax(-1))
                graph.replay()
                assert not torch.equal(replayed, previous), "Graph replay returned stale expert outputs"
                expected = reference(*values, activation)
                torch.testing.assert_close(replayed, expected, rtol=tolerance, atol=tolerance)
            print(json.dumps({"model": args.model, "batch": batch,
                              "dynamic_graph_replays": 2, "real_weights": bool(args.checkpoint)}), flush=True)


if __name__ == "__main__":
    main()
