"""Compare generic and tuned 8060S expert kernels inside the model image.

Uses small synthetic expert tensors without resetting the serving cache.
Each candidate gets its own graph: configuration changes after capture do
not change captured kernel launches. Output is checked before timing it.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import vllm
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, get_default_config


def measure(call):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            call()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 10)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen3-omni-30b", "gemma-4-26b"), default="qwen3-omni-30b")
    parser.add_argument("--profile", type=Path, help="Override the installed tuning profile")
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32, 128, 512))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reverse", action="store_true", help="Time the tuned profile first")
    args = parser.parse_args()
    if any(batch < 1 or batch > 512 for batch in args.batches):
        parser.error("Keep steps between 1 and the 512-token prefill safety cap")
    device = torch.cuda.get_device_properties(0)
    if device.gcnArchName.split(":")[0] != "gfx1151":
        raise SystemExit("This comparison is specific to the 8060S (gfx1151)")
    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    experts, topk = 128, 8
    if args.model == "gemma-4-26b":
        hidden, intermediate = 2816, 704
        dtype, activation = torch.float16, MoEActivation.GELU_TANH
        filename = "E=128,N=704,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json"
    else:
        hidden, intermediate = 2048, 768
        dtype, activation = torch.bfloat16, MoEActivation.SILU
        filename = "E=128,N=768,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json"
    w1 = torch.randint(0, 256, (experts, intermediate * 2, hidden // 2), dtype=torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), dtype=torch.uint8)
    s1 = torch.full((experts, intermediate * 2, hidden // 32), 0.01, dtype=dtype)
    s2 = torch.full((experts, hidden, intermediate // 32), 0.01, dtype=dtype)
    quant = int4_w4a16_moe_quant_config(s1, s2, block_shape=[0, 32])
    configs = Path(vllm.__file__).parent / "model_executor/layers/fused_moe/configs"
    profile = json.loads((args.profile or configs / filename).read_text())
    profile.pop("triton_version", None)

    for batch in args.batches:
        x = torch.randn((batch, hidden), dtype=dtype)
        scores = torch.randn((batch, experts), dtype=torch.float32)
        weights, ids = torch.topk(scores, topk)
        weights, ids = weights.softmax(-1), ids.to(torch.int32)
        default = get_default_config(batch, experts, intermediate, hidden // 2,
                                     topk, "int4_w4a16", [0, 32])
        nearest = min(map(int, profile), key=lambda size: abs(size - batch))
        reference = None
        candidates = [("default", default), ("profile", profile[str(nearest)])]
        if args.reverse:
            candidates.reverse()
        for name, config in candidates:
            with override_config(config):
                def call():
                    return fused_experts(x, w1, w2, weights, ids, activation=activation, quant_config=quant)

                output = call()
                torch.cuda.synchronize()
                if not torch.isfinite(output).all():
                    raise RuntimeError(f"Non-finite expert output at batch {batch} ({name})")
                if reference is None:
                    reference = output.clone()
                else:
                    tolerance = 0.005 if dtype == torch.float16 else 0.02
                    torch.testing.assert_close(output, reference, rtol=tolerance, atol=tolerance)
                ms = measure(call)
                print(json.dumps({"model": args.model, "seed": args.seed,
                                  "batch": batch, "config": name, "ms": ms}), flush=True)


if __name__ == "__main__":
    main()
