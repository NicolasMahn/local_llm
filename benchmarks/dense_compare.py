"""Check and time dense INT4 experiments inside the Strix Halo model image."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm_plugins"))
from local_llm_dense import splitk_w4a16_gemm
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    _rdna_hybrid_w4a16_apply_impl,
)
from vllm.utils.platform_utils import num_compute_units


SHAPES = {
    "granite42-down": (32768, 4096, 128),
    "granite42-gate": (4096, 65536, 128),
    "olmo31-down": (27648, 5120, 32),
    "olmo31-gate": (5120, 55296, 32),
    "qwen-down": (17408, 5120, 128),
    "muse-down": (19968, 6656, 128),
    "gemma31-down": (21504, 5376, 32),
    "gemma12-down": (15360, 3840, 32),
    "qwen-gate": (5120, 34816, 128),
    "muse-gate": (6656, 39936, 128),
    "gemma31-gate": (5376, 43008, 32),
    "gemma12-gate": (3840, 30720, 32),
    "gemma26-gate": (2816, 4224, 32),
    "gemma26-down": (2112, 2816, 32),
    "omni-qkv": (2048, 5120, 32),
    "omni-out": (4096, 2048, 32),
}


def dequantize(q, scales, group, dtype):
    """Independent ExLlama unshuffle and group-scale reference in PyTorch."""
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=q.device)
    values = ((q[:, :, None].to(torch.int64) >> shifts) & 15).reshape(q.shape[0], -1)
    return ((values - 8).to(dtype) * scales.repeat_interleave(group, dim=1)).to(dtype)


def measure(call):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            call()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 10)
    return statistics.median(samples)


def check_reference(output, reference):
    # Native BF16 dequant/matmul arithmetic already differs from the eager
    # FP32 reference by ~0.3% in L2. A fixed absolute tolerance near zero
    # incorrectly rejects cancellation-sensitive elements in the stock path.
    if not torch.isfinite(output).all() or not torch.isfinite(reference).all():
        raise AssertionError("Non-finite independent-reference inputs")
    difference = output.float() - reference
    relative_l2 = (difference.square().sum() / reference.square().sum().clamp_min(1e-30)).sqrt().item()
    max_error = difference.abs().max().item()
    quantum = 1 / 128 if output.dtype == torch.bfloat16 else 1 / 1024
    if relative_l2 >= (0.006 if output.dtype == torch.bfloat16 else 0.001):
        raise AssertionError(f"Independent-reference relative L2 error: {relative_l2}")
    if max_error > quantum * (reference.abs().max().item() + 1):
        raise AssertionError(f"Independent-reference maximum error: {max_error}")
    return {"relative_l2": relative_l2, "max_error": max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", choices=tuple(SHAPES), nargs="+", default=["qwen-down", "gemma31-down"])
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--split", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--block-k", type=int, nargs="+", default=[128])
    parser.add_argument("--block-n", type=int, nargs="+", default=[32])
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Record rejected candidates without timing or enabling them")
    args = parser.parse_args()
    if any(m < 1 or m > 512 for m in args.batches):
        parser.error("Keep batches within the 512-token prefill safety cap")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] != "gfx1151":
        raise SystemExit("These experiments are intended for gfx1151")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    torch.set_default_device("cuda")
    cu_count = num_compute_units()

    # Odd tail dimensions and changing scales catch packing, masking and
    # group-boundary mistakes before the expensive full-size cases.
    for group in (32, 128):
        for m, k, n in ((3, 384, 67), (17, 640, 129)):
            q = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32)
            scales = (torch.rand(n, k // group) * 0.03 + 0.005).to(dtype)
            a = torch.randn(m, k, dtype=dtype)
            reference = a.float() @ dequantize(q, scales, group, dtype).float().T
            for split in args.split:
                for block_k in args.block_k:
                    for block_n in args.block_n:
                        out = splitk_w4a16_gemm(a, q, scales, group, split_k=split,
                                                block_k=block_k, block_n=block_n,
                                                block_m=args.block_m,
                                                num_stages=args.stages, num_warps=args.warps)
                        check_reference(out, reference)
    print(json.dumps({"independent_reference": "passed", "dtype": args.dtype}), flush=True)

    for shape in args.shapes:
        k, n, group = SHAPES[shape]
        q = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32)
        scales = (torch.rand(n, k // group) * 0.03 + 0.005).to(dtype)
        for m in args.batches:
            a = torch.randn(m, k, dtype=dtype)
            stock = lambda: _rdna_hybrid_w4a16_apply_impl(a, q.view(torch.int8), scales,
                                                         None, None, cu_count, group)
            reference = stock()
            sampled = a.float() @ dequantize(q[:64], scales[:64], group, dtype).float().T
            check_reference(reference[:, :64], sampled)
            candidates = [("stock", {}, stock)]
            for split in args.split:
                for block_k in args.block_k:
                    for block_n in args.block_n:
                        config = {"split_k": split, "block_k": block_k, "block_n": block_n,
                                  "block_m": args.block_m,
                                  "num_stages": args.stages, "num_warps": args.warps}
                        def call(config=config):
                            return splitk_w4a16_gemm(a, q, scales, group, **config)
                        candidates.append(("split_k", config, call))
            if args.reverse:
                candidates.reverse()
            for name, config, call in candidates:
                out = call()
                if not torch.isfinite(out).all():
                    raise RuntimeError(f"Non-finite output: {shape}, {m}, {config}")
                try:
                    torch.testing.assert_close(out, reference,
                                               rtol=0.01 if dtype == torch.bfloat16 else 0.002,
                                               atol=0.005 if dtype == torch.bfloat16 else 0.002)
                    errors = check_reference(out[:, :64], sampled)
                except AssertionError as exc:
                    if not args.keep_going:
                        raise
                    print(json.dumps({"shape": shape, "batch": m, "dtype": args.dtype,
                                      "kernel": name, "config": config, "accepted": False,
                                      "rejection": str(exc)}), flush=True)
                    continue
                ms = measure(call)
                print(json.dumps({"shape": shape, "batch": m, "dtype": args.dtype,
                                  "kernel": name, "config": config, "ms": ms,
                                  "reference_errors": errors}), flush=True)


if __name__ == "__main__":
    main()
