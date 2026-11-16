"""How fast this GPU can read memory: the ceiling for generating one token.

A dense model reads all of its weights for every token, and every model reads
its whole attention cache. Tokens per second times bytes read per token cannot
exceed what this measures, so it tells a kernel that is slow from one that is
already limited by the memory itself.

Run inside the model image with the GPU devices, while no model is serving.
Each operation only reads the buffer; the fastest one is the estimate.
"""

import argparse
import json
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=float, default=8, help="Buffer size; far larger than any cache")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    import torch

    size = int(args.gib * 1024**3)
    data = torch.randint(0, 127, (size,), device="cuda", dtype=torch.int8)
    operations = {
        "max_int8": lambda: data.max(),
        "max_int32": lambda: data.view(torch.int32).max(),
        "max_int64": lambda: data.view(torch.int64).max(),
        "sum_bf16": lambda: data.view(torch.bfloat16).float().sum(),
        "count_nonzero_int64": lambda: torch.count_nonzero(data.view(torch.int64)),
    }
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    result = {"device": torch.cuda.get_device_properties(0).name, "buffer_gib": args.gib, "read_gb_per_s": {}}
    for name, call in operations.items():
        try:
            call()
            samples = []
            for _ in range(args.repeats):
                start.record()
                call()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 1000)
            result["read_gb_per_s"][name] = size / statistics.median(samples) / 1e9
        except RuntimeError as exc:
            result["read_gb_per_s"][name] = f"failed: {exc}"[:120]
    rates = [rate for rate in result["read_gb_per_s"].values() if isinstance(rate, float)]
    result["best_read_gb_per_s"] = max(rates) if rates else None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
