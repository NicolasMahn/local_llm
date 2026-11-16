"""Measure Omni encoder-stack scheduling with actual vLLM attention wrappers.

Uses one synthetic transformer block repeatedly at Omni's actual encoder
widths and depths. Includes dense projections, layer norms, attention, GELU,
residuals and one bound preparation per stack. Omits audio convolutions,
vision patch embedding/rotary positions and the merger. These timings are an
isolated scheduling experiment, not full-model latency or quality results.
Run inside the GPU image with local_llm_encoder.py on PYTHONPATH.
"""

import argparse
import json
import statistics
import time
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from local_llm_encoder import attention_bound_on_host
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention


class EncoderBlock(nn.Module):
    def __init__(self, hidden, intermediate, heads, epsilon):
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden, eps=epsilon)
        self.norm2 = nn.LayerNorm(hidden, eps=epsilon)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=True)
        self.out = nn.Linear(hidden, hidden, bias=True)
        self.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.fc2 = nn.Linear(intermediate, hidden, bias=True)
        self.attention = MMEncoderAttention(num_heads=heads, head_size=self.head_dim)

    def forward(self, x, cu_seqlens, maximum):
        tokens = x.shape[0]
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(1, tokens, self.heads, self.head_dim) for t in (q, k, v)]
        attended = self.attention(q, k, v, cu_seqlens=cu_seqlens, max_seqlen=maximum)
        x = x + self.out(attended.reshape(tokens, self.hidden))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


def check_metadata_guards():
    cpu = torch.tensor(17, dtype=torch.int32, device="cpu")
    assert attention_bound_on_host(None) is None
    assert attention_bound_on_host(cpu) is cpu
    gpu = cpu.cuda()
    host = attention_bound_on_host(gpu)
    assert host.device.type == "cpu" and host.dtype == gpu.dtype
    assert host.shape == gpu.shape and host.item() == 17
    with patch("torch.compiler.is_compiling", return_value=True):
        assert attention_bound_on_host(gpu) is gpu
    gpu + 1  # Load the scalar-add kernel before graph capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert attention_bound_on_host(gpu) is gpu
        replayed = gpu + 1
    graph.replay()
    torch.cuda.synchronize()
    assert replayed.item() == 18


def scalar_reads(call):
    counts = {"cpu": 0, "cuda": 0}
    item = torch.Tensor.item

    def count(tensor, *args, **kwargs):
        counts[tensor.device.type] = counts.get(tensor.device.type, 0) + 1
        return item(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "item", count):
        call()
        torch.cuda.synchronize()
    return counts


def measure(call):
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000, output


@torch.inference_mode()
def run_case(name, tokens, repeats):
    audio = name == "audio"
    hidden, intermediate, heads, depth, epsilon = (
        (1280, 5120, 20, 32, 1e-5) if audio else (1152, 4304, 16, 27, 1e-6)
    )
    block = EncoderBlock(hidden, intermediate, heads, epsilon).eval()
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    # The audio tower attends in windows of at most 104 post-convolution
    # tokens. Vision attends across the image's unmerged patch sequence.
    boundaries = list(range(0, tokens, 104)) + [tokens] if audio else [0, tokens]
    cu = torch.tensor(boundaries, dtype=torch.int32, device="cuda")

    def forward(host_bound):
        maximum = (cu[1:] - cu[:-1]).max()
        if host_bound:
            maximum = attention_bound_on_host(maximum)
        result = x
        for _ in range(depth):
            result = block(result, cu, maximum)
        return result

    for _ in range(2):
        forward(False)
        forward(True)
    torch.cuda.synchronize()
    stock, optimized = forward(False), forward(True)
    torch.cuda.synchronize()
    assert torch.isfinite(stock).all() and torch.isfinite(optimized).all()
    torch.testing.assert_close(optimized, stock, rtol=0, atol=0)
    samples = {"gpu_bound": [], "host_bound": []}
    for repeat in range(repeats):
        candidates = [("gpu_bound", False), ("host_bound", True)]
        if repeat % 2:
            candidates.reverse()
        for label, host in candidates:
            elapsed, _ = measure(lambda: forward(host))
            samples[label].append(elapsed)
    reads = {
        "gpu_bound": scalar_reads(lambda: forward(False)),
        "host_bound": scalar_reads(lambda: forward(True)),
    }
    print(json.dumps({
        "case": name, "tokens": tokens, "layers": depth,
        "backend": str(block.attention.attn_backend), "dtype": "bfloat16",
        "bitwise_equal": True,
        "median_ms": {key: statistics.median(values) for key, values in samples.items()},
        "samples_ms": samples, "scalar_reads": reads,
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("audio", "vision", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least three alternating repetitions")
    device = torch.cuda.get_device_properties(0)
    if device.gcnArchName.split(":")[0] != "gfx1151":
        raise SystemExit("This experiment targets the 8060S (gfx1151)")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(args.seed)
    with set_current_vllm_config(VllmConfig()):
        check_metadata_guards()
        if args.modality in ("audio", "both"):
            for tokens in (64, 312):
                run_case("audio", tokens, args.repeats)
        if args.modality in ("vision", "both"):
            for tokens in (512, 3960):
                run_case("vision", tokens, args.repeats)


if __name__ == "__main__":
    main()
