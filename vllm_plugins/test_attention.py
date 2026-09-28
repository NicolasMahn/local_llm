"""Checks the fast prefill path against vLLM's own kernel, on the GPU.

Run inside the model image: python3 vllm_plugins/test_attention.py
Contexts stay short enough that vLLM's slow kernel cannot trip the GPU watchdog.
"""

import time

import torch
import vllm.v1.attention.backends.triton_attn as triton_attn

import local_llm_attention

stock = triton_attn.unified_attention
local_llm_attention.register()
fast = triton_attn.unified_attention
BLOCK = 16


def batch(heads, kv_heads, size, requests, window, images=None, clamp=False):
    """requests: (new tokens, sequence length) per request, as vLLM batches them."""
    torch.manual_seed(0)
    blocks_per_request = max(length for _, length in requests) // BLOCK + 1
    k = torch.randn(len(requests) * blocks_per_request + 1, BLOCK, kv_heads, size,
                    dtype=torch.float16, device="cuda")
    # Pages are shuffled so a mix-up between requests or blocks cannot go unseen.
    order = torch.randperm(k.shape[0] - 1, device="cuda").int()
    tokens = sum(new for new, _ in requests)
    starts = [0]
    for new, _ in requests:
        starts.append(starts[-1] + new)
    ranges = None
    if images is not None:
        width = max(len(r) for r in images)
        ranges = torch.tensor([r + [(0, 0)] * (width - len(r)) for r in images],
                              dtype=torch.int32, device="cuda")
    return dict(
        q=torch.randn(tokens, heads, size, dtype=torch.float16, device="cuda"),
        k=k, v=torch.randn_like(k),
        cu_seqlens_q=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        max_seqlen_q=max(new for new, _ in requests),
        seqused_k=torch.tensor([length for _, length in requests], dtype=torch.int32, device="cuda"),
        max_seqlen_k=max(length for _, length in requests),
        softmax_scale=size ** -0.5, causal=True, window_size=(window, window),
        block_table=order.view(len(requests), -1)[:, :blocks_per_request].contiguous()
        if order.numel() >= len(requests) * blocks_per_request else None,
        softcap=0, q_descale=None, k_descale=None, v_descale=None,
        mm_prefix_range=ranges, mm_prefix_clamp_sliding_window=clamp,
    )


def compare(name, args):
    expected, got = torch.empty_like(args["q"]), torch.empty_like(args["q"])
    stock(out=expected, **args)
    fast(out=got, **args)
    torch.cuda.synchronize()
    error = (expected.float() - got.float()).abs().max().item()
    assert error < 2e-3, f"{name}: off by {error}"
    print(f"ok  {name} (max difference {error:.1e})")


mixed = [(1, 5000), (512, 20000), (100, 300), (1, 17)]
compare("Gemma full attention, 512-wide heads, mixed batch",
        batch(16, 2, 512, mixed, window=-1))
compare("Gemma full attention with an image in the prompt",
        batch(16, 2, 512, mixed, window=-1, images=[[], [(19700, 19979)], [(10, 60)], []]))
compare("Gemma sliding window with images, clamped like Gemma 4",
        batch(16, 8, 256, mixed, window=1023, images=[[(100, 379)], [(18000, 18279), (19700, 19979)], [(10, 60)], []],
              clamp=True))
compare("sliding window with an unclamped image stays on vLLM's kernel",
        batch(16, 8, 256, mixed, window=1023, images=[[], [(15000, 19800)], [], []]))

args = batch(16, 2, 512, [(512, 32768)], window=-1)
out = torch.empty_like(args["q"])
for kernel in (stock, fast):
    kernel(out=out, **args)
    torch.cuda.synchronize()
    start = time.time()
    kernel(out=out, **args)
    torch.cuda.synchronize()
    print(f"{'stock' if kernel is stock else 'fast '} 512 tokens at 32k: {(time.time() - start) * 1000:.0f} ms per layer")
