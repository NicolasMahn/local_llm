# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT4 matrix kernels for measured 8060S shapes.

The stock path limits each K tile to one quantization group, requiring many
short iterations for group32 weights. Expanding group scales inside a wider
tile removes that restriction. Optional split-K experiments use deterministic
FP32 reduction; the measured serving dispatch needs just one partition.

This module does not register or replace a serving kernel on import.
Packed-weight decoding derives from vLLM's RDNA hybrid W4A16 kernel;
the accompanying LICENSE-vllm preserves its license.
"""

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


def dense_config(m, k, n, group_size, dtype):
    """Keep the wider tiles restricted to measured BF16 projections and steps."""
    if str(dtype).removeprefix("torch.") != "bfloat16":
        return None
    if group_size == 128 and (((k, n) == (32768, 4096) and m == 512)
                              or ((k, n) == (4096, 65536) and m == 128)):
        block_m = 64
    elif group_size != 32:
        return None
    # Olmo. Several requests generating at once: 2-16 tokens per step. Up to
    # five, the native path sends gate/up to its HIP kernel instead.
    elif (k, n) == (27648, 5120) and 2 <= m <= 16:
        block_m = 16
    elif (k, n) == (5120, 55296) and 6 <= m <= 16:
        block_m = 16
    elif (k, n) in ((27648, 5120), (5120, 55296)) and m in (128, 512):
        block_m = 64
    # Gemma 31B, likewise.
    elif (k, n) == (21504, 5376) and 2 <= m <= 16:
        block_m = 16
    elif (k, n) == (5376, 43008) and 6 <= m <= 16:
        block_m = 16
    elif (k, n) in ((21504, 5376), (5376, 43008)) and m in (128, 512):
        block_m = 64
    else:
        return None
    return {"split_k": 1, "block_m": block_m, "block_n": 64, "block_k": 128,
            "num_warps": 4, "num_stages": 1}


def register():
    """Keep the native custom-op lifecycle and HIP decode path intact."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        return
    from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as native

    stock = native.triton_w4a16_skinny_fmt_gemm
    if getattr(stock, "_local_llm_dense", False):
        return

    def gemm(a, b_q, scales, group_size, zp_bias=8, zp=None):
        # This function is called by the native implementation only after its
        # HIP skinny eligibility check. Bias addition remains in that caller.
        config = None
        if (a.ndim == b_q.ndim == scales.ndim == 2 and a.is_cuda
                and a.device.index == 0 and b_q.device == scales.device == a.device
                and a.is_contiguous() and b_q.is_contiguous() and scales.is_contiguous()
                and b_q.dtype == torch.int32 and scales.dtype == a.dtype
                and group_size in (32, 128) and zp is None and zp_bias == 8):
            m, k = a.shape
            n = b_q.shape[0]
            if b_q.shape == (n, k // 8) and scales.shape == (n, k // group_size):
                config = dense_config(m, k, n, group_size, a.dtype)
        if config is None:
            return stock(a, b_q, scales, group_size, zp_bias=zp_bias, zp=zp)
        if not torch.compiler.is_compiling():
            logger.info_once("Using local multi-group INT4 matmul: K=%s N=%s tile_m=%s",
                             k, n, config["block_m"], scope="global")
        return splitk_w4a16_gemm(a, b_q, scales, group_size, zp_bias=zp_bias, **config)

    gemm._local_llm_dense = True
    native.triton_w4a16_skinny_fmt_gemm = gemm


@triton.jit
def _split_k_matmul(
    a_ptr, b_ptr, scales_ptr, zp_ptr, partial_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    GROUP: tl.constexpr, ZP_BIAS: tl.constexpr, HAS_ZP: tl.constexpr,
    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m, pid_n, pid_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_tiles = tl.cdiv(K, SPLIT_K * BLOCK_K)
    start = pid_s * k_tiles * BLOCK_K
    num_groups = K // GROUP
    shifts = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts = tl.reshape(tl.broadcast_to(shifts[None, :], (BLOCK_K // 8, 8)), (BLOCK_K,))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for tile in range(k_tiles):
        k_start = start + tile * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    (offs_m[:, None] < M) & (offs_k[None, :] < K), 0)
        offs_k8 = k_start // 8 + tl.arange(0, BLOCK_K // 8)
        packed = tl.load(b_ptr + offs_n[:, None] * (K // 8) + offs_k8[None, :],
                         (offs_n[:, None] < N) & (offs_k8[None, :] < K // 8), 0)
        q = tl.interleave(packed, packed)
        q = tl.interleave(q, q)
        q = tl.interleave(q, q)
        q = (q >> shifts[None, :]) & 15

        groups_per_tile: tl.constexpr = (BLOCK_K // GROUP) if BLOCK_K >= GROUP else 1
        group_offsets = k_start // GROUP + tl.arange(0, groups_per_tile)
        scale = tl.load(scales_ptr + offs_n[:, None] * num_groups + group_offsets[None, :],
                        (offs_n[:, None] < N) & (group_offsets[None, :] < num_groups), 0)
        if BLOCK_K >= GROUP:
            scale = tl.reshape(tl.broadcast_to(scale[:, :, None],
                               (BLOCK_N, groups_per_tile, GROUP)), (BLOCK_N, BLOCK_K))
        else:
            scale = tl.broadcast_to(scale, (BLOCK_N, BLOCK_K))
        if HAS_ZP:
            zp = tl.load(zp_ptr + offs_n[:, None] * num_groups + group_offsets[None, :],
                         (offs_n[:, None] < N) & (group_offsets[None, :] < num_groups), 0)
            if BLOCK_K >= GROUP:
                zp = tl.reshape(tl.broadcast_to(zp[:, :, None],
                                (BLOCK_N, groups_per_tile, GROUP)), (BLOCK_N, BLOCK_K))
            else:
                zp = tl.broadcast_to(zp, (BLOCK_N, BLOCK_K))
            weight = ((q.to(scale.dtype) - zp) * scale).to(a.dtype)
        else:
            weight = ((q - ZP_BIAS).to(scale.dtype) * scale).to(a.dtype)
        acc += tl.dot(a, tl.trans(weight), out_dtype=tl.float32)

    tl.store(partial_ptr + pid_s * M * N + offs_m[:, None] * N + offs_n[None, :], acc,
             (offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _sum_splits(partial_ptr, out_ptr, SIZE: tl.constexpr, SPLIT_K: tl.constexpr,
                BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(partial_ptr + splits[:, None] * SIZE + offsets[None, :],
                     (splits[:, None] < SPLIT_K) & (offsets[None, :] < SIZE), 0)
    tl.store(out_ptr + offsets, tl.sum(values, axis=0), offsets < SIZE)


def splitk_w4a16_gemm(a, b_q, scales, group_size, zp_bias=8, zp=None, *,
                      split_k=8, block_m=16, block_n=32, block_k=128,
                      num_warps=4, num_stages=2):
    """Read the existing ExLlama-shuffled int32 [N,K/8] weight buffer."""
    if a.ndim != 2 or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Expected a two-dimensional FP16/BF16 activation matrix")
    m, k = a.shape
    n = b_q.shape[0]
    if min(m, n, k) <= 0 or not a.is_cuda:
        raise ValueError("Expected nonempty GPU matrices")
    if group_size not in (32, 64, 128) or k % group_size:
        raise ValueError("K must contain complete quantization groups of 32, 64 or 128")
    if b_q.dtype != torch.int32 or b_q.shape != (n, k // 8):
        raise ValueError("Expected ExLlama-shuffled int32 weights with shape [N,K/8]")
    if scales.shape != (n, k // group_size) or scales.dtype != a.dtype:
        raise ValueError("Expected activation-dtype per-group scales with shape [N,K/group]")
    if zp is not None and (zp.shape != scales.shape or zp.dtype != scales.dtype):
        raise ValueError("Zero points must have the same shape and dtype as scales")
    tensors = [a, b_q, scales] + ([zp] if zp is not None else [])
    if any(not t.is_contiguous() or t.device != a.device for t in tensors):
        raise ValueError("All tensors must be contiguous and on the same device")
    if split_k not in (1, 2, 4, 8, 16, 32) or block_k not in (32, 64, 128, 256):
        raise ValueError("Unsupported reduction tiling")
    if block_m not in (16, 32, 64) or block_n not in (16, 32, 64, 128):
        raise ValueError("Unsupported output tiling")
    if split_k * m * n * 4 > 256 * 1024**2:
        raise ValueError("Split-K workspace exceeds the 256 MiB experiment budget")
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    # With one partition the matmul can cast directly into the result: no
    # extra buffer or reduction launch is needed.
    partial = out if split_k == 1 else torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), split_k)
    _split_k_matmul[grid](a, b_q, scales, zp if zp is not None else scales, partial,
                          m, n, k, group_size, zp_bias, zp is not None,
                          split_k, block_m, block_n, block_k,
                          num_warps=num_warps, num_stages=num_stages)
    if split_k != 1:
        _sum_splits[(triton.cdiv(m * n, 256),)](partial, out, m * n, split_k, 256)
    return out
