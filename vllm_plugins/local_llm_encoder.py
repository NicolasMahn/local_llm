"""Prepare Omni's attention launch bound once, before its encoder layers."""

from functools import wraps

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def attention_bound_on_host(bound: torch.Tensor | None) -> torch.Tensor | None:
    """Keep scalar metadata off the GPU once eager encoder execution starts.

    vLLM's attention wrappers need a Python launch bound and call ``item()``
    every layer. Passing a CPU scalar preserves their Tensor schema without
    waiting for all prior GPU work at every layer. Graph capture and compile
    retain the original behavior; moving data to the host there is unsafe.
    """
    if bound is None or bound.device.type != "cuda":
        return bound
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return bound
    if bound.ndim != 0:
        raise ValueError("Encoder attention launch bound must be a scalar")
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    with gpu_sync_allowed():
        logger.info_once("Using local Omni encoder host attention bound", scope="global")
        return bound.cpu()


def _prepare_once(stock):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    needs_host_bound = {
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
        AttentionBackendEnum.TRITON_ATTN,
    }

    @wraps(stock)
    def prepare(self, cu_seqlens):
        bound = stock(self, cu_seqlens)
        return attention_bound_on_host(bound) if self.attn_backend in needs_host_bound else bound

    prepare._local_llm_encoder_bound = True
    return prepare


def register():
    """Called by vLLM at startup, only after this experiment is integrated."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        return
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3Omni_VisionTransformer,
        Qwen3OmniMoeAudioEncoder,
    )

    for encoder in (Qwen3Omni_VisionTransformer, Qwen3OmniMoeAudioEncoder):
        stock = encoder.compute_attn_mask_seqlen
        if not getattr(stock, "_local_llm_encoder_bound", False):
            encoder.compute_attn_mask_seqlen = _prepare_once(stock)
