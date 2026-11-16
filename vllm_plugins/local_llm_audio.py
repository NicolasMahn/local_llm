# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's qwen3_omni_moe_thinker.py, copyright the vLLM contributors.
# See LICENSE-vllm. The convolution, positional and transformer operations stay
# in their original order; only their length metadata and row packing change.
"""Keep Omni's known audio lengths on the host instead of reading them back."""

from dataclasses import dataclass
from functools import wraps
from numbers import Integral


# These are the source bodies whose data flow is adapted below. Updating the
# image must not silently keep a copy of an upstream algorithm that changed.
_NATIVE_SOURCE_HASHES = (
    "78cb7ec276c2581cdf2a21423ed6d164478359fd379d8e46778b28e6df649997",
    "8738faef68ec8a15f7d5fbf6f1e7d992ce85f3f011fc67e4cf1d5805432c1b82",
    "0e7a72093dc556a2aa1bafcb641c018f437ce12721568b66a40ed414f2b47c38",
)


def native_contract_matches(encoder_forward, process_audio, output_lengths):
    import hashlib
    import inspect
    import textwrap

    try:
        actual = tuple(hashlib.sha256(textwrap.dedent(inspect.getsource(function)).strip().encode()).hexdigest()
                       for function in (encoder_forward, process_audio, output_lengths))
    except (OSError, TypeError):
        return False
    return actual == _NATIVE_SOURCE_HASHES


@dataclass(frozen=True)
class AudioLayout:
    feature_lengths: tuple[int, ...]
    chunk_lengths: tuple[int, ...]
    conv_lengths: tuple[int, ...]
    padded_conv_length: int
    valid_rows: tuple[int, ...]
    output_lengths: tuple[int, ...]
    cu_seqlens: tuple[int, ...]
    max_seqlen: int


def audio_layout(feature_lengths, *, window=100, inference_window=800):
    """Derive native chunking and packing from CPU lengths, without tensors.

    The convolution runs on padded 100-frame chunks. Packing must remove each
    chunk's padding before combining chunks into the larger attention windows.
    In particular, rounding a whole recording's length by eight is incorrect.
    """
    if (not isinstance(window, Integral) or isinstance(window, bool) or window <= 0
            or not isinstance(inference_window, Integral) or isinstance(inference_window, bool)
            or inference_window < window or inference_window % window):
        raise ValueError("Audio windows must be positive integral multiples")
    lengths = tuple(feature_lengths)
    if not lengths or any(not isinstance(n, Integral) or isinstance(n, bool) or n <= 0 for n in lengths):
        raise ValueError("Audio feature lengths must be positive integers")
    lengths = tuple(int(n) for n in lengths)
    chunks, outputs = [], []
    for length in lengths:
        full, remainder = divmod(length, window)
        recording_chunks = [int(window)] * full + ([remainder] if remainder else [])
        chunks.extend(recording_chunks)
        outputs.append(sum((n + 7) // 8 for n in recording_chunks))
    conv_lengths = tuple((n + 7) // 8 for n in chunks)
    maximum = max(conv_lengths)
    valid_rows = tuple(i * maximum + row for i, length in enumerate(conv_lengths) for row in range(length))
    attention_window = maximum * (inference_window // window)
    cu = [0]
    for length in outputs:
        full, remainder = divmod(length, attention_window)
        for _ in range(full):
            cu.append(cu[-1] + attention_window)
        if remainder:
            cu.append(cu[-1] + remainder)
    return AudioLayout(lengths, tuple(chunks), conv_lengths, maximum, valid_rows,
                       tuple(outputs), tuple(cu), max(b - a for a, b in zip(cu, cu[1:])))


def supported_audio_input(tower, features, lengths):
    """Restrict the adaptation to the BF16 Omni layout measured on this box."""
    import torch

    return (
        getattr(tower, "training", True) is False
        and getattr(tower, "num_mel_bins", None) == 128
        and getattr(tower, "n_window", None) == 50
        and getattr(tower, "n_window_infer", None) == 800
        and getattr(getattr(tower, "attn_backend", None), "name", None) in {
            "FLASH_ATTN", "ROCM_AITER_FA", "TRITON_ATTN",
        }
        and isinstance(getattr(tower, "conv_chunksize", None), int)
        and tower.conv_chunksize > 0
        and getattr(tower, "dtype", None) == torch.bfloat16
        and getattr(features, "ndim", None) == 2 and features.shape[0] == 128 and features.shape[1] > 0
        and features.device.type == "cuda" and features.device.index == 0
        and features.device == tower.device and features.is_contiguous()
        and features.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and getattr(lengths, "ndim", None) == 1 and lengths.shape[0] > 0
        and lengths.device.type == "cpu" and lengths.is_contiguous()
        and lengths.dtype in (torch.int32, torch.int64)
    )


def encode_audio(tower, features, layout):
    """Run native audio math with a precomputed host layout.

    Native forward mixes its GPU work with length calculations and dynamic
    boolean indexing, so a wrapper around that forward cannot remove the
    intermediate synchronizations. Keep this adaptation close to that body.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F
    from vllm.utils.torch_utils import async_tensor_h2d

    chunks = features.T.split(layout.chunk_lengths, dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2).unsqueeze(1)

    def convolve(chunk):
        result = F.gelu(tower.conv2d1(chunk))
        result = F.gelu(tower.conv2d2(result))
        return F.gelu(tower.conv2d3(result))

    if padded_feature.shape[0] <= tower.conv_chunksize:
        padded_embed = convolve(padded_feature)
    else:
        padded_embed = torch.cat([convolve(chunk) for chunk in padded_feature.split(tower.conv_chunksize)], dim=0)

    batch, channels, frequency, time = padded_embed.shape
    padded_embed = tower.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * frequency)
    )
    position = tower.positional_embedding.positional_embedding[:padded_embed.shape[1], :].unsqueeze(0).to(padded_embed.dtype)
    padded_embed = padded_embed + position

    indices = async_tensor_h2d(layout.valid_rows, dtype=torch.long, device=features.device)
    hidden_states = padded_embed.flatten(0, 1).index_select(0, indices)
    cu = async_tensor_h2d(layout.cu_seqlens, dtype=torch.int32, device=features.device)
    # Attention wrappers require Tensor metadata but only read this scalar on
    # the CPU. Its value was already known before any encoder GPU work began.
    maximum = torch.tensor(layout.max_seqlen, dtype=torch.int32, device="cpu")
    for layer in tower.layers:
        hidden_states = layer(hidden_states, cu, maximum)
    hidden_states = tower.ln_post(hidden_states)
    hidden_states = tower.proj1(hidden_states)
    hidden_states = tower.act(hidden_states)
    return tower.proj2(hidden_states)


def _prepare_audio(stock, tower_type, logger=None):
    @wraps(stock)
    def process(self, audio_input):
        import torch

        tower = self.audio_tower
        features = audio_input["input_features"]
        lengths = audio_input["audio_feature_lengths"]
        if (not isinstance(tower, tower_type) or not supported_audio_input(tower, features, lengths)
                or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()):
            return stock(self, audio_input)
        try:
            layout = audio_layout(lengths.tolist())
        except (TypeError, ValueError):
            return stock(self, audio_input)
        if sum(layout.feature_lengths) != features.shape[1]:
            return stock(self, audio_input)
        if logger is not None:
            logger.info_once("Using local Omni audio CPU layout and fixed row packing", scope="global")
        result = encode_audio(tower, features.to(tower.dtype), layout)
        return result.split(layout.output_lengths)

    process._local_llm_host_audio = True
    return process


def register():
    """Install before model creation; unknown inputs keep native execution."""
    from vllm.platforms import current_platform
    if not current_platform.is_rocm():
        return
    from vllm.platforms.rocm import on_gfx1151
    if not on_gfx1151():
        return
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeConditionalGenerationMixin,
        _get_feat_extract_output_lengths,
    )

    stock = Qwen3OmniMoeConditionalGenerationMixin._process_audio_input
    if getattr(stock, "_local_llm_host_audio", False):
        return
    from vllm.logger import init_logger
    logger = init_logger("vllm.local_llm_audio")
    if not native_contract_matches(Qwen3OmniMoeAudioEncoder.forward, stock, _get_feat_extract_output_lengths):
        logger.warning("Local Omni audio optimization disabled: the native audio implementation changed or its source is unavailable")
        return
    Qwen3OmniMoeConditionalGenerationMixin._process_audio_input = _prepare_audio(stock, Qwen3OmniMoeAudioEncoder, logger)
