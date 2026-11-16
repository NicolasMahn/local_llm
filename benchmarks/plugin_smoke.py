"""Check the installed entry point and startup opt-out without loading weights."""

import importlib.metadata
import os

os.environ["LOCAL_LLM_FAST_KERNELS"] = "0"

from vllm.plugins import load_general_plugins
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as dense
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder

entries = importlib.metadata.entry_points(group="vllm.general_plugins")
assert any(entry.name == "local_llm_kernels" for entry in entries)
load_general_plugins()
assert not getattr(dense.triton_w4a16_skinny_fmt_gemm, "_local_llm_dense", False)
assert not getattr(TritonWNA16Experts.apply, "_local_llm_vector", False)
assert not getattr(Qwen3OmniMoeAudioEncoder.compute_attn_mask_seqlen, "_local_llm_encoder_bound", False)

# Still before model creation: explicitly exercise the enabled startup path.
os.environ["LOCAL_LLM_FAST_KERNELS"] = "1"
import local_llm_kernels

local_llm_kernels.register()
assert dense.triton_w4a16_skinny_fmt_gemm._local_llm_dense
assert TritonWNA16Experts.apply._local_llm_vector
assert Qwen3OmniMoeAudioEncoder.compute_attn_mask_seqlen._local_llm_encoder_bound
before = (dense.triton_w4a16_skinny_fmt_gemm, TritonWNA16Experts.apply,
          Qwen3OmniMoeAudioEncoder.compute_attn_mask_seqlen)
local_llm_kernels.register()
assert before == (dense.triton_w4a16_skinny_fmt_gemm, TritonWNA16Experts.apply,
                  Qwen3OmniMoeAudioEncoder.compute_attn_mask_seqlen)
print("Installed plugin discovered; opt-out and all three idempotent startup hooks passed.")
