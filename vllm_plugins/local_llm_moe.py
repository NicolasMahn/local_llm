"""Small-batch symmetric INT4 experts without sorting or padded matrix rows.

The regular matrix kernel pads each selected expert to at least sixteen rows.
Decode often has only one real row per expert. These two vector kernels read
only the selected experts and fuse the activation and the expert reduction.
Weights keep vLLM's uint8 layout; no second copy or re-quantization is needed.

Gated experts (Qwen Omni, Gemma) multiply an activated gate projection with an
up projection. Nemotron's experts have one up projection, squared after ReLU.
"""

from functools import wraps

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from triton.language.extra import libdevice

logger = init_logger(__name__)


@triton.jit
def _gate_up(X, W, S, IDS, A,
             H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, GROUP: tl.constexpr,
             GELU: tl.constexpr, ROWS: tl.constexpr, KH: tl.constexpr):
    route = tl.program_id(0)
    token = route // TOPK
    expert = tl.load(IDS + route)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, KH)
    valid = (rows[:, None] < I) & (k[None, :] < H)
    x = tl.load(X + token * H + k, k < H, 0).to(tl.float32)
    offset = expert * (2 * I * (H // 2)) + rows[:, None] * (H // 2) + k[None, :] // 2
    gate_q = tl.load(W + offset, valid, 0)
    up_q = tl.load(W + offset + I * (H // 2), valid, 0)
    shift = (k[None, :] % 2) * 4
    gate_q = ((gate_q >> shift) & 15).to(tl.float32) - 8
    up_q = ((up_q >> shift) & 15).to(tl.float32) - 8
    scale_offset = expert * (2 * I * (H // GROUP)) + rows[:, None] * (H // GROUP) + k[None, :] // GROUP
    gate_s = tl.load(S + scale_offset, valid, 0).to(tl.float32)
    up_s = tl.load(S + scale_offset + I * (H // GROUP), valid, 0).to(tl.float32)
    # vLLM rounds dequantized weights and both GEMM outputs to activation
    # precision. Preserve those rounding points before fusing the activation.
    gate_w = (gate_q * gate_s).to(X.dtype.element_ty).to(tl.float32)
    up_w = (up_q * up_s).to(X.dtype.element_ty).to(tl.float32)
    gate = tl.sum(gate_w * x[None, :], 1).to(X.dtype.element_ty).to(tl.float32)
    up = tl.sum(up_w * x[None, :], 1).to(X.dtype.element_ty).to(tl.float32)
    if GELU:
        activated = 0.5 * gate * (1 + libdevice.tanh(0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)))
    else:
        activated = gate / (1 + tl.exp(-gate))
    tl.store(A + route * I + rows, (activated * up).to(A.dtype.element_ty), rows < I)


@triton.jit
def _up_square(X, W, S, IDS, A,
               H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, GROUP: tl.constexpr,
               ROWS: tl.constexpr, KH: tl.constexpr):
    route = tl.program_id(0)
    token = route // TOPK
    expert = tl.load(IDS + route)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, KH)
    valid = (rows[:, None] < I) & (k[None, :] < H)
    x = tl.load(X + token * H + k, k < H, 0).to(tl.float32)
    packed = tl.load(W + expert * (I * (H // 2)) + rows[:, None] * (H // 2) + k[None, :] // 2, valid, 0)
    q = ((packed >> ((k[None, :] % 2) * 4)) & 15).to(tl.float32) - 8
    scale = tl.load(S + expert * (I * (H // GROUP)) + rows[:, None] * (H // GROUP) + k[None, :] // GROUP,
                    valid, 0).to(tl.float32)
    # Same rounding points as the native path: weights, then the projection.
    weights = (q * scale).to(X.dtype.element_ty).to(tl.float32)
    up = tl.sum(weights * x[None, :], 1).to(X.dtype.element_ty).to(tl.float32)
    up = tl.maximum(up, 0)
    tl.store(A + route * I + rows, (up * up).to(A.dtype.element_ty), rows < I)


@triton.jit
def _down_reduce(A, W, S, IDS, ROUTE_WEIGHT, OUT,
                 H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, GROUP: tl.constexpr,
                 ROWS: tl.constexpr, KI: tl.constexpr):
    token = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, KI)
    valid = (rows[:, None] < H) & (k[None, :] < I)
    result = tl.full((ROWS,), 0, tl.float32)
    for slot in range(TOPK):
        route = token * TOPK + slot
        expert = tl.load(IDS + route)
        x = tl.load(A + route * I + k, k < I, 0).to(tl.float32)
        offset = expert * (H * (I // 2)) + rows[:, None] * (I // 2) + k[None, :] // 2
        packed = tl.load(W + offset, valid, 0)
        q = ((packed >> ((k[None, :] % 2) * 4)) & 15).to(tl.float32) - 8
        scale_offset = expert * (H * (I // GROUP)) + rows[:, None] * (I // GROUP) + k[None, :] // GROUP
        scale = tl.load(S + scale_offset, valid, 0).to(tl.float32)
        weights = (q * scale).to(A.dtype.element_ty).to(tl.float32)
        projection = tl.sum(weights * x[None, :], 1)
        routing = tl.load(ROUTE_WEIGHT + route).to(tl.float32)
        weighted = (projection * routing).to(A.dtype.element_ty).to(tl.float32)
        result += weighted
    tl.store(OUT + token * H + rows, result.to(OUT.dtype.element_ty), rows < H)


def vector_experts(x, w1, w2, s1, s2, routing_weights, expert_ids,
                   *, activation="silu", group=32, output=None, rows=2, warps=4):
    """Experimental vector path; caller must supply contiguous symmetric INT4 tensors."""
    batch, hidden = x.shape
    intermediate = w2.shape[2] * 2
    topk = expert_ids.shape[1]
    if activation not in ("silu", "gelu_tanh", "relu2_no_mul"):
        raise ValueError(f"Unsupported expert activation: {activation}")
    temporary = torch.empty((batch * topk, intermediate), dtype=x.dtype, device=x.device)
    if output is None:
        output = torch.empty_like(x)
    grid = (batch * topk, triton.cdiv(intermediate, rows))
    width = triton.next_power_of_2(hidden)
    if activation == "relu2_no_mul":
        _up_square[grid](x, w1, s1, expert_ids, temporary,
                         hidden, intermediate, topk, group, rows, width, num_warps=warps)
    else:
        _gate_up[grid](x, w1, s1, expert_ids, temporary,
                       hidden, intermediate, topk, group, activation == "gelu_tanh", rows, width,
                       num_warps=warps)
    _down_reduce[(batch, triton.cdiv(hidden, rows))](
        temporary, w2, s2, expert_ids, routing_weights, output,
        hidden, intermediate, topk, group, rows, triton.next_power_of_2(intermediate),
        num_warps=warps,
    )
    return output


# (hidden, expert width, precision, activation) -> the quantization group and
# routing width that were measured with it, and the tiling that was fastest.
# Anything else stays native.
MEASURED = {
    # Qwen Omni
    (2048, 768, torch.bfloat16, "silu"): {"group": 32, "topk": 8, "rows": 2, "warps": 4},
    # Gemma 26B
    (2816, 704, torch.float16, "gelu_tanh"): {"group": 32, "topk": 8, "rows": 2, "warps": 4},
    # Nemotron 3.5 Lightning: 1.7x warm and 1.35-1.5x with a cold cache over
    # the native path, two layers and seeds (benchmarks/all-models-2026-11-14.md).
    (2688, 1856, torch.bfloat16, "relu2_no_mul"): {"group": 64, "topk": 6, "rows": 2, "warps": 8},
}


def vector_config(batch, hidden, intermediate, dtype, activation):
    """Only measured decode shapes may bypass vLLM's general implementation."""
    # Two tokens can select the same experts. Native matrix grouping then
    # reuses weights better than these separate vector operations.
    layout = MEASURED.get((hidden, intermediate, dtype, activation))
    if batch == 1 and layout is not None:
        return {"rows": layout["rows"], "warps": layout["warps"]}
    return None


def supported_inputs(experts, output, x, w1, w2, routing, ids, activation,
                     global_experts, expert_map, a1_scale, a2_scale,
                     router_weight_on_input):
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or ids.ndim != 2:
        return None
    name = getattr(activation, "value", activation)
    hidden, intermediate = x.shape[1], w2.shape[2] * 2
    config = vector_config(x.shape[0], hidden, intermediate, x.dtype, name)
    if config is None:
        return None
    layout = MEASURED[hidden, intermediate, x.dtype, name]
    group, topk = layout["group"], layout["topk"]
    # A gated expert stores its gate and up projections in one matrix.
    width = intermediate * (1 if name.endswith("_no_mul") else 2)
    quant = experts.quant_config
    activation_config = getattr(experts, "activation_config", None)
    if (not quant.use_int4_w4a16 or quant.w1_zp is not None or quant.w2_zp is not None
            or experts.block_shape is None or tuple(experts.block_shape) != (0, group)
            or expert_map is not None or global_experts not in (-1, 128)
            or a1_scale is not None or a2_scale is not None or router_weight_on_input
            or getattr(experts, "_lora_context", None) is not None
            or getattr(activation_config, "clamp_limit", None) is not None):
        return None
    if (w1.shape != (128, width, hidden // 2)
            or w2.shape != (128, hidden, intermediate // 2)
            or experts.w1_scale.shape != (128, width, hidden // group)
            or experts.w2_scale.shape != (128, hidden, intermediate // group)
            or ids.shape != (x.shape[0], topk) or routing.shape != ids.shape
            or output.shape != x.shape):
        return None
    if (w1.dtype != torch.uint8 or w2.dtype != torch.uint8
            or experts.w1_scale.dtype != x.dtype or experts.w2_scale.dtype != x.dtype
            or output.dtype != x.dtype or routing.dtype != torch.float32
            or ids.dtype not in (torch.int32, torch.int64)):
        return None
    tensors = (x, w1, w2, experts.w1_scale, experts.w2_scale, routing, ids, output)
    if (x.device.type != "cuda" or x.device.index != 0
            or any(t.device != x.device or not t.is_contiguous() for t in tensors)):
        return None
    return config


def register():
    """Replace only the validated local decode case; all other paths stay native."""
    from vllm.platforms import current_platform
    if not current_platform.is_rocm():
        return
    from vllm.platforms.rocm import on_gfx1151
    if not on_gfx1151():
        return
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts

    stock = TritonWNA16Experts.apply
    if getattr(stock, "_local_llm_vector", False):
        return

    @wraps(stock)
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
        config = supported_inputs(self, output, hidden_states, w1, w2,
                                  topk_weights, topk_ids, activation, global_num_experts,
                                  expert_map, a1q_scale, a2_scale, apply_router_weight_on_input)
        if config is not None:
            if not torch.compiler.is_compiling():
                logger.info_once("Using local vector experts: dtype=%s hidden=%s intermediate=%s",
                                 hidden_states.dtype, hidden_states.shape[1], w2.shape[2] * 2, scope="global")
            name = getattr(activation, "value", activation)
            layout = MEASURED[hidden_states.shape[1], w2.shape[2] * 2, hidden_states.dtype, name]
            vector_experts(hidden_states, w1, w2, self.w1_scale, self.w2_scale,
                           topk_weights, topk_ids, activation=name, group=layout["group"],
                           output=output, **config)
            return
        if (not torch.compiler.is_compiling() and getattr(hidden_states, "ndim", None) == 2 and w2.ndim == 3
                and vector_config(hidden_states.shape[0], hidden_states.shape[1], w2.shape[2] * 2,
                                  hidden_states.dtype, getattr(activation, "value", activation)) is not None):
            logger.info_once("Local vector uses native fallback: x=%s w1=%s scales=%s/%s "
                             "routing=%s ids=%s contiguous=%s/%s/%s/%s block=%s",
                             hidden_states.dtype, w1.dtype, self.w1_scale.dtype, self.w2_scale.dtype,
                             topk_weights.dtype, topk_ids.dtype, hidden_states.is_contiguous(),
                             self.w1_scale.is_contiguous(), self.w2_scale.is_contiguous(),
                             output.is_contiguous(), str(self.block_shape), scope="global")
        return stock(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                     activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                     workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input)

    apply._local_llm_vector = True
    TritonWNA16Experts.apply = apply
