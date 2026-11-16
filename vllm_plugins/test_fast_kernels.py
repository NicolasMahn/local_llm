"""CPU-only checks for fast-path boundaries and the native fallback contract.

Run in the model image, where torch and vLLM are installed. Tensor metadata
stands in for the large expert buffers; no test launches a GPU kernel.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import local_llm_encoder as encoder
import local_llm_dense as dense
import local_llm_kernels as coordinator
import local_llm_moe as moe


@pytest.mark.parametrize("m,k,n,group", [
    (512, 32768, 4096, 128), (128, 4096, 65536, 128),
    (128, 27648, 5120, 32), (512, 27648, 5120, 32),
    (128, 5120, 55296, 32), (512, 5120, 55296, 32),
])
def test_added_dense_shapes_have_only_the_measured_tile(m, k, n, group):
    config = dense.dense_config(m, k, n, group, torch.bfloat16)
    assert config == {"split_k": 1, "block_m": 64, "block_n": 64, "block_k": 128,
                      "num_warps": 4, "num_stages": 1}


@pytest.mark.parametrize("m,k,n", [
    *((m, 27648, 5120) for m in range(2, 17)),
    *((m, 5120, 55296) for m in range(6, 17)),
])
def test_olmo_concurrent_decode_steps_use_the_small_row_tile(m, k, n):
    config = dense.dense_config(m, k, n, 32, torch.bfloat16)
    assert config == {"split_k": 1, "block_m": 16, "block_n": 64, "block_k": 128,
                      "num_warps": 4, "num_stages": 1}


@pytest.mark.parametrize("m,k,n,group", [
    # Granite's native group-128 kernels were faster at every small step measured.
    (1, 32768, 4096, 128), (4, 32768, 4096, 128), (16, 32768, 4096, 128), (128, 32768, 4096, 128),
    (1, 4096, 65536, 128), (8, 4096, 65536, 128), (512, 4096, 65536, 128),
    (1, 27648, 5120, 32), (17, 27648, 5120, 32), (64, 27648, 5120, 32),
    # Up to five tokens Olmo's gate/up runs in the native HIP kernel, which rounds differently.
    (1, 5120, 55296, 32), (4, 5120, 55296, 32), (5, 5120, 55296, 32), (17, 5120, 55296, 32),
])
def test_added_dense_unmeasured_or_rejected_steps_stay_native(m, k, n, group):
    assert dense.dense_config(m, k, n, group, torch.bfloat16) is None


@dataclass
class TensorMetadata:
    shape: tuple
    dtype: object
    device: object = torch.device("cuda:0")
    contiguous: bool = True

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def is_contiguous(self):
        return self.contiguous


@pytest.fixture
def valid_inputs():
    dtype = torch.bfloat16
    experts = SimpleNamespace(
        quant_config=SimpleNamespace(use_int4_w4a16=True, w1_zp=None, w2_zp=None),
        block_shape=[0, 32],
        w1_scale=TensorMetadata((128, 1536, 64), dtype),
        w2_scale=TensorMetadata((128, 2048, 24), dtype),
        _lora_context=None,
        activation_config=SimpleNamespace(clamp_limit=None),
    )
    return {
        "experts": experts,
        "output": TensorMetadata((1, 2048), dtype),
        "x": TensorMetadata((1, 2048), dtype),
        "w1": TensorMetadata((128, 1536, 1024), torch.uint8),
        "w2": TensorMetadata((128, 2048, 384), torch.uint8),
        "routing": TensorMetadata((1, 8), torch.float32),
        "ids": TensorMetadata((1, 8), torch.int32),
        "activation": SimpleNamespace(value="silu"),
        "global_experts": 128,
        "expert_map": None,
        "a1_scale": None,
        "a2_scale": None,
        "router_weight_on_input": False,
    }


@pytest.mark.parametrize("batch", [1])
def test_measured_omni_decode_shape_is_selected(batch):
    assert moe.vector_config(batch, 2048, 768, torch.bfloat16, "silu") == {"rows": 2, "warps": 4}


@pytest.mark.parametrize("batch", [1])
def test_measured_gemma_decode_shape_is_selected(batch):
    assert moe.vector_config(batch, 2816, 704, torch.float16, "gelu_tanh") == {"rows": 2, "warps": 4}


def test_complete_gemma_metadata_selects_fast_path(valid_inputs):
    p = valid_inputs
    dtype = torch.float16
    p["x"] = TensorMetadata((1, 2816), dtype)
    p["output"] = TensorMetadata((1, 2816), dtype)
    p["w1"] = TensorMetadata((128, 1408, 1408), torch.uint8)
    p["w2"] = TensorMetadata((128, 2816, 352), torch.uint8)
    p["experts"].w1_scale = TensorMetadata((128, 1408, 88), dtype)
    p["experts"].w2_scale = TensorMetadata((128, 2816, 22), dtype)
    p["activation"] = SimpleNamespace(value="gelu_tanh")
    assert moe.supported_inputs(**p) == {"rows": 2, "warps": 4}


@pytest.mark.parametrize("parameters", [
    (0, 2048, 768, torch.bfloat16, "silu"),
    (2, 2048, 768, torch.bfloat16, "silu"),
    (2, 2816, 704, torch.float16, "gelu_tanh"),
    (3, 2048, 768, torch.bfloat16, "silu"),
    (512, 2048, 768, torch.bfloat16, "silu"),
    (1, 2049, 768, torch.bfloat16, "silu"),
    (1, 2048, 800, torch.bfloat16, "silu"),
    (1, 2048, 768, torch.float16, "silu"),
    (1, 2048, 768, torch.bfloat16, "gelu_tanh"),
])
def test_unmeasured_omni_shape_keeps_native_dispatch(parameters):
    assert moe.vector_config(*parameters) is None


def test_complete_supported_metadata_selects_fast_path_without_allocating(valid_inputs):
    assert moe.supported_inputs(**valid_inputs) == {"rows": 2, "warps": 4}
    valid_inputs["global_experts"] = -1
    valid_inputs["ids"].dtype = torch.int64
    assert moe.supported_inputs(**valid_inputs) == {"rows": 2, "warps": 4}


@pytest.mark.parametrize("field,value", [
    ("quant_config.use_int4_w4a16", False),
    ("quant_config.w1_zp", object()),
    ("quant_config.w2_zp", object()),
    ("block_shape", None),
    ("block_shape", [0, 128]),
    ("_lora_context", object()),
    ("activation_config.clamp_limit", 7.0),
    ("w1_scale", TensorMetadata((128, 1536, 64), torch.float16)),
    ("w2_scale", TensorMetadata((128, 2048, 25), torch.bfloat16)),
    ("w1_scale", TensorMetadata((128, 1536, 64), torch.bfloat16, contiguous=False)),
])
def test_quantization_or_expert_features_outside_contract_fall_back(valid_inputs, field, value):
    target = valid_inputs["experts"]
    pieces = field.split(".")
    for piece in pieces[:-1]:
        target = getattr(target, piece)
    setattr(target, pieces[-1], value)
    assert moe.supported_inputs(**valid_inputs) is None


@pytest.mark.parametrize("field,value", [
    ("global_experts", 127),
    ("expert_map", object()),
    ("a1_scale", object()),
    ("a2_scale", object()),
    ("router_weight_on_input", True),
    ("w1", TensorMetadata((128, 1536), torch.uint8)),
    ("w1", TensorMetadata((129, 1536, 1024), torch.uint8)),
    ("w2", TensorMetadata((128, 2048, 384), torch.int32)),
    ("ids", TensorMetadata((1, 7), torch.int32)),
    ("ids", TensorMetadata((1, 8), torch.float32)),
    ("routing", TensorMetadata((1, 8), torch.float16)),
    ("routing", TensorMetadata((1, 7), torch.float32)),
    ("output", TensorMetadata((1, 2047), torch.bfloat16)),
    ("output", TensorMetadata((1, 2048), torch.float16)),
    ("x", TensorMetadata((1, 2048), torch.bfloat16, device=torch.device("cpu"))),
    ("w1", TensorMetadata((128, 1536, 1024), torch.uint8, device=torch.device("cuda:1"))),
    ("x", TensorMetadata((1, 2048), torch.bfloat16, contiguous=False)),
])
def test_tensor_layout_routing_and_device_mismatches_fall_back(valid_inputs, field, value):
    valid_inputs[field] = value
    assert moe.supported_inputs(**valid_inputs) is None


NEMOTRON = (2688, 1856, torch.bfloat16, "relu2_no_mul")


@pytest.fixture
def nemotron_inputs(valid_inputs):
    """Nemotron's experts: one ungated up projection, group 64, six routes."""
    p = valid_inputs
    dtype = torch.bfloat16
    p["x"] = TensorMetadata((1, 2688), dtype)
    p["output"] = TensorMetadata((1, 2688), dtype)
    p["w1"] = TensorMetadata((128, 1856, 1344), torch.uint8)
    p["w2"] = TensorMetadata((128, 2688, 928), torch.uint8)
    p["experts"].block_shape = [0, 64]
    p["experts"].w1_scale = TensorMetadata((128, 1856, 42), dtype)
    p["experts"].w2_scale = TensorMetadata((128, 2688, 29), dtype)
    p["routing"] = TensorMetadata((1, 6), torch.float32)
    p["ids"] = TensorMetadata((1, 6), torch.int32)
    p["activation"] = SimpleNamespace(value="relu2_no_mul")
    return p


def test_ungated_group64_layout_is_selected_only_with_its_own_shapes(nemotron_inputs):
    assert moe.MEASURED[NEMOTRON] == {"group": 64, "topk": 6, "rows": 2, "warps": 8}
    assert moe.supported_inputs(**nemotron_inputs) == {"rows": 2, "warps": 8}
    assert moe.vector_config(2, *NEMOTRON) is None


@pytest.mark.parametrize("field,value", [
    # A gated matrix of twice the rows, group-32 scales or eight routes belong to other models.
    ("w1", TensorMetadata((128, 3712, 1344), torch.uint8)),
    ("ids", TensorMetadata((1, 8), torch.int32)),
    ("x", TensorMetadata((2, 2688), torch.bfloat16)),
])
def test_ungated_layout_with_another_models_shapes_falls_back(nemotron_inputs, field, value):
    nemotron_inputs[field] = value
    assert moe.supported_inputs(**nemotron_inputs) is None


def test_ungated_layout_with_group32_scales_falls_back(nemotron_inputs):
    nemotron_inputs["experts"].block_shape = [0, 32]
    assert moe.supported_inputs(**nemotron_inputs) is None
    nemotron_inputs["experts"].block_shape = [0, 64]
    nemotron_inputs["experts"].w1_scale = TensorMetadata((128, 1856, 84), torch.bfloat16)
    assert moe.supported_inputs(**nemotron_inputs) is None


def test_fast_dispatch_passes_the_models_own_quantization_group(monkeypatch, nemotron_inputs):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts

    fake_platform(monkeypatch)
    stock = Mock(side_effect=AssertionError("Validated fast dispatch should not call stock"))
    stock._local_llm_vector = False
    vector = Mock()
    monkeypatch.setattr(TritonWNA16Experts, "apply", stock)
    monkeypatch.setattr(moe, "vector_experts", vector)
    moe.register()
    p = nemotron_inputs
    TritonWNA16Experts.apply(p["experts"], p["output"], p["x"], p["w1"], p["w2"], p["routing"], p["ids"],
                             p["activation"], 128, None, None, None, object(), object(), None, False)
    assert vector.call_args.kwargs["group"] == 64
    assert vector.call_args.kwargs["activation"] == "relu2_no_mul"


def fake_platform(monkeypatch, *, rocm=True, gfx1151=True):
    import vllm.platforms
    import vllm.platforms.rocm

    monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace(is_rocm=lambda: rocm))
    monkeypatch.setattr(vllm.platforms.rocm, "on_gfx1151", lambda: gfx1151)


def test_registration_is_idempotent_and_fallback_forwards_every_argument(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts

    fake_platform(monkeypatch)
    arguments = tuple(object() for _ in range(16))
    sentinel = object()
    stock = Mock(return_value=sentinel)
    stock._local_llm_vector = False
    vector = Mock(side_effect=AssertionError("Fallback must not launch the vector kernel"))
    monkeypatch.setattr(TritonWNA16Experts, "apply", stock)
    monkeypatch.setattr(moe, "supported_inputs", Mock(return_value=None))
    monkeypatch.setattr(moe, "vector_experts", vector)

    moe.register()
    wrapped = TritonWNA16Experts.apply
    moe.register()
    assert TritonWNA16Experts.apply is wrapped
    assert wrapped(*arguments) is sentinel
    stock.assert_called_once_with(*arguments)
    vector.assert_not_called()


def test_fast_dispatch_writes_the_supplied_output_and_returns_none(monkeypatch, valid_inputs):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts

    fake_platform(monkeypatch)
    stock = Mock(side_effect=AssertionError("Validated fast dispatch should not call stock"))
    stock._local_llm_vector = False
    vector = Mock()
    monkeypatch.setattr(TritonWNA16Experts, "apply", stock)
    monkeypatch.setattr(moe, "supported_inputs", Mock(return_value={"rows": 2, "warps": 4}))
    monkeypatch.setattr(moe, "vector_experts", vector)
    moe.register()
    p = valid_inputs
    result = TritonWNA16Experts.apply(
        p["experts"], p["output"], p["x"], p["w1"], p["w2"], p["routing"], p["ids"],
        p["activation"], 128, None, None, None, object(), object(), None, False,
    )
    assert result is None
    vector.assert_called_once_with(
        p["x"], p["w1"], p["w2"], p["experts"].w1_scale, p["experts"].w2_scale,
        p["routing"], p["ids"], activation="silu", group=32, output=p["output"], rows=2, warps=4,
    )
    stock.assert_not_called()


def test_moe_compilation_does_not_emit_python_logs(monkeypatch, valid_inputs):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts

    fake_platform(monkeypatch)
    stock = Mock()
    stock._local_llm_vector = False
    monkeypatch.setattr(TritonWNA16Experts, "apply", stock)
    monkeypatch.setattr(moe, "supported_inputs", Mock(return_value={"rows": 2, "warps": 4}))
    monkeypatch.setattr(moe, "vector_experts", Mock())
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    log = Mock(side_effect=AssertionError("Python logging must stay outside compiled graphs"))
    monkeypatch.setattr(moe.logger, "info_once", log)
    moe.register()
    p = valid_inputs
    TritonWNA16Experts.apply(p["experts"], p["output"], p["x"], p["w1"], p["w2"],
                            p["routing"], p["ids"], p["activation"], 128, None, None, None,
                            object(), object(), None, False)
    moe.vector_experts.assert_called_once()
    log.assert_not_called()


@pytest.mark.parametrize("rocm,gfx1151", [(False, True), (True, False)])
def test_registration_leaves_other_platforms_native(monkeypatch, rocm, gfx1151):
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts

    fake_platform(monkeypatch, rocm=rocm, gfx1151=gfx1151)
    stock = Mock()
    monkeypatch.setattr(TritonWNA16Experts, "apply", stock)
    moe.register()
    assert TritonWNA16Experts.apply is stock


class FakeGpuBound:
    device = torch.device("cuda:0")

    def __init__(self, ndim=0):
        self.ndim = ndim
        self.cpu_result = object()
        self.cpu = Mock(return_value=self.cpu_result)


def test_encoder_none_and_cpu_scalar_are_identity_without_gpu_queries(monkeypatch):
    query = Mock(side_effect=AssertionError("CPU metadata must not query GPU capture state"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", query)
    cpu = torch.tensor(17, dtype=torch.int32, device="cpu")
    assert encoder.attention_bound_on_host(None) is None
    assert encoder.attention_bound_on_host(cpu) is cpu
    query.assert_not_called()


@pytest.mark.parametrize("compiling,capturing", [(True, False), (False, True)])
def test_encoder_compile_and_capture_keep_gpu_metadata(monkeypatch, compiling, capturing):
    bound = FakeGpuBound()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    assert encoder.attention_bound_on_host(bound) is bound
    bound.cpu.assert_not_called()


def test_encoder_eager_metadata_copies_to_cpu_once(monkeypatch):
    from vllm.utils import gpu_sync_debug

    bound = FakeGpuBound()
    permission = Mock(side_effect=nullcontext)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(gpu_sync_debug, "gpu_sync_allowed", permission)
    assert encoder.attention_bound_on_host(bound) is bound.cpu_result
    bound.cpu.assert_called_once_with()
    permission.assert_called_once_with()


def test_encoder_rejects_a_nonscalar_before_copying(monkeypatch):
    bound = FakeGpuBound(ndim=1)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(ValueError, match="must be a scalar"):
        encoder.attention_bound_on_host(bound)
    bound.cpu.assert_not_called()


def test_encoder_wrapper_prepares_once_only_when_the_backend_uses_the_bound(monkeypatch):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    gpu_bound = object()
    host_bound = object()
    stock = Mock(return_value=gpu_bound)
    prepare = Mock(return_value=host_bound)
    monkeypatch.setattr(encoder, "attention_bound_on_host", prepare)
    wrapped = encoder._prepare_once(stock)
    flash = SimpleNamespace(attn_backend=AttentionBackendEnum.FLASH_ATTN)
    sdpa = SimpleNamespace(attn_backend=AttentionBackendEnum.TORCH_SDPA)
    cu_seqlens = object()

    assert wrapped(flash, cu_seqlens) is host_bound
    stock.assert_called_once_with(flash, cu_seqlens)
    prepare.assert_called_once_with(gpu_bound)
    stock.reset_mock()
    prepare.reset_mock()
    assert wrapped(sdpa, cu_seqlens) is gpu_bound
    stock.assert_called_once_with(sdpa, cu_seqlens)
    prepare.assert_not_called()


@pytest.mark.parametrize("batch", range(2, 17))
def test_dense_measured_small_down_projection_is_selected(batch):
    assert dense.dense_config(batch, 21504, 5376, 32, torch.bfloat16) == {
        "split_k": 1, "block_m": 16, "block_n": 64, "block_k": 128,
        "num_warps": 4, "num_stages": 1,
    }


@pytest.mark.parametrize("batch", range(6, 17))
def test_dense_measured_small_gate_up_projection_is_selected(batch):
    assert dense.dense_config(batch, 5376, 43008, 32, torch.bfloat16) == {
        "split_k": 1, "block_m": 16, "block_n": 64, "block_k": 128,
        "num_warps": 4, "num_stages": 1,
    }


@pytest.mark.parametrize("batch", [128, 512])
@pytest.mark.parametrize("shape", [(21504, 5376), (5376, 43008)])
def test_dense_measured_prefill_projections_are_selected(batch, shape):
    assert dense.dense_config(batch, *shape, 32, torch.bfloat16) == {
        "split_k": 1, "block_m": 64, "block_n": 64, "block_k": 128,
        "num_warps": 4, "num_stages": 1,
    }


@pytest.mark.parametrize("parameters", [
    (1, 21504, 5376, 32, torch.bfloat16),
    (17, 21504, 5376, 32, torch.bfloat16),
    (32, 21504, 5376, 32, torch.bfloat16),
    (64, 21504, 5376, 32, torch.bfloat16),
    (127, 21504, 5376, 32, torch.bfloat16),
    (129, 21504, 5376, 32, torch.bfloat16),
    (511, 21504, 5376, 32, torch.bfloat16),
    (513, 21504, 5376, 32, torch.bfloat16),
    (2, 5376, 43008, 32, torch.bfloat16),
    (5, 5376, 43008, 32, torch.bfloat16),
    (17, 5376, 43008, 32, torch.bfloat16),
    (2, 21504, 5376, 128, torch.bfloat16),
    (2, 21504, 5376, 32, torch.float16),
    (2, 17408, 5120, 128, torch.bfloat16),
    (2, 19968, 6656, 128, torch.bfloat16),
    (2, 15360, 3840, 32, torch.bfloat16),
])
def test_dense_unmeasured_shapes_and_precisions_keep_native_dispatch(parameters):
    assert dense.dense_config(*parameters) is None


@pytest.fixture
def valid_dense_inputs():
    return {
        "a": TensorMetadata((2, 21504), torch.bfloat16),
        "b_q": TensorMetadata((5376, 2688), torch.int32),
        "scales": TensorMetadata((5376, 672), torch.bfloat16),
        "group_size": 32,
        "zp_bias": 8,
        "zp": None,
    }


def register_dense_spies(monkeypatch):
    from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as native

    fake_platform(monkeypatch)
    stock = Mock(return_value=object())
    stock._local_llm_dense = False
    candidate = Mock(return_value=object())
    monkeypatch.setattr(native, "triton_w4a16_skinny_fmt_gemm", stock)
    monkeypatch.setattr(dense, "splitk_w4a16_gemm", candidate)
    dense.register()
    return native.triton_w4a16_skinny_fmt_gemm, stock, candidate


def test_dense_registration_is_idempotent_and_dispatches_without_changing_arguments(monkeypatch, valid_dense_inputs):
    from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as native

    wrapped, stock, candidate = register_dense_spies(monkeypatch)
    dense.register()
    assert native.triton_w4a16_skinny_fmt_gemm is wrapped
    result = wrapped(**valid_dense_inputs)
    assert result is candidate.return_value
    p = valid_dense_inputs
    candidate.assert_called_once_with(
        p["a"], p["b_q"], p["scales"], 32, zp_bias=8,
        split_k=1, block_m=16, block_n=64, block_k=128, num_warps=4, num_stages=1,
    )
    stock.assert_not_called()


def test_dense_compilation_does_not_emit_python_logs(monkeypatch, valid_dense_inputs):
    wrapped, stock, candidate = register_dense_spies(monkeypatch)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    log = Mock(side_effect=AssertionError("Python logging must stay outside compiled graphs"))
    monkeypatch.setattr(dense.logger, "info_once", log)
    assert wrapped(**valid_dense_inputs) is candidate.return_value
    log.assert_not_called()
    stock.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("a", TensorMetadata((1, 21504), torch.bfloat16)),
    ("a", TensorMetadata((2, 21504), torch.bfloat16, device=torch.device("cpu"))),
    ("a", TensorMetadata((2, 21504), torch.bfloat16, device=torch.device("cuda:1"))),
    ("a", TensorMetadata((2, 21504), torch.bfloat16, contiguous=False)),
    ("b_q", TensorMetadata((5376, 2688), torch.uint8)),
    ("b_q", TensorMetadata((5376, 2687), torch.int32)),
    ("b_q", TensorMetadata((5376, 2688), torch.int32, device=torch.device("cuda:1"))),
    ("b_q", TensorMetadata((5376, 2688), torch.int32, contiguous=False)),
    ("scales", TensorMetadata((5376, 672), torch.float16)),
    ("scales", TensorMetadata((5376, 671), torch.bfloat16)),
    ("scales", TensorMetadata((5376, 672), torch.bfloat16, contiguous=False)),
    ("group_size", 128),
    ("zp_bias", 0),
    ("zp", object()),
])
def test_dense_layout_quantization_or_unmeasured_shape_preserves_native_fallback(
        monkeypatch, valid_dense_inputs, field, value):
    wrapped, stock, candidate = register_dense_spies(monkeypatch)
    valid_dense_inputs[field] = value
    p = valid_dense_inputs
    assert wrapped(**p) is stock.return_value
    stock.assert_called_once_with(
        p["a"], p["b_q"], p["scales"], p["group_size"], zp_bias=p["zp_bias"], zp=p["zp"],
    )
    candidate.assert_not_called()


@pytest.mark.parametrize("rocm,gfx1151", [(False, True), (True, False)])
def test_dense_registration_does_not_change_other_platforms(monkeypatch, rocm, gfx1151):
    from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as native

    fake_platform(monkeypatch, rocm=rocm, gfx1151=gfx1151)
    stock = Mock()
    monkeypatch.setattr(native, "triton_w4a16_skinny_fmt_gemm", stock)
    dense.register()
    assert native.triton_w4a16_skinny_fmt_gemm is stock


def mocked_coordinator_modules(monkeypatch):
    calls = []
    registrations = {}
    for name in ("local_llm_dense", "local_llm_encoder", "local_llm_moe"):
        module = ModuleType(name)
        module.register = Mock(side_effect=lambda name=name: calls.append(name))
        monkeypatch.setitem(sys.modules, name, module)
        registrations[name] = module.register
    return calls, registrations


def test_kernel_coordinator_disable_switch_invokes_no_registration(monkeypatch):
    calls, registrations = mocked_coordinator_modules(monkeypatch)
    monkeypatch.setenv("LOCAL_LLM_FAST_KERNELS", "0")
    coordinator.register()
    assert calls == []
    for register in registrations.values():
        register.assert_not_called()


@pytest.mark.parametrize("setting", [None, "1"])
def test_kernel_coordinator_registers_each_improvement_by_default(monkeypatch, setting):
    calls, registrations = mocked_coordinator_modules(monkeypatch)
    if setting is None:
        monkeypatch.delenv("LOCAL_LLM_FAST_KERNELS", raising=False)
    else:
        monkeypatch.setenv("LOCAL_LLM_FAST_KERNELS", setting)
    coordinator.register()
    assert calls == ["local_llm_dense", "local_llm_encoder", "local_llm_moe"]
    for register in registrations.values():
        register.assert_called_once_with()
