"""CPU-only coverage of the real symmetric GPTQ post-load failure contract."""

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest

import local_llm_rdna


MODULE = "vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16"


@pytest.fixture
def kernel_type(monkeypatch):
    class Kernel:
        def __init__(self, zero_points=False, name="qzeros"):
            self.config = SimpleNamespace(zero_points=zero_points)
            self.w_zp_name = name
            self.context = object()
            self.calls = []

        def process_weights_after_loading(self, layer):
            self.calls.append((layer, getattr(layer, self.w_zp_name, None) if self.w_zp_name else None))
            if self.config.zero_points:
                layer.qzeros = layer.processed_zeros
            layer.weights_converted = True
            return "native-result"

        def apply_weights(self, layer, x, bias=None):
            return x, getattr(layer, self.w_zp_name, None) if self.w_zp_name else None, bias, self.context

    module = ModuleType(MODULE)
    module.RDNAHybridW4A16LinearKernel = Kernel
    monkeypatch.setitem(sys.modules, MODULE, module)
    return Kernel


def test_symmetric_raw_zeros_are_cleared_after_native_conversion_only(kernel_type, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_FAST_KERNELS", "0")
    kernel = kernel_type()
    raw_zeros, x, bias = object(), object(), object()
    layer = SimpleNamespace(qzeros=raw_zeros)
    apply = kernel_type.apply_weights
    context = kernel.context
    local_llm_rdna.register()
    assert kernel.process_weights_after_loading(layer) == "native-result"
    assert kernel.calls == [(layer, raw_zeros)]
    assert layer.weights_converted and layer.qzeros is None
    assert kernel_type.apply_weights is apply
    assert kernel.apply_weights(layer, x, bias) == (x, None, bias, context)


def test_asymmetric_processed_zeros_and_forward_stay_native(kernel_type):
    kernel = kernel_type(zero_points=True)
    processed = object()
    layer = SimpleNamespace(qzeros=object(), processed_zeros=processed)
    apply = kernel_type.apply_weights
    local_llm_rdna.register()
    assert kernel.process_weights_after_loading(layer) == "native-result"
    assert layer.qzeros is processed
    assert kernel_type.apply_weights is apply
    assert kernel.apply_weights(layer, "x", "bias")[1:3] == (processed, "bias")


@pytest.mark.parametrize("name,present", [(None, False), ("qzeros", False), ("qzeros", True)])
def test_absent_and_already_empty_optional_parameters_are_unchanged(kernel_type, name, present):
    kernel = kernel_type(name=name)
    layer = SimpleNamespace(**({"qzeros": None} if present else {}))
    local_llm_rdna.register()
    kernel.process_weights_after_loading(layer)
    assert hasattr(layer, "qzeros") is present


def test_registration_is_idempotent_and_preserves_native_signature(kernel_type):
    stock = kernel_type.process_weights_after_loading
    local_llm_rdna.register()
    wrapped = kernel_type.process_weights_after_loading
    local_llm_rdna.register()
    assert kernel_type.process_weights_after_loading is wrapped
    assert inspect.unwrap(wrapped) is stock
    assert inspect.signature(wrapped) == inspect.signature(stock)


def test_native_conversion_failure_is_not_hidden_or_mutated(kernel_type, monkeypatch):
    def broken(self, layer):
        raise ValueError("native packing failure")

    monkeypatch.setattr(kernel_type, "process_weights_after_loading", broken)
    zeros = object()
    layer = SimpleNamespace(qzeros=zeros)
    local_llm_rdna.register()
    with pytest.raises(ValueError, match="native packing failure"):
        kernel_type().process_weights_after_loading(layer)
    assert layer.qzeros is zeros


def real_modules():
    torch = pytest.importorskip("torch")
    # The ROCm package queries device metadata at import. The actual test
    # tensors stay on meta, and no forward or GPU kernel is executed.
    if not torch.cuda.is_available():
        pytest.skip("The installed ROCm package needs device visibility for import")
    from vllm.model_executor import parameter
    from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as native
    return torch, parameter, native


def patch_parallel_rank(monkeypatch, parameter):
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)


def test_real_autogptq_factory_granite_qkv_meta_postload(monkeypatch):
    torch, parameter, native = real_modules()
    from vllm.model_executor.layers.quantization import auto_gptq

    patch_parallel_rank(monkeypatch, parameter)
    monkeypatch.setattr(auto_gptq, "verify_marlin_supported", lambda **kwargs: None)
    monkeypatch.setattr(auto_gptq, "choose_mp_linear_kernel", lambda config: native.RDNAHybridW4A16LinearKernel)
    stock = native.RDNAHybridW4A16LinearKernel.process_weights_after_loading
    if getattr(stock, "_local_llm_rdna_symmetric_zp", False):
        stock = stock.__wrapped__
    monkeypatch.setattr(native.RDNAHybridW4A16LinearKernel, "process_weights_after_loading", stock)
    quant = auto_gptq.AutoGPTQConfig(4, 128, False, True, False, {}, {"autoround_version": "0.15.1"})

    def factory():
        method = auto_gptq.AutoGPTQLinearMethod(quant)
        with torch.device("meta"):
            layer = torch.nn.Module()
            method.create_weights(layer, 4096, [4096, 1024, 1024], 4096, 6144,
                                  torch.bfloat16, weight_loader=lambda *args: None)
        assert layer.qzeros.shape == (32, 768)
        assert method.kernel.config.zero_points is False
        return method, layer

    method, broken = factory()
    method.process_weights_after_loading(broken)
    old_zeros = method.kernel._get_weight_params(broken)[2]
    # Also tolerate a future upstream fix which already removes these zeros.
    if old_zeros is not None:
        assert old_zeros.shape == (32, 768)
        assert old_zeros.shape != (6144, 32)
    apply = native.RDNAHybridW4A16LinearKernel.apply_weights
    local_llm_rdna.register()
    method, layer = factory()
    method.process_weights_after_loading(layer)
    weights, scales, zeros, _ = method.kernel._get_weight_params(layer)
    assert weights.shape == (6144, 2048) and weights.dtype == torch.int8
    assert scales.shape == (6144, 32) and scales.dtype == torch.bfloat16
    assert zeros is None and layer.qzeros is None
    assert layer._parameters["qzeros"] is None
    assert weights.device.type == scales.device.type == "meta"
    assert native.RDNAHybridW4A16LinearKernel.apply_weights is apply


def test_real_asymmetric_meta_conversion_preserves_processed_zero_points(monkeypatch):
    torch, parameter, native = real_modules()
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
    from vllm.scalar_type import scalar_types

    patch_parallel_rank(monkeypatch, parameter)
    local_llm_rdna.register()
    config = MPLinearLayerConfig((256, 64), (256, 64), scalar_types.uint4,
                                 torch.bfloat16, 32, True, False)
    kernel = native.RDNAHybridW4A16LinearKernel(config, "weight_packed", "weight_scale", "qzeros")
    with torch.device("meta"):
        layer = torch.nn.Module()
        layer.register_parameter("weight_packed", parameter.PackedvLLMParameter(
            data=torch.empty((64, 32), dtype=torch.int32), input_dim=1, output_dim=0,
            packed_dim=1, packed_factor=8, weight_loader=lambda *args: None))
        layer.register_parameter("weight_scale", parameter.GroupQuantScaleParameter(
            data=torch.empty((64, 8), dtype=torch.bfloat16), input_dim=1, output_dim=0,
            weight_loader=lambda *args: None))
        layer.register_parameter("qzeros", parameter.PackedvLLMParameter(
            data=torch.empty((8, 8), dtype=torch.int32), input_dim=1, output_dim=0,
            packed_dim=0, packed_factor=8, weight_loader=lambda *args: None))
    kernel.process_weights_after_loading(layer)
    zeros = kernel._get_weight_params(layer)[2]
    assert zeros is layer.qzeros
    assert zeros.shape == (64, 8) and zeros.dtype == torch.bfloat16
    assert zeros.device.type == "meta"
