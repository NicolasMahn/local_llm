"""Exercise the real vLLM operation hooks without loading model instances."""

import argparse
import json
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from dense_compare import check_reference, dequantize, measure
from local_llm_dense import register as register_dense
from local_llm_encoder import register as register_encoder
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as native


def dense():
    torch.manual_seed(137)
    m, k, n, group = 4, 21504, 5376, 32
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    packed = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), device="cuda", dtype=torch.int32)
    packed_bytes = packed.view(torch.int8)
    scales = (0.003 + torch.rand((n, k // group), device="cuda") * 0.02).to(x.dtype)
    bias = (torch.randn((n,), device="cuda") * 0.05).to(x.dtype)
    cus = torch.cuda.get_device_properties(0).multi_processor_count

    def operation(activations):
        return torch.ops.vllm.rdna_hybrid_w4a16_apply(
            activations, packed_bytes, scales, None, bias, cus, group,
        )

    baseline = operation(x)
    hip_baseline = operation(x[:1])
    large_x = torch.randn((32, k), device="cuda", dtype=x.dtype)
    fallback_baseline = operation(large_x)
    before_ms = measure(lambda: operation(x))
    original = native.triton_w4a16_skinny_fmt_gemm
    register_dense()
    registered = native.triton_w4a16_skinny_fmt_gemm
    assert registered is not original and registered._local_llm_dense
    register_dense()
    assert native.triton_w4a16_skinny_fmt_gemm is registered
    actual = operation(x)
    torch.testing.assert_close(actual, baseline, rtol=0.01, atol=0.005)
    torch.testing.assert_close(operation(x[:1]), hip_baseline, rtol=0, atol=0)
    torch.testing.assert_close(operation(large_x), fallback_baseline, rtol=0, atol=0)
    # Bias remains a separate native step, with its original BF16 rounding.
    sample = x.float() @ dequantize(packed[:64], scales[:64], group, x.dtype).float().T
    sample = (sample.to(x.dtype).float() + bias[:64].float()).to(x.dtype).float()
    errors = check_reference(actual[:, :64], sample)
    with FakeTensorMode() as mode:
        fake_out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
            mode.from_tensor(x), mode.from_tensor(packed_bytes),
            mode.from_tensor(scales), None, mode.from_tensor(bias), cus, group,
        )
        assert fake_out.shape == (m, n) and fake_out.dtype == x.dtype
    # measure() captures and replays the real torch.ops route, including bias.
    after_ms = measure(lambda: operation(x))
    print(json.dumps({"case": "dense_torch_operation", "before_ms": before_ms,
                      "after_ms": after_ms, "bias_preserved": True, "fake_impl_preserved": True,
                      "hip_and_unmeasured_fallback_bitwise": True, "reference_error": errors}), flush=True)


def encoder():
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3Omni_VisionTransformer, Qwen3OmniMoeAudioEncoder,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    context = SimpleNamespace(attn_backend=AttentionBackendEnum.FLASH_ATTN)
    cu = torch.tensor([0, 64, 168], device="cuda", dtype=torch.int32)
    classes = (Qwen3Omni_VisionTransformer, Qwen3OmniMoeAudioEncoder)
    originals = [cls.compute_attn_mask_seqlen for cls in classes]
    expected = [method(context, cu) for method in originals]
    register_encoder()
    for cls, stock, bound in zip(classes, originals, expected):
        method = cls.compute_attn_mask_seqlen
        assert method is not stock and method._local_llm_encoder_bound
        result = method(context, cu)
        assert result.device.type == "cpu" and result.dtype == bound.dtype
        torch.testing.assert_close(result, bound.cpu(), rtol=0, atol=0)
        # Warm both the reduction and add before capture.
        _ = stock(context, cu) + 1
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method(context, cu)
            plus_one = captured + 1
        assert captured.device.type == "cuda"
        graph.replay()
        torch.cuda.synchronize()
        assert plus_one.item() == 105
        register_encoder()
        assert cls.compute_attn_mask_seqlen is method
    print(json.dumps({"case": "registered_omni_encoder_methods", "classes": len(classes),
                      "metadata_equal": True, "actual_capture_fallback": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("dense", "encoder", "both"), default="both")
    args = parser.parse_args()
    if args.case in ("dense", "both"):
        dense()
    if args.case in ("encoder", "both"):
        encoder()


if __name__ == "__main__":
    main()
