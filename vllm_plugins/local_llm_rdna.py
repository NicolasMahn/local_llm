"""Keep unused GPTQ zero points out of the symmetric RDNA apply operation."""

from functools import wraps


def register():
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        RDNAHybridW4A16LinearKernel,
    )

    stock = RDNAHybridW4A16LinearKernel.process_weights_after_loading
    if getattr(stock, "_local_llm_rdna_symmetric_zp", False):
        return

    @wraps(stock)
    def process_weights_after_loading(self, layer):
        result = stock(self, layer)
        # AutoGPTQ registers raw qzeros even for biased unsigned symmetric
        # weights. RDNA's post-load conversion skips them in that case, but
        # its apply method still reads the named parameter. Clear it before
        # compilation so the native operation receives its intended None.
        name = self.w_zp_name
        if self.config.zero_points is False and name is not None and getattr(layer, name, None) is not None:
            setattr(layer, name, None)
        return result

    process_weights_after_loading._local_llm_rdna_symmetric_zp = True
    RDNAHybridW4A16LinearKernel.process_weights_after_loading = process_weights_after_loading
