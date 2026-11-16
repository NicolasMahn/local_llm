"""One startup switch for the measured 8060S kernel and encoder improvements."""

import os


def register():
    if os.environ.get("LOCAL_LLM_FAST_KERNELS", "1") == "0":
        return
    from local_llm_dense import register as dense
    from local_llm_encoder import register as encoder
    from local_llm_moe import register as experts

    dense()
    encoder()
    experts()
