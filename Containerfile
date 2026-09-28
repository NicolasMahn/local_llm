# The Strix Halo vLLM build.
FROM docker.io/kyuz0/vllm-therock-gfx1151:latest

# The base image ships PyTorch's AOTriton library without its precompiled
# kernels, so PyTorch's own attention (used by Gemma's image encoder) fails
# with "invalid argument". The version must match libaotriton_v2.so.
ARG AOTRITON=0.13.50tp
RUN curl -fsSL "https://github.com/ROCm/aotriton/releases/download/${AOTRITON}/aotriton-${AOTRITON}-images-amd-gfx115x.tar.gz" \
    | tar xz --strip-components=2 -C /opt/venv/lib/python3.12/site-packages/torch/lib aotriton/lib/aotriton.images

# vLLM plugins, loaded in every model container: long-context prefill
# attention for Gemma 4 (README, "When things go wrong") and forced tool calls
# for Gemma 4 (README, "Notes"). The patch gives Qwen larger kernel tiles; it
# fails the build if vLLM's code moved, rather than leaving the slow kernel in
# place.
COPY vllm_plugins /opt/local-llm-plugins
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/local-llm-plugins \
 && git -C /opt/venv/lib/python3.12/site-packages apply /opt/local-llm-plugins/prefill-tiles.patch
