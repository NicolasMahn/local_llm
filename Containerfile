# The Strix Halo vLLM build, plus the audio extras speech-to-text needs
# (vllm[audio]; installed piecemeal so pip leaves the patched vLLM alone).
FROM docker.io/kyuz0/vllm-therock-gfx1151:latest
RUN pip install --no-cache-dir av scipy soundfile soxr "mistral_common[audio]"

# The base image ships PyTorch's AOTriton library without its precompiled
# kernels, so PyTorch's own attention (used by Gemma's image encoder) fails
# with "invalid argument". The version must match libaotriton_v2.so.
ARG AOTRITON=0.13.50tp
RUN curl -fsSL "https://github.com/ROCm/aotriton/releases/download/${AOTRITON}/aotriton-${AOTRITON}-images-amd-gfx115x.tar.gz" \
    | tar xz --strip-components=2 -C /opt/venv/lib/python3.12/site-packages/torch/lib aotriton/lib/aotriton.images

# Long-context prefill attention (README, "When things go wrong"): a plugin
# vLLM loads in every model container for Gemma 4, and larger kernel tiles for
# Qwen. The patch fails the build if vLLM's code moved, rather than leaving the
# slow kernel in place.
COPY attention /opt/local-llm-attention
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/local-llm-attention \
 && git -C /opt/venv/lib/python3.12/site-packages apply /opt/local-llm-attention/prefill-tiles.patch
