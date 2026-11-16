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
# for Gemma 4 (README, "Notes"). The patch gives full-attention prefill larger
# kernel tiles (Qwen, Granite, Nemotron, Olmo, Muse); it fails the build if
# vLLM's code moved, rather than leaving the slow kernel in place.
# Audio decoding for Gemma 4 12B, which hears: vLLM has the model code but
# not the libraries that read WAV/FLAC (soundfile) and MP3/M4A/WebM (av), so
# every audio request failed. Not "vllm[audio]", which can replace the Strix
# build of vLLM from PyPI. The import fails the build rather than the requests.
RUN pip install --no-cache-dir soundfile av && python3 -c "import soundfile, av"

COPY vllm_plugins /opt/local-llm-plugins
# The Omni expert tuning shipped in the base image uses an older 8060S name;
# Gemma 26B needs its own profile. Install both during the build.
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/local-llm-plugins \
 && git -C /opt/venv/lib/python3.12/site-packages apply /opt/local-llm-plugins/prefill-tiles.patch \
 && git -C /opt/venv/lib/python3.12/site-packages apply /opt/local-llm-plugins/pixtral-positions.patch \
 && python3 /opt/local-llm-plugins/alias_moe_config.py

# A symmetric GPTQ export still contains packed qzeros. The RDNA kernel must
# discard that unused parameter after conversion, before its first profile run.
RUN PYTHONPATH=/opt/local-llm-plugins python3 -m pytest -q -p no:cacheprovider /opt/local-llm-plugins/test_rdna_gptq.py
LABEL local-llm.rdna-symmetric-gptq="1"

# Olmo opens its thought in the prompt. Count that thought and preserve the
# same reasoning/final split when a nonstream response reaches its token cap.
RUN VLLM_TARGET_DEVICE=cpu PYTHONPATH=/opt/local-llm-plugins python3 -m pytest -q -p no:cacheprovider /opt/local-llm-plugins/test_olmo_reasoning.py
LABEL local-llm.olmo-reasoning="3"

# vLLM 0.29 imports the Transformers 4 Pixtral rotation, which Transformers 5
# replaced; without the patch Mistral Small 3.2 and Ministral 3 cannot start.
RUN VLLM_TARGET_DEVICE=cpu PYTHONPATH=/opt/local-llm-plugins python3 -m pytest -q -p no:cacheprovider /opt/local-llm-plugins/test_pixtral_positions.py
LABEL local-llm.pixtral-positions="1"
