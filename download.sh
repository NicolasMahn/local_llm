#!/usr/bin/env bash
# Download the model weights into the Hugging Face cache. Safe to re-run:
# finished files are skipped and partial ones resume.
set -u
for repo in RedHatAI/Qwen3.8-27B-INT4 cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4; do
  echo "START $repo $(date -Is)"
  hf download "$repo" >/dev/null && echo "DONE $repo $(date -Is)" || echo "FAIL $repo $(date -Is)"
done
