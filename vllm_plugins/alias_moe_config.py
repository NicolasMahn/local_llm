"""Install the 8060S expert profiles without overriding newer native tuning."""

import importlib.util
import json
from pathlib import Path
from shutil import copyfile

PROFILE = "E=128,N=768,device_name={},dtype=int4_w4a16.json"
GEMMA_PROFILE = "E=128,N=704,device_name=AMD_Radeon_8060S,dtype=int4_w4a16.json"


def install_alias(configs: Path):
    target = configs / PROFILE.format("AMD_Radeon_8060S")
    source = configs / PROFILE.format("Radeon_8060S_Graphics")
    _copy_profile(source, target, (1, 2, 4, 8, 16, 32))


def _copy_profile(source: Path, target: Path, batches):
    if target.exists():
        return  # A native profile from a newer image takes precedence.
    profile = json.loads(source.read_text())
    if not all(str(batch) in profile for batch in batches):
        raise ValueError(f"Incomplete 8060S tuning profile: {source}")
    copyfile(source, target)


def install_configs(configs: Path, local_configs: Path | None = None):
    install_alias(configs)
    local_configs = local_configs if local_configs is not None else Path(__file__).with_name("moe_configs")
    # vLLM chooses the nearest batch key. Both sides of 40/41 must exist so
    # decode tuning cannot spill into prefill, where it regressed Gemma.
    _copy_profile(local_configs / GEMMA_PROFILE, configs / GEMMA_PROFILE,
                  (1, 2, 4, 8, 16, 32, 40, 41, 512))


if __name__ == "__main__":
    package = Path(importlib.util.find_spec("vllm").origin).parent
    install_configs(package / "model_executor/layers/fused_moe/configs")
