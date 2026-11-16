"""The patched Pixtral encoder must rotate each image patch exactly as before.

Runs at image build, on the CPU. Mistral Small 3.2 and Ministral 3 read
images through this encoder; wrong positions would not fail, they would
quietly blur where things are in a picture.
"""

import ast
import importlib.util
from pathlib import Path

import torch
from transformers import PixtralVisionConfig


def patched_pixtral():
    """The installed module's Pixtral import and position function, without
    the rest of it: importing the module needs a GPU, a build has none."""
    path = Path(importlib.util.find_spec("vllm").origin).parent / "model_executor/models/pixtral.py"
    tree = ast.parse(path.read_text())
    keep = [node for node in tree.body
            if (isinstance(node, ast.ImportFrom) and node.module == "transformers.models.pixtral.modeling_pixtral")
            or (isinstance(node, ast.FunctionDef) and node.name == "position_ids_in_meshgrid")]
    assert len(keep) == 2, "pixtral-positions.patch is not applied"
    namespace = {"torch": torch}
    exec(compile(ast.Module(keep, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["PixtralRotaryEmbedding"], namespace["position_ids_in_meshgrid"]


def transformers4_angles(config, flat_ids):
    """Transformers 4's PixtralRotaryEmbedding, indexed by row * max_width + column."""
    dim, side = config.head_dim, config.image_size // config.patch_size
    freqs = 1.0 / (config.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim))
    rows = torch.outer(torch.arange(side), freqs[::2]).float()
    columns = torch.outer(torch.arange(side), freqs[1::2]).float()
    grid = torch.cat([rows[:, None, :].repeat(1, side, 1), columns[None, :, :].repeat(side, 1, 1)],
                     dim=-1).reshape(-1, dim // 2)
    return torch.cat((grid, grid), dim=-1)[flat_ids]


def test_patched_positions_give_the_transformers4_rotation_for_every_image():
    PixtralRotaryEmbedding, position_ids_in_meshgrid = patched_pixtral()

    config = PixtralVisionConfig()
    side = config.image_size // config.patch_size
    images = [torch.zeros(1, 8, 5, 7), torch.zeros(1, 8, 3, 2)]  # two sizes, packed together
    positions = position_ids_in_meshgrid(images, max_width=side)
    flat = torch.cat([(torch.arange(h)[:, None] * side + torch.arange(w)).flatten()
                      for h, w in ((5, 7), (3, 2))])
    cos, sin = PixtralRotaryEmbedding(config, None)(torch.zeros(1, len(flat), 8), positions)
    angles = transformers4_angles(config, flat)
    torch.testing.assert_close(cos, angles.cos(), rtol=0, atol=1e-6)
    torch.testing.assert_close(sin, angles.sin(), rtol=0, atol=1e-6)
