import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("nemo_moe_compare", Path(__file__).parent / "benchmarks/nemo_moe_compare.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def valid_config():
    return {"architectures": ["NemotronHForCausalLM"], "hidden_size": 2688,
            "moe_intermediate_size": 1856, "n_routed_experts": 128, "num_experts_per_tok": 6,
            "mlp_hidden_act": "relu2", "routed_scaling_factor": 2.5,
            "layers_block_type": ["mamba", "moe"], "quantization_config": {
                "quant_method": "compressed-tensors", "transform_config": None, "config_groups": {
                    "group_0": {"weights": {"num_bits": 4, "symmetric": True, "group_size": 64,
                                             "actorder": "static", "type": "int"}}}}}


def test_exact_non_gated_checkpoint_contract_is_accepted():
    probe.validate_checkpoint(valid_config(), 1)


def test_empty_checkpoint_transform_configuration_is_equivalent_to_none():
    config = valid_config()
    config["quantization_config"]["transform_config"] = {}
    probe.validate_checkpoint(config, 1)


def test_nonempty_checkpoint_transform_configuration_is_not_silently_ignored():
    config = valid_config()
    config["quantization_config"]["transform_config"] = {"unexpected_transform": {}}
    with pytest.raises(ValueError, match="group64"):
        probe.validate_checkpoint(config, 1)


@pytest.mark.parametrize("key,value", [("hidden_size", 2048), ("num_experts_per_tok", 8),
                                       ("mlp_hidden_act", "silu"), ("moe_intermediate_size", 768)])
def test_other_models_cannot_silently_reuse_the_probe(key, value):
    config = valid_config()
    config[key] = value
    with pytest.raises(ValueError, match="expert contract"):
        probe.validate_checkpoint(config, 1)


@pytest.mark.parametrize("key,value", [("group_size", 32), ("actorder", "group"), ("symmetric", False)])
def test_other_quantization_layouts_are_rejected(key, value):
    config = valid_config()
    config["quantization_config"]["config_groups"]["group_0"]["weights"][key] = value
    with pytest.raises(ValueError, match="group64"):
        probe.validate_checkpoint(config, 1)


def test_candidate_grid_is_small_and_preserves_unsplit_native_routing():
    grid = probe.candidates()
    assert len(grid) == 8 and len({tuple(sorted(config.items())) for _, config in grid}) == 8
    for _, config in grid:
        assert config["SPLIT_K"] == config["GROUP_SIZE_M"] == 1
        assert config["BLOCK_SIZE_M"] in (16, 32, 64)
        assert config["BLOCK_SIZE_K"] in (64, 128, 256)


def test_cpu_reference_has_non_gated_relu_squared_and_group64_boundaries():
    pytest.importorskip("torch")
    probe.cpu_self_test()


def test_small_wrong_outputs_cannot_hide_inside_the_absolute_tolerance():
    torch = pytest.importorskip("torch")
    expected = torch.full((2, 4), 0.00001, dtype=torch.bfloat16)
    wrong = expected * 2
    torch.testing.assert_close(wrong, expected, **probe.TOLERANCE)
    with pytest.raises(AssertionError, match="relative L2 error"):
        probe.check(wrong, expected)
    exact = probe.check(expected, expected)
    assert exact["relative_l2_error"] == 0
    assert exact["relative_l2_limit"] == 0.01


class FakeSafetensors:
    def __init__(self, values):
        self.values = values
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def get_tensor(self, name):
        return self.values[name]


def test_single_file_checkpoint_uses_one_cached_cpu_handle_without_an_index(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"stand-in")
    first, second = object(), object()
    source = FakeSafetensors({"up.weight_packed": first, "up.weight_scale": second})
    opened = []

    def opener(path, **kwargs):
        opened.append((path, kwargs))
        return source

    with probe.checkpoint_reader(tmp_path, opener) as tensor:
        assert tensor("up.weight_packed") is first
        assert tensor("up.weight_scale") is second
        assert not source.closed
    assert source.closed
    assert opened == [(tmp_path / "model.safetensors", {"framework": "pt", "device": "cpu"})]


def test_sharded_checkpoints_still_select_and_cache_the_indexed_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"up": "one.safetensors", "down": "two.safetensors"}}))
    for filename in ("one.safetensors", "two.safetensors"):
        (tmp_path / filename).write_bytes(b"stand-in")
    sources = {"one.safetensors": FakeSafetensors({"up": object()}),
               "two.safetensors": FakeSafetensors({"down": object()})}
    opened = []

    def opener(path, **kwargs):
        opened.append(path.name)
        return sources[path.name]

    with probe.checkpoint_reader(tmp_path, opener) as tensor:
        assert tensor("up") is tensor("up")
        assert tensor("down") is sources["two.safetensors"].values["down"]
    assert opened == ["one.safetensors", "two.safetensors"]
    assert all(source.closed for source in sources.values())


def test_reader_refuses_missing_files_and_unsafe_index_paths_before_opening(tmp_path):
    def never_open(*args, **kwargs):
        raise AssertionError("The unsafe or incomplete checkpoint must not be opened")

    with probe.checkpoint_reader(tmp_path, never_open) as tensor:
        with pytest.raises(ValueError, match="not fully downloaded"):
            tensor("up")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"up": "../outside.safetensors"}}))
    with probe.checkpoint_reader(tmp_path, never_open) as tensor:
        with pytest.raises(ValueError, match="Unsafe shard"):
            tensor("up")
