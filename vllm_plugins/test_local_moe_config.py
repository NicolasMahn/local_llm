import json
import tempfile
import unittest
from pathlib import Path

from alias_moe_config import GEMMA_PROFILE, PROFILE, install_configs


class LocalMoEConfigTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.configs = root / "installed"
        self.local = root / "local"
        self.configs.mkdir()
        self.local.mkdir()
        # Keep these tests independent of the already tested Omni alias.
        (self.configs / PROFILE.format("AMD_Radeon_8060S")).write_text("{}")
        self.profile = {
            str(batch): {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
            for batch in (1, 2, 4, 8, 16, 32, 40, 41, 512)
        }
        self.source = self.local / GEMMA_PROFILE
        self.target = self.configs / GEMMA_PROFILE

    def test_local_gemma_profile_is_installed_without_changing_its_contents(self):
        self.source.write_text(json.dumps(self.profile, indent=2) + "\n")

        install_configs(self.configs, self.local)

        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())

    def test_future_native_gemma_profile_wins_without_a_local_source(self):
        native = {"1": {"BLOCK_SIZE_M": 64}}
        self.target.write_text(json.dumps(native))

        install_configs(self.configs, self.local)

        self.assertEqual(json.loads(self.target.read_text()), native)
        self.assertFalse(self.source.exists())

    def test_invalid_local_source_cannot_leave_an_installed_gemma_profile(self):
        cases = [
            ("missing file", None, FileNotFoundError),
            ("invalid JSON", "not JSON", ValueError),
            ("missing small-batch boundary", {k: v for k, v in self.profile.items() if k != "40"}, ValueError),
            ("missing prefill boundary", {k: v for k, v in self.profile.items() if k != "41"}, ValueError),
            ("missing prefill fallback", {k: v for k, v in self.profile.items() if k != "512"}, ValueError),
        ]
        for name, source, error in cases:
            with self.subTest(name=name):
                if source is None:
                    self.source.unlink(missing_ok=True)
                else:
                    self.source.write_text(source if isinstance(source, str) else json.dumps(source))
                with self.assertRaises(error):
                    install_configs(self.configs, self.local)
                self.assertFalse(self.target.exists())

    def test_shipped_profile_preserves_generic_prefill_settings(self):
        source = Path(__file__).with_name("moe_configs") / GEMMA_PROFILE
        profile = json.loads(source.read_text())
        profile.pop("triton_version", None)
        batches = [int(key) for key in profile]
        generic = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        self.assertIn("BLOCK_SIZE_N", profile["40"])
        for batch in range(41, 513):
            # Match vLLM's nearest-key lookup, including the 40/41 boundary.
            nearest = min(batches, key=lambda key: abs(key - batch))
            self.assertEqual(profile[str(nearest)], generic, f"Prefill step {batch}")


if __name__ == "__main__":
    unittest.main()
