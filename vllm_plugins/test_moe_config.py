import json
import tempfile
import unittest
from pathlib import Path

from alias_moe_config import PROFILE, install_alias


class MoEConfigTest(unittest.TestCase):
    def test_legacy_profile_becomes_reachable_under_current_device_name(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            profile = {str(batch): {"BLOCK_SIZE_M": 16} for batch in (1, 2, 4, 8, 16, 32)}
            source = configs / PROFILE.format("Radeon_8060S_Graphics")
            source.write_text(json.dumps(profile))
            install_alias(configs)
            self.assertEqual((configs / PROFILE.format("AMD_Radeon_8060S")).read_bytes(), source.read_bytes())

    def test_future_native_profile_takes_precedence_even_without_legacy_file(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            target = configs / PROFILE.format("AMD_Radeon_8060S")
            target.write_text('{"1": {"native": true}}')
            install_alias(configs)
            self.assertEqual(json.loads(target.read_text()), {"1": {"native": True}})

    def test_missing_or_incomplete_source_fails_build(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            with self.assertRaises(FileNotFoundError):
                install_alias(configs)
            (configs / PROFILE.format("Radeon_8060S_Graphics")).write_text('{"1": {}}')
            with self.assertRaises(ValueError):
                install_alias(configs)
            self.assertFalse((configs / PROFILE.format("AMD_Radeon_8060S")).exists())


if __name__ == "__main__":
    unittest.main()
