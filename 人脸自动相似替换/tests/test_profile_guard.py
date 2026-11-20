import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ai人像摄影技能包"
    / "scripts"
    / "profile_guard.py"
)
SPEC = importlib.util.spec_from_file_location("profile_guard_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProfileGuardTests(unittest.TestCase):
    def test_validated_profile_is_intact_and_golden_is_regression_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "人物_特征.md"
            reference = root / "真人.jpg"
            golden = root / "golden.png"
            profile.write_text("基线锁脸段成品：P0/P1/P2", encoding="utf-8")
            reference.write_bytes(b"\xff\xd8\xffreference")
            golden.write_bytes(b"\x89PNG\r\n\x1a\ngolden")

            metadata = MODULE.validate_profile(profile, [reference], golden, "human ok")
            report = MODULE.check_profile(profile)

            self.assertEqual(metadata["status"], "validated")
            self.assertTrue(report["profile_intact"])
            self.assertEqual(metadata["references"][0]["role"], "real_identity_source")
            self.assertEqual(metadata["golden_regression"]["role"], "regression_only")
            self.assertFalse(metadata["golden_regression"]["identity_source"])

    def test_validated_profile_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "人物_特征.md"
            reference = root / "真人.jpg"
            profile.write_text("已验证档案", encoding="utf-8")
            reference.write_bytes(b"\xff\xd8\xffreference")
            MODULE.validate_profile(profile, [reference], None)

            with self.assertRaisesRegex(ValueError, "validated"):
                MODULE.guard_output(profile, replace_provisional=True)

    def test_profile_hash_change_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "人物_特征.md"
            reference = root / "真人.jpg"
            profile.write_text("已验证档案", encoding="utf-8")
            reference.write_bytes(b"\xff\xd8\xffreference")
            MODULE.validate_profile(profile, [reference], None)
            profile.write_text("被修改的档案", encoding="utf-8")

            self.assertFalse(MODULE.check_profile(profile)["ok"])


if __name__ == "__main__":
    unittest.main()
