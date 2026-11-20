import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "ai人像摄影技能包" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "vision.py"
SPEC = importlib.util.spec_from_file_location("vision_under_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VisionProfileFlowTests(unittest.TestCase):
    def test_model_prompt_requests_p0_p1_p2_baseline_not_120_char_anchor(self):
        self.assertIn("基线锁脸段成品", MODULE.MODEL_PROMPT_BASE)
        self.assertIn("P0 与 P1 必须完整", MODULE.MODEL_PROMPT_BASE)
        self.assertNotIn("限制在120个中文字符以内", MODULE.MODEL_PROMPT_BASE)

    def test_split_normalizes_legacy_golden_baseline(self):
        feature, baseline = MODULE._split_lock(
            "1. 眼距偏宽\n锁脸段成品：保留P0/P1/P2，省略P3。"
        )
        self.assertEqual(feature, "1. 眼距偏宽")
        self.assertTrue(baseline.startswith("基线锁脸段成品："))

    def test_fix_prompt_forbids_full_archive_rewrite(self):
        self.assertIn("只输出一个修正重点", MODULE.FIX_PROMPT)
        self.assertIn("不输出完整锁脸段", MODULE.FIX_PROMPT)
        self.assertNotIn("修正后锁脸段：以第一张参考照片", MODULE.FIX_PROMPT)


if __name__ == "__main__":
    unittest.main()
