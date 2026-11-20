import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "calibrate_thresholds.py"
SPEC = importlib.util.spec_from_file_location("calibrate_thresholds", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CalibrationTests(unittest.TestCase):
    def test_separable_samples_produce_calibrated_threshold(self):
        samples = [(0.80, 1), (0.75, 1), (0.70, 1), (0.30, 0), (0.25, 0), (0.20, 0)]
        report = MODULE.calibrate_metric(samples, min_per_class=3)
        self.assertEqual(report["status"], "calibrated")
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertGreater(report["threshold"], 0.30)
        self.assertLess(report["threshold"], 0.70)

    def test_insufficient_human_labels_remain_uncalibrated(self):
        report = MODULE.calibrate_metric([(0.8, 1), (0.2, 0)], min_per_class=2)
        self.assertEqual(report["status"], "uncalibrated")

    def test_report_does_not_claim_probability(self):
        rows = [
            {"human_recognizable": "1", "sface": "0.8"},
            {"human_recognizable": "1", "sface": "0.7"},
            {"human_recognizable": "0", "sface": "0.3"},
            {"human_recognizable": "0", "sface": "0.2"},
        ]
        report = MODULE.calibrate_rows(rows, metrics=["sface"], min_per_class=2)
        self.assertIn("不是通用的人类识别概率", report["warning"])


if __name__ == "__main__":
    unittest.main()
