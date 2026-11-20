import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from candidate_quality import evaluate_image_integrity


class CandidateQualityTests(unittest.TestCase):
    def test_normal_image_passes(self):
        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        image[120:390, 160:350] = (80, 130, 190)
        report = evaluate_image_integrity(image)
        self.assertTrue(report["passed"])
        self.assertEqual(report["reasons"], [])

    def test_uniform_black_or_white_image_fails(self):
        for value in (0, 255):
            with self.subTest(value=value):
                report = evaluate_image_integrity(
                    np.full((512, 512, 3), value, dtype=np.uint8)
                )
                self.assertFalse(report["passed"])
                self.assertIn("画面近乎空白", "；".join(report["reasons"]))

    def test_small_image_fails(self):
        image = np.full((128, 128, 3), 120, dtype=np.uint8)
        image[32:96, 32:96] = 180
        report = evaluate_image_integrity(image)
        self.assertFalse(report["passed"])
        self.assertIn("尺寸过小", "；".join(report["reasons"]))

    def test_large_black_bottom_band_fails(self):
        image = np.full((1200, 800, 3), 150, dtype=np.uint8)
        image[200:900, 180:620] = (90, 140, 200)
        image[-100:, :] = 0
        report = evaluate_image_integrity(image)
        self.assertFalse(report["passed"])
        self.assertEqual(report["black_edge_bands"]["bottom"], 100)
        self.assertIn("连续纯黑边", "；".join(report["reasons"]))


if __name__ == "__main__":
    unittest.main()
