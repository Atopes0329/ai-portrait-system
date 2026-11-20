import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from identity_lock import derive_dense_metrics


def synthetic_face(mouth_open=0.02):
    points = [[0.0, 0.0, 0.0] for _ in range(468)]
    coordinates = {
        234: (-1.0, 0.0), 454: (1.0, 0.0),
        10: (0.0, -1.3), 152: (0.0, 1.3),
        172: (-0.7, 0.8), 397: (0.7, 0.8),
        33: (-0.6, -0.2), 133: (-0.2, -0.2),
        362: (0.2, -0.2), 263: (0.6, -0.2),
        98: (-0.25, 0.1), 327: (0.25, 0.1),
        168: (0.0, -0.5), 2: (0.0, 0.3),
        61: (-0.4, 0.6), 291: (0.4, 0.6),
        13: (0.0, 0.55), 14: (0.0, 0.55 + mouth_open),
    }
    for index, (x, y) in coordinates.items():
        points[index] = [x, y, 0.0]
    return points


class DenseMetricTests(unittest.TestCase):
    def test_extracts_normalized_identity_ratios(self):
        metrics = derive_dense_metrics(synthetic_face(), presence=0.9)
        self.assertAlmostEqual(metrics["dense_face_ratio"], 1.3)
        self.assertAlmostEqual(metrics["dense_jaw_width_ratio"], 0.7)
        self.assertAlmostEqual(metrics["dense_intercanthal_ratio"], 0.2)
        self.assertAlmostEqual(metrics["dense_eye_width_ratio"], 0.2)
        self.assertAlmostEqual(metrics["dense_nose_width_ratio"], 0.25)
        self.assertAlmostEqual(metrics["dense_mouth_width_ratio"], 0.4)
        self.assertAlmostEqual(metrics["dense_presence"], 0.9)

    def test_open_mouth_reduces_only_mouth_reliability(self):
        neutral = derive_dense_metrics(synthetic_face(mouth_open=0.02))
        open_mouth = derive_dense_metrics(synthetic_face(mouth_open=0.22))
        self.assertGreater(neutral["dense_mouth_reliability"], 0.9)
        self.assertEqual(open_mouth["dense_mouth_reliability"], 0.0)
        self.assertEqual(neutral["dense_nose_width_ratio"], open_mouth["dense_nose_width_ratio"])

    def test_rejects_incomplete_mesh(self):
        with self.assertRaises(ValueError):
            derive_dense_metrics([[0.0, 0.0, 0.0]] * 100)


if __name__ == "__main__":
    unittest.main()
