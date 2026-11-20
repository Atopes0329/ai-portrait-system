import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from face_detection_utils import merge_duplicate_faces


def face(x, y, w, h, confidence, landmark_shift=(0.0, 0.0)):
    sx, sy = landmark_shift
    points = np.array(
        [
            x + 0.32 * w + sx, y + 0.38 * h + sy,
            x + 0.68 * w + sx, y + 0.38 * h + sy,
            x + 0.50 * w + sx, y + 0.56 * h + sy,
            x + 0.38 * w + sx, y + 0.75 * h + sy,
            x + 0.62 * w + sx, y + 0.75 * h + sy,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [np.array([x, y, w, h], dtype=np.float32), points, [confidence]]
    )


class FaceDetectionUtilsTests(unittest.TestCase):
    def test_overlapping_duplicate_keeps_higher_confidence(self):
        large = face(100, 100, 500, 500, 0.68)
        smaller = face(180, 80, 300, 400, 0.81, landmark_shift=(20, 30))
        merged, audit = merge_duplicate_faces([large, smaller])
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(float(merged[0][14]), 0.81, places=4)
        self.assertEqual(audit["merged_duplicate_count"], 1)

    def test_spatially_separate_faces_are_preserved(self):
        first = face(20, 30, 220, 260, 0.9)
        second = face(500, 40, 210, 250, 0.88)
        merged, audit = merge_duplicate_faces([first, second])
        self.assertEqual(len(merged), 2)
        self.assertEqual(audit["merged_duplicate_count"], 0)

    def test_overlapping_boxes_with_far_landmarks_are_not_merged(self):
        first = face(100, 100, 500, 500, 0.9)
        second = face(180, 80, 300, 400, 0.88, landmark_shift=(300, 300))
        merged, _audit = merge_duplicate_faces([first, second])
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
