import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from identity_lock import build_identity_profile, derive_face_metrics, evaluate_attention


REFERENCE = build_identity_profile(
    {"eye_distance": 0.40, "face_ratio": 1.30, "nose_to_eye": 0.28}
)


def candidate(eye=0.40, face=1.30, nose=0.28, lateral=0.02, tilt=0.02):
    return {
        "eye_distance": eye,
        "face_ratio": face,
        "nose_to_eye": nose,
        "lateral_asymmetry": lateral,
        "eye_line_tilt": tilt,
        "detector_confidence": 0.99,
    }


class AttentionPolicyTests(unittest.TestCase):
    def test_reliable_anchor_drift_rejects_candidate(self):
        report = evaluate_attention(REFERENCE, candidate(eye=0.45), 0.04)
        self.assertEqual(report["decision"], "reject")
        self.assertEqual(report["next_focus"][0]["key"], "eye_distance")
        self.assertTrue(report["reject_reasons"])

    def test_low_observability_is_not_identity_rejection(self):
        report = evaluate_attention(REFERENCE, candidate(eye=0.48, lateral=0.16), 0.004)
        self.assertEqual(report["decision"], "unreliable")
        self.assertEqual(report["reject_reasons"], [])

    def test_only_top_deviation_consumes_repair_budget(self):
        report = evaluate_attention(
            REFERENCE,
            candidate(eye=0.43, face=1.44, nose=0.34),
            0.04,
            model_disagreement=0.2,
        )
        self.assertEqual(len(report["next_focus"]), 1)

    def test_matched_candidate_has_no_repair_instruction(self):
        report = evaluate_attention(REFERENCE, candidate(), 0.04)
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["next_focus"], [])
        self.assertGreater(report["identity_attention_score"], 0.99)

    def test_dense_advisory_can_focus_but_cannot_reject(self):
        reference = build_identity_profile({
            "eye_distance": 0.40,
            "face_ratio": 1.30,
            "nose_to_eye": 0.28,
            "dense_jaw_width_ratio": 0.70,
        })
        values = candidate()
        values.update({"dense_jaw_width_ratio": 0.80, "dense_presence": 0.99})
        report = evaluate_attention(reference, values, 0.04)
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["next_focus"][0]["key"], "dense_jaw_width_ratio")
        self.assertEqual(report["next_focus"][0]["evidence_mode"], "advisory")
        self.assertEqual(report["reject_reasons"], [])
        self.assertTrue(report["advisory_repair_recommended"])

    def test_missing_candidate_dense_metrics_falls_back_to_base(self):
        reference = build_identity_profile({
            "eye_distance": 0.40,
            "face_ratio": 1.30,
            "nose_to_eye": 0.28,
            "dense_nose_width_ratio": 0.25,
        })
        report = evaluate_attention(reference, candidate(), 0.04)
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["feature_sources"]["facemesh_468"], 0)

    def test_focus_budget_never_emits_multiple_face_regions(self):
        reference = build_identity_profile({
            "eye_distance": 0.40,
            "face_ratio": 1.30,
            "nose_to_eye": 0.28,
            "dense_intercanthal_ratio": 0.20,
            "dense_eye_width_ratio": 0.20,
            "dense_nose_width_ratio": 0.25,
        })
        values = candidate()
        values.update({
            "dense_intercanthal_ratio": 0.24,
            "dense_eye_width_ratio": 0.25,
            "dense_nose_width_ratio": 0.30,
            "dense_presence": 0.99,
        })
        report = evaluate_attention(reference, values, 0.04)
        self.assertEqual(len(report["next_focus"]), 1)

    def test_existing_yunet_mouth_points_become_advisory_evidence(self):
        reference_face = [0, 0, 100, 130, 30, 40, 70, 40, 50, 65, 38, 90, 62, 90, 0.99]
        candidate_face = [0, 0, 100, 130, 30, 40, 70, 40, 50, 65, 30, 90, 70, 90, 0.99]
        reference_metrics = derive_face_metrics(reference_face)
        candidate_metrics = derive_face_metrics(candidate_face)
        profile = build_identity_profile(reference_metrics)
        report = evaluate_attention(profile, candidate_metrics, 0.04)
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["next_focus"][0]["key"], "mouth_width")
        self.assertEqual(report["next_focus"][0]["evidence_mode"], "advisory")

    def test_style_identity_conflict_overrides_local_feature_repairs(self):
        report = evaluate_attention(
            REFERENCE,
            candidate(),
            0.04,
            identity_conflict={
                "active": True,
                "source_similarity": 0.86,
                "style_similarity": 0.84,
                "source_lead": 0.02,
            },
        )
        self.assertEqual(report["decision"], "reject")
        self.assertEqual(report["next_focus"][0]["key"], "identity_source_conflict")
        self.assertEqual(report["feature_sources"]["identity_comparator"], 1)


if __name__ == "__main__":
    unittest.main()
