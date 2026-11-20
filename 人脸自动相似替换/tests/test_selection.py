import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from identity_lock import select_final_candidate
from face_guard import finalize_command


def candidate(
    path: str,
    *,
    source: float,
    attention: float,
    spread: float,
    passed: bool = False,
) -> dict:
    return {
        "candidate": path,
        "source_similarity": source,
        "identity_model_spread": spread,
        "passed": passed,
        "gates": {
            "observability": True,
            "embedding_identity": passed,
            "identity_source": True,
            "anchor": True,
        },
        "attention_priority": {
            "identity_attention_score": attention,
            "decision": "pass",
        },
    }


class FinalSelectionTests(unittest.TestCase):
    def setUp(self):
        # 真实试跑的冲突形态：第二张局部分更高，但状态机因整体身份退化保留第一张。
        self.roll_1 = candidate(
            "/session/candidates/roll_01.png",
            source=0.450610,
            attention=0.469499,
            spread=0.037430,
        )
        self.roll_2 = candidate(
            "/session/candidates/roll_02.png",
            source=0.499446,
            attention=0.516914,
            spread=0.103112,
        )

    def test_iteration_best_is_authoritative_over_second_ranking(self):
        result = select_final_candidate(
            [self.roll_1, self.roll_2],
            iteration_state={"best": {"candidate": self.roll_1["candidate"]}},
        )
        self.assertEqual(result["selected"]["candidate"], self.roll_1["candidate"])
        self.assertEqual(result["selection_source"], "iteration_state")

    def test_manual_visual_preference_remains_highest_priority(self):
        result = select_final_candidate(
            [self.roll_1, self.roll_2],
            iteration_state={"best": {"candidate": self.roll_1["candidate"]}},
            preferred_candidate=self.roll_2["candidate"],
        )
        self.assertEqual(result["selected"]["candidate"], self.roll_2["candidate"])
        self.assertEqual(result["selection_source"], "manual_review")

    def test_stale_iteration_best_falls_back_with_reason(self):
        result = select_final_candidate(
            [self.roll_1, self.roll_2],
            iteration_state={"best": {"candidate": "/missing/roll.png"}},
        )
        self.assertIn(result["selected"], [self.roll_1, self.roll_2])
        self.assertEqual(result["selection_source"], "machine_fallback")
        self.assertIn("没有对应评分记录", result["fallback_reason"])

    def test_passed_candidate_wins_machine_fallback(self):
        passed = candidate(
            "/session/candidates/roll_03.png",
            source=0.91,
            attention=0.70,
            spread=0.02,
            passed=True,
        )
        result = select_final_candidate([self.roll_1, passed])
        self.assertEqual(result["selected"]["candidate"], passed["candidate"])

    def test_quality_failed_candidate_never_enters_selection_pool(self):
        broken = candidate(
            "/session/candidates/broken.png",
            source=0.99,
            attention=0.99,
            spread=0.01,
            passed=False,
        )
        broken["gates"]["quality"] = False
        result = select_final_candidate([self.roll_1, broken])
        self.assertEqual(result["selected"]["candidate"], self.roll_1["candidate"])

        with self.assertRaisesRegex(ValueError, "未通过画面完整性"):
            select_final_candidate(
                [self.roll_1, broken], preferred_candidate=broken["candidate"]
            )

    def test_all_unscorable_candidates_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "全部候选都无法检测人脸"):
            select_final_candidate([
                {"candidate": "bad.png", "source_similarity": None, "passed": False}
            ])

    def test_finalize_copies_iteration_best_and_reports_unpassed(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            candidates = session / "candidates"
            candidates.mkdir()
            roll_1 = candidates / "roll_01.png"
            roll_2 = candidates / "roll_02.png"
            roll_1.write_bytes(b"iteration-best")
            roll_2.write_bytes(b"higher-single-score")

            score_1 = dict(self.roll_1, candidate=str(roll_1))
            score_2 = dict(self.roll_2, candidate=str(roll_2))
            (candidates / "roll_01.png.score.json").write_text(
                json.dumps(score_1), encoding="utf-8"
            )
            (candidates / "roll_02.png.score.json").write_text(
                json.dumps(score_2), encoding="utf-8"
            )
            (session / "iteration_state.json").write_text(
                json.dumps({"best": {"candidate": str(roll_1)}}), encoding="utf-8"
            )
            job = session / "job.json"
            job.write_text(
                json.dumps({"session_dir": str(session), "style_source": {"kind": "none"}}),
                encoding="utf-8",
            )

            report = finalize_command(argparse.Namespace(job=str(job), prefer=None))

            self.assertEqual(report["selection_source"], "iteration_state")
            self.assertEqual(report["status"], "未达标，保留最佳候选")
            self.assertEqual(Path(report["final_image"]).read_bytes(), b"iteration-best")


if __name__ == "__main__":
    unittest.main()
