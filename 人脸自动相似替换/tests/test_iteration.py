import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from identity_lock import (
    advance_iteration,
    compile_prompt_with_budget,
    compose_iteration_prompt,
    extract_identity_anchor,
    new_iteration_state,
    strip_identity_blocks,
)


def score(
    *,
    candidate="roll.png",
    source=0.91,
    passed=False,
    decision="reject",
    key="eye_distance",
    severity=1.4,
    observability=0.95,
    embedding=True,
    identity_source=True,
):
    features = []
    next_focus = []
    if key:
        feature = {
            "key": key,
            "group": "eye_nose_configuration",
            "provider": "yunet_5point",
            "evidence_mode": "gate",
            "severity": severity,
            "observability": observability,
            "priority": severity,
            "instruction": "保持本人眼距。",
        }
        features.append(feature)
        if severity >= 0.5:
            next_focus.append(feature)
    return {
        "candidate": candidate,
        "source_similarity": source,
        "passed": passed,
        "suspected_style_identity_leak": not identity_source,
        "gates": {
            "observability": observability >= 0.35,
            "embedding_identity": embedding,
            "identity_source": identity_source,
        },
        "attention_priority": {
            "decision": decision,
            "identity_attention_score": max(0.0, 1.0 - severity / 3.0),
            "features": features,
            "next_focus": next_focus,
            "advisory_repair_recommended": False,
        },
    }


class IterationStateTests(unittest.TestCase):
    def test_first_score_locks_one_focus(self):
        state = advance_iteration(new_iteration_state(), score())
        self.assertEqual(state["next_action"], "retry_focus")
        self.assertEqual(state["current_focus"]["key"], "eye_distance")

    def test_focus_does_not_advance_until_threshold_is_met(self):
        state = advance_iteration(new_iteration_state(), score(severity=1.5))
        state = advance_iteration(state, score(candidate="roll2.png", severity=1.2))
        self.assertEqual(state["next_action"], "retry_focus")
        self.assertEqual(state["current_focus"]["key"], "eye_distance")
        self.assertEqual(state["best"]["candidate"], "roll2.png")

    def test_passing_focus_completes_when_nothing_else_remains(self):
        state = advance_iteration(new_iteration_state(), score(severity=1.5))
        state = advance_iteration(
            state,
            score(candidate="roll2.png", severity=0.8, passed=True, decision="pass"),
        )
        self.assertEqual(state["status"], "completed")
        self.assertIn("eye_distance", state["completed_focus"])

    def test_identity_regression_rejects_local_improvement(self):
        state = advance_iteration(new_iteration_state(), score(source=0.92, severity=1.5))
        state = advance_iteration(
            state,
            score(candidate="roll2.png", source=0.88, severity=0.8, passed=True, decision="pass"),
        )
        self.assertEqual(state["next_action"], "retry_focus")
        self.assertEqual(state["best"]["candidate"], "roll.png")

    def test_unobservable_candidate_requests_composition_retry(self):
        state = advance_iteration(
            new_iteration_state(), score(observability=0.2, decision="unreliable")
        )
        self.assertEqual(state["next_action"], "retry_observability")
        self.assertIsNone(state["current_focus"])

    def test_failed_focus_stops_at_attempt_limit(self):
        state = advance_iteration(
            new_iteration_state(max_focus_attempts=2), score(severity=1.5)
        )
        state = advance_iteration(state, score(candidate="roll2.png", severity=1.5))
        state = advance_iteration(state, score(candidate="roll3.png", severity=1.5))
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["next_action"], "stop")

    def test_prompt_is_rebuilt_with_only_current_focus(self):
        state = advance_iteration(new_iteration_state(), score())
        prompt = compose_iteration_prompt("干净画面要求", "简短身份锚点", state)
        self.assertIn("简短身份锚点", prompt)
        self.assertIn("本轮唯一身份修正目标", prompt)
        self.assertNotIn("上一轮", prompt)
        self.assertLess(prompt.index("本轮唯一身份修正目标"), prompt.index("干净画面要求"))

    def test_explicit_identity_block_is_removed_from_scene_template(self):
        full_prompt = """【P0/P1/P2 身份基线】保持本人眼距、脸型和眼鼻关系。

【系统摄影风格】
半身肖像，暖色棚拍，no text。

质量硬约束：皮肤自然，镜框清晰。"""
        scene, audit = strip_identity_blocks(full_prompt)
        self.assertNotIn("P0/P1/P2", scene)
        self.assertIn("半身肖像", scene)
        self.assertIn("质量硬约束", scene)
        self.assertEqual(audit["removed_identity_block_count"], 1)

    def test_budget_compiler_keeps_one_baseline_and_one_focus(self):
        full_prompt = """【P0/P1/P2 身份基线】重复的完整身份段。

【系统摄影风格】
现代杂志半身肖像。

本轮唯一身份修正目标：旧重点，不得保留。

身份优先规则：旧规则，不得累积。"""
        anchor = "身份基线：眼距偏宽、鹅蛋偏圆脸、中庭略长。"
        state = {"current_focus": {"instruction": "保持本人鼻尖相对双眼的位置关系。"}}
        prompt, audit = compile_prompt_with_budget(full_prompt, anchor, state)
        self.assertEqual(prompt.count(anchor), 1)
        self.assertEqual(prompt.count("本轮唯一身份修正目标："), 1)
        self.assertNotIn("旧重点", prompt)
        self.assertNotIn("重复的完整身份段", prompt)
        self.assertIn("现代杂志半身肖像", prompt)
        self.assertEqual(audit["identity_anchor_occurrences"], 1)
        self.assertEqual(audit["focus_occurrences"], 1)
        self.assertTrue(audit["within_single_focus_budget"])

    def test_global_identity_failure_interrupts_local_focus(self):
        state = advance_iteration(new_iteration_state(), score(severity=1.5))
        state = advance_iteration(
            state,
            score(candidate="roll2.png", severity=1.2, embedding=False, source=0.70),
        )
        self.assertEqual(state["current_focus"]["key"], "global_identity")

    def test_identity_source_conflict_interrupts_local_focus(self):
        state = advance_iteration(new_iteration_state(), score(severity=1.5))
        state = advance_iteration(
            state,
            score(candidate="roll2.png", severity=1.2, identity_source=False),
        )
        self.assertEqual(state["current_focus"]["key"], "identity_source_conflict")

    def test_extracts_short_anchor_from_full_identity_archive(self):
        text = """# 特征\n\n很多内部测量\n\n身份锚点成品：偏长鹅蛋脸，眼距适中，眼鼻嘴布局紧凑。\n\n锁脸段成品：很长的十项清单"""
        anchor = extract_identity_anchor(text, "fallback")
        self.assertEqual(anchor, "身份基线：很长的十项清单")

    def test_prefers_p0_p1_p2_baseline_over_short_anchor(self):
        text = """身份锚点成品：过度压缩的短锚点。

## 首轮身份基线

基线锁脸段成品：保持眼距偏宽、鹅蛋偏圆脸和中庭略长；内双短眼裂、鼻偏长；保留黑框方眼镜与微卷碎刘海。
"""
        anchor = extract_identity_anchor(text, "fallback")
        self.assertTrue(anchor.startswith("身份基线："))
        self.assertIn("眼距偏宽", anchor)
        self.assertNotIn("过度压缩", anchor)

    def test_full_legacy_archive_falls_back_instead_of_overloading_prompt(self):
        text = "## 锁脸段成品\n" + ("全部五官描述" * 100)
        self.assertEqual(extract_identity_anchor(text, "简短默认锚点"), "简短默认锚点")

    def test_calibrated_ai_gate_can_hold_current_focus(self):
        state = advance_iteration(new_iteration_state(), score(severity=1.5))
        state = advance_iteration(
            state,
            score(candidate="roll2.png", severity=0.8, passed=True, decision="pass"),
            vision_judge={
                "hard_gate_enabled": True,
                "confidence": 0.9,
                "identity_score": 85,
                "identity_threshold": 80,
                "focus_pass": False,
                "regressions": [],
            },
        )
        self.assertEqual(state["status"], "active")
        self.assertEqual(state["current_focus"]["key"], "eye_distance")

    def test_calibrated_ai_identity_failure_interrupts_local_focus(self):
        state = advance_iteration(new_iteration_state(), score(severity=1.5))
        state = advance_iteration(
            state,
            score(candidate="roll2.png", severity=0.8, passed=True, decision="pass"),
            vision_judge={
                "hard_gate_enabled": True,
                "confidence": 0.9,
                "identity_score": 75,
                "identity_threshold": 80,
                "focus_pass": True,
                "regressions": [],
            },
        )
        self.assertEqual(state["current_focus"]["key"], "global_identity")


if __name__ == "__main__":
    unittest.main()
