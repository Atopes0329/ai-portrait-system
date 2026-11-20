"""模型无关的单重点身份修正状态机。

它不调用生图或视觉接口，只根据每轮评分决定：继续当前重点、移动到下一重点、
完成或停止。所有状态都是普通 JSON 数据，便于任何模型或 Agent 调用。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
import re

from .prompt_budget import compile_prompt_with_budget


SCHEMA_VERSION = "1.0"


def new_iteration_state(
    max_rounds: int = 6,
    max_focus_attempts: int = 3,
    min_focus_improvement: float = 0.10,
    max_identity_regression: float = 0.015,
) -> dict[str, Any]:
    if max_rounds < 1:
        raise ValueError("max_rounds 必须至少为 1")
    if max_focus_attempts < 1:
        raise ValueError("max_focus_attempts 必须至少为 1")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "identity_iteration_state",
        "status": "active",
        "round": 0,
        "max_rounds": int(max_rounds),
        "max_focus_attempts": int(max_focus_attempts),
        "min_focus_improvement": float(min_focus_improvement),
        "max_identity_regression": float(max_identity_regression),
        "current_focus": None,
        "completed_focus": [],
        "focus_attempts": 0,
        "best": None,
        "history": [],
        "stop_reason": None,
    }


def _attention(score: Mapping[str, Any]) -> Mapping[str, Any]:
    return score.get("attention_priority") or {}


def _gates(score: Mapping[str, Any]) -> dict[str, bool]:
    supplied = score.get("gates") or {}
    attention = _attention(score)
    return {
        "observability": bool(
            supplied.get("observability", attention.get("decision") != "unreliable")
        ),
        "embedding_identity": bool(
            supplied.get("embedding_identity", score.get("passed", False))
        ),
        "identity_source": bool(
            supplied.get("identity_source", not score.get("suspected_style_identity_leak", False))
        ),
    }


def _find_feature(score: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    for feature in _attention(score).get("features") or []:
        if feature.get("key") == key:
            return feature
    return None


def _eligible_focus(
    score: Mapping[str, Any], excluded: set[str] | None = None
) -> dict[str, Any] | None:
    excluded = excluded or set()
    attention = _attention(score)
    for focus in attention.get("next_focus") or []:
        key = str(focus.get("key") or "")
        if not key or key in excluded:
            continue
        if focus.get("evidence_mode") == "advisory" and not attention.get(
            "advisory_repair_recommended", False
        ):
            continue
        result = dict(focus)
        result["baseline_severity"] = float(focus.get("severity", 0.0))
        return result
    return None


def _global_focus(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": "global_identity",
        "group": "identity_global",
        "provider": "identity_ensemble",
        "evidence_mode": "gate",
        "severity": None,
        "baseline_severity": None,
        "instruction": (
            "保持图1人物的整体身份、脸型轮廓与五官空间关系；"
            "不要追加局部美化要求，不得继承图2人物身份。"
        ),
    }


def _identity_source_focus(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": "identity_source_conflict",
        "group": "identity_source",
        "provider": "identity_comparator",
        "evidence_mode": "gate",
        "severity": 2.0,
        "baseline_severity": 2.0,
        "instruction": (
            "重新强调图1是唯一身份来源，禁止从图2遮挡区域猜测或继承任何脸型和五官。"
        ),
    }


def _snapshot(score: Mapping[str, Any]) -> dict[str, Any]:
    attention_score = _attention(score).get("identity_attention_score")
    return {
        "candidate": score.get("candidate"),
        "source_similarity": score.get("source_similarity"),
        "identity_attention_score": attention_score,
        "passed": bool(score.get("passed", False)),
    }


def _no_regression(
    state: Mapping[str, Any], score: Mapping[str, Any], vision_judge: Mapping[str, Any] | None
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    best = state.get("best") or {}
    tolerance = float(state.get("max_identity_regression", 0.015))
    previous = best.get("source_similarity")
    current = score.get("source_similarity")
    if previous is not None and current is not None and float(current) < float(previous) - tolerance:
        reasons.append(
            f"整体身份分从 {float(previous):.3f} 降到 {float(current):.3f}，超过退化容差"
        )

    previous_attention = best.get("identity_attention_score")
    current_attention = _attention(score).get("identity_attention_score")
    if (
        previous_attention is not None
        and current_attention is not None
        and float(current_attention) < float(previous_attention) - 0.05
    ):
        reasons.append("身份锚点整体稳定度明显下降")

    if vision_judge and vision_judge.get("hard_gate_enabled"):
        confidence = float(vision_judge.get("confidence", 0.0))
        regressions = vision_judge.get("regressions") or []
        if confidence >= 0.70 and regressions:
            reasons.append("已校准视觉评分发现其他身份特征退化")
    return not reasons, reasons


def _focus_result(
    focus: Mapping[str, Any], score: Mapping[str, Any], min_improvement: float
) -> tuple[bool, bool, float | None, str]:
    key = str(focus.get("key"))
    gates = _gates(score)
    if key == "global_identity":
        passed = gates["embedding_identity"] and gates["identity_source"]
        return passed, passed, None, "整体身份门已通过" if passed else "整体身份门未通过"
    if key == "identity_source_conflict":
        passed = gates["identity_source"]
        return passed, passed, 0.0 if passed else 2.0, "身份来源冲突已消除" if passed else "仍疑似串入图2身份"

    feature = _find_feature(score, key)
    if not feature or float(feature.get("observability", 0.0)) < 0.35:
        return False, False, None, "当前重点不可可靠观测"
    severity = float(feature.get("severity", 0.0))
    baseline = focus.get("baseline_severity")
    improvement = baseline is not None and float(baseline) - severity >= min_improvement
    passed = severity < 1.0
    if passed:
        reason = f"当前重点 {key} 已进入允许范围"
    elif improvement:
        reason = f"当前重点 {key} 有改善但尚未达标"
    else:
        reason = f"当前重点 {key} 未取得有效改善"
    return passed, improvement, severity, reason


def _record(
    state: dict[str, Any], action: str, score: Mapping[str, Any], reasons: list[str]
) -> None:
    state["history"].append({
        "round": state["round"],
        "candidate": score.get("candidate"),
        "action": action,
        "focus": (state.get("current_focus") or {}).get("key"),
        "source_similarity": score.get("source_similarity"),
        "reasons": reasons,
    })


def advance_iteration(
    state: Mapping[str, Any],
    score: Mapping[str, Any],
    vision_judge: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """消费一轮评分，返回更新后的 JSON 状态。"""
    result = deepcopy(dict(state))
    if result.get("status") != "active":
        raise ValueError("迭代状态已经结束，不能继续推进")
    result["round"] = int(result.get("round", 0)) + 1
    gates = _gates(score)

    if result.get("best") is None and score.get("source_similarity") is not None:
        result["best"] = _snapshot(score)

    if not gates["observability"]:
        reasons = ["当前人脸不可可靠评分；先改善脸部尺寸、清晰度或姿态"]
        action = "retry_observability"
        if result["round"] >= result["max_rounds"]:
            result["status"] = "stopped"
            result["stop_reason"] = "达到最大轮数且仍不可可靠评分"
            action = "stop"
        _record(result, action, score, reasons)
        result["next_action"] = action
        result["decision_reasons"] = reasons
        return result

    current = result.get("current_focus")
    if current is None:
        focus = _eligible_focus(score, set(result.get("completed_focus") or []))
        if focus is None and not gates["embedding_identity"]:
            focus = _global_focus(score)
        if focus is None and score.get("passed"):
            result["status"] = "completed"
            result["next_action"] = "complete"
            result["decision_reasons"] = ["整体身份和可观测硬锚点均已通过"]
            _record(result, "complete", score, result["decision_reasons"])
            return result
        if focus is None:
            focus = _global_focus(score)
        result["current_focus"] = focus
        result["focus_attempts"] = 0
        result["next_action"] = "retry_focus"
        result["decision_reasons"] = [f"锁定当前唯一重点：{focus['key']}"]
        _record(result, "retry_focus", score, result["decision_reasons"])
        return result

    # 整体身份和身份来源永远高于局部。它们一旦失守，立即暂停当前局部修正。
    if not gates["identity_source"] and current.get("key") != "identity_source_conflict":
        result["current_focus"] = _identity_source_focus(score)
        result["focus_attempts"] = 0
        result["next_action"] = "retry_focus"
        result["decision_reasons"] = ["检测到身份来源冲突，暂停局部修正并先隔离图2身份"]
        _record(result, "retry_focus", score, result["decision_reasons"])
        return result
    if not gates["embedding_identity"] and current.get("key") != "global_identity":
        result["current_focus"] = _global_focus(score)
        result["focus_attempts"] = 0
        result["next_action"] = "retry_focus"
        result["decision_reasons"] = ["整体身份门失守，暂停局部修正并先恢复图1身份"]
        _record(result, "retry_focus", score, result["decision_reasons"])
        return result

    focus_passed, improved, severity, focus_reason = _focus_result(
        current, score, float(result.get("min_focus_improvement", 0.10))
    )
    no_regression, regression_reasons = _no_regression(result, score, vision_judge)
    global_ok = gates["embedding_identity"] and gates["identity_source"]
    reasons = [focus_reason] + regression_reasons
    ai_identity_failed = False
    if vision_judge and vision_judge.get("hard_gate_enabled"):
        confidence = float(vision_judge.get("confidence", 0.0))
        if confidence >= 0.70:
            identity_threshold = vision_judge.get("identity_threshold")
            if (
                identity_threshold is not None
                and float(vision_judge.get("identity_score", 0.0)) < float(identity_threshold)
            ):
                global_ok = False
                ai_identity_failed = True
                reasons.append("已校准AI身份分未达到真人标注分界线")
            if current.get("key") not in ("global_identity", "identity_source_conflict"):
                if not bool(vision_judge.get("focus_pass", False)):
                    focus_passed = False
                    reasons.append("已校准AI评分认为当前局部仍未达到人眼可辨识要求")

    if ai_identity_failed and current.get("key") != "global_identity":
        result["current_focus"] = _global_focus(score)
        result["focus_attempts"] = 0
        result["next_action"] = "retry_focus"
        reasons.append("暂停局部修正，先恢复整体身份")
        result["decision_reasons"] = reasons
        _record(result, "retry_focus", score, reasons)
        return result

    if focus_passed and no_regression and global_ok:
        key = str(current.get("key"))
        completed = list(result.get("completed_focus") or [])
        if key not in completed:
            completed.append(key)
        result["completed_focus"] = completed
        result["best"] = _snapshot(score)
        next_focus = _eligible_focus(score, set(completed))
        if next_focus is None:
            result["current_focus"] = None
            result["focus_attempts"] = 0
            result["status"] = "completed"
            result["next_action"] = "complete"
            reasons.append("没有剩余的可观测硬身份重点")
            _record(result, "complete", score, reasons)
        else:
            result["current_focus"] = next_focus
            result["focus_attempts"] = 0
            result["next_action"] = "advance_focus"
            reasons.append(f"进入下一个唯一重点：{next_focus['key']}")
            _record(result, "advance_focus", score, reasons)
        result["decision_reasons"] = reasons
        return result

    result["focus_attempts"] = int(result.get("focus_attempts", 0)) + 1
    if improved and no_regression:
        updated = dict(current)
        updated["baseline_severity"] = severity
        result["current_focus"] = updated
        result["best"] = _snapshot(score)
        reasons.append("保留本轮为当前重点的新基线，但不移动到下一部位")
    elif not no_regression:
        reasons.append("丢弃本轮，继续使用历史最佳候选")

    exhausted = (
        result["focus_attempts"] >= result["max_focus_attempts"]
        or result["round"] >= result["max_rounds"]
    )
    if exhausted:
        result["status"] = "stopped"
        result["stop_reason"] = "当前重点连续未达标或达到最大总轮数"
        result["next_action"] = "stop"
        _record(result, "stop", score, reasons)
    else:
        result["next_action"] = "retry_focus"
        _record(result, "retry_focus", score, reasons)
    result["decision_reasons"] = reasons
    return result


def compose_iteration_prompt(
    base_prompt: str,
    identity_anchor: str,
    state: Mapping[str, Any],
) -> str:
    """从干净模板重建本轮提示词，不携带历轮局部修正历史。"""
    prompt, _audit = compile_prompt_with_budget(base_prompt, identity_anchor, state)
    return prompt


def extract_identity_anchor(text: str, fallback: str) -> str:
    """提取经注意力分级的首轮身份基线；拒绝整份十项档案和失控长文本。"""
    for marker in ("基线锁脸段成品：", "锁脸段成品："):
        match = re.search(
            re.escape(marker) + r"\s*(.+?)(?=\n\s*\n|\n---|\n##|$)",
            text,
            re.S,
        )
        if match:
            baseline = " ".join(match.group(1).split())
            if baseline and len(baseline) <= 500:
                return "身份基线：" + baseline
    match = re.search(
        r"身份锚点成品：\s*(.+?)(?=\n\s*\n|\n---|\n##|\n(?:基线)?锁脸段成品：|$)",
        text,
        re.S,
    )
    if match:
        legacy_anchor = " ".join(match.group(1).split())
        if legacy_anchor and len(legacy_anchor) <= 300:
            return legacy_anchor
    stripped = text.strip()
    if stripped and len(stripped) <= 300 and "##" not in stripped and "锁脸段成品" not in stripped:
        return stripped
    return fallback
