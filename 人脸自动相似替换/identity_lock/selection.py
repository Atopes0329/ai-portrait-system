"""模型无关的最终候选选择策略。

最终器不能另起一套排序覆盖迭代状态机已经做出的回退决定。这里统一候选
选择的权威顺序，并只在状态缺失时提供保守、可解释的机器兜底。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def _normalized_candidate(value: Any) -> str | None:
    if not value:
        return None
    return str(Path(str(value)).expanduser().resolve(strict=False))


def _number(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _machine_rank(score: Mapping[str, Any]) -> tuple[Any, ...]:
    """无状态时的保守兜底；不把不同模型的原始分数做虚假平均。"""
    gates = score.get("gates") or {}
    attention = score.get("attention_priority") or {}
    return (
        bool(score.get("passed", False)),
        bool(gates.get("observability", True)),
        bool(gates.get("identity_source", True)),
        bool(gates.get("embedding_identity", score.get("passed", False))),
        bool(gates.get("anchor", attention.get("decision") != "reject")),
        -_number(score.get("identity_model_spread"), 0.0),
        _number(attention.get("identity_attention_score"), -2.0),
        _number(score.get("source_similarity"), -2.0),
        _number(score.get("identity_margin"), -2.0),
    )


def _find_candidate(
    scores: Sequence[Mapping[str, Any]], candidate: Any
) -> Mapping[str, Any] | None:
    wanted = _normalized_candidate(candidate)
    if wanted is None:
        return None
    for score in scores:
        if _normalized_candidate(score.get("candidate")) == wanted:
            return score
    return None


def select_final_candidate(
    scores: Sequence[Mapping[str, Any]],
    *,
    iteration_state: Mapping[str, Any] | None = None,
    preferred_candidate: Any = None,
) -> dict[str, Any]:
    """按人工复核、状态机最佳、机器兜底的顺序选出最终候选。"""
    scorable = [
        score
        for score in scores
        if score.get("source_similarity") is not None
        and bool(
            (score.get("gates") or {}).get(
                "quality", (score.get("image_integrity") or {}).get("passed", True)
            )
        )
    ]
    if not scorable:
        raise ValueError("全部候选都无法检测人脸，不能选出结果。")

    if preferred_candidate is not None:
        preferred = _find_candidate(scorable, preferred_candidate)
        if preferred is None:
            recorded = _find_candidate(scores, preferred_candidate)
            if recorded is not None:
                raise ValueError(
                    f"指定的视觉优选图未通过画面完整性或身份可评分检查："
                    f"{preferred_candidate}"
                )
            raise ValueError(f"指定的视觉优选图没有评分记录：{preferred_candidate}")
        return {
            "selected": preferred,
            "selection_mode": "视觉复核优先",
            "selection_source": "manual_review",
            "fallback_reason": None,
        }

    fallback_reason: str
    if iteration_state is not None:
        state_candidate = (iteration_state.get("best") or {}).get("candidate")
        if state_candidate:
            state_best = _find_candidate(scorable, state_candidate)
            if state_best is not None:
                return {
                    "selected": state_best,
                    "selection_mode": "迭代状态机历史最佳",
                    "selection_source": "iteration_state",
                    "fallback_reason": None,
                }
            fallback_reason = "迭代状态机最佳候选没有对应评分记录"
        else:
            fallback_reason = "迭代状态机尚未记录最佳候选"
    else:
        fallback_reason = "没有可用的迭代状态文件"

    passed = [score for score in scorable if score.get("passed")]
    pool = passed or scorable
    return {
        "selected": max(pool, key=_machine_rank),
        "selection_mode": "保守机器兜底",
        "selection_source": "machine_fallback",
        "fallback_reason": fallback_reason,
    }
