"""模型无关的身份扫描路径控制器。

本模块不控制生成模型内部 attention。它接收任意 landmark 提供器已经提取的
可解释指标，把有限的身份修正预算分给当前偏差最大、又可可靠观测的特征。
"""

from __future__ import annotations

from math import hypot
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.1"

# 已经进入硬门槛的基础锚点。它们来自 YuNet 五点 landmark。
BASE_FEATURE_SPECS = (
    {
        "key": "eye_distance",
        "group": "eye_nose_configuration",
        "tier": "anchor",
        "base_weight": 1.20,
        "threshold": 0.08,
        "evidence_mode": "gate",
        "provider": "yunet_5point",
        "instruction": "保持本人双眼间距相对脸宽的比例，不得拉宽或收窄。",
    },
    {
        "key": "face_ratio",
        "group": "face_outline",
        "tier": "anchor",
        "base_weight": 1.00,
        "threshold": 0.08,
        "evidence_mode": "gate",
        "provider": "yunet_5point",
        "instruction": "保持本人脸部宽长比例与整体轮廓，不得把脸拉长、压宽或收尖。",
    },
    {
        "key": "nose_to_eye",
        "group": "eye_nose_configuration",
        "tier": "anchor",
        "base_weight": 0.95,
        "threshold": 0.10,
        "evidence_mode": "gate",
        "provider": "yunet_5point",
        "instruction": "保持本人鼻尖相对双眼中心的位置关系，不得上移、下移或改变中庭比例。",
    },
)

# YuNet 原本就提供两个嘴角，旧流程没有利用。它们容易受笑容影响，因此只作为
# 条件性建议证据；即使偏差很大也不能单独判定身份失败。
FIVE_POINT_ADVISORY_SPECS = (
    {
        "key": "mouth_width",
        "group": "mouth_structure",
        "tier": "conditional_advisory",
        "base_weight": 0.55,
        "threshold": 0.15,
        "evidence_mode": "advisory",
        "provider": "yunet_5point",
        "instruction": "在保留当前表情的前提下，保持本人嘴角间距与脸宽比例。",
    },
    {
        "key": "nose_to_mouth",
        "group": "lower_face_configuration",
        "tier": "conditional_advisory",
        "base_weight": 0.65,
        "threshold": 0.12,
        "evidence_mode": "advisory",
        "provider": "yunet_5point",
        "instruction": "保持本人鼻尖到嘴部中心的距离关系，不要随造型改变下半脸比例。",
    },
)

# 旧 Skill 的 Face Mesh 实验被拆成可解释比例。阈值是工程初值，尚未用真人样本
# 校准，因此只能影响修正优先级和排序，不能单独触发身份拒绝。
DENSE_FEATURE_SPECS = (
    {
        "key": "dense_face_ratio",
        "group": "face_outline",
        "tier": "dense_advisory",
        "base_weight": 1.05,
        "threshold": 0.07,
        "evidence_mode": "advisory",
        "provider": "facemesh_468",
        "instruction": "保持本人额顶到下巴与两颊宽度的整体比例。",
    },
    {
        "key": "dense_jaw_width_ratio",
        "group": "face_outline",
        "tier": "dense_advisory",
        "base_weight": 1.10,
        "threshold": 0.08,
        "evidence_mode": "advisory",
        "provider": "facemesh_468",
        "instruction": "保持本人下颌宽度与收束程度，不要把下巴削尖或横向撑宽。",
    },
    {
        "key": "dense_intercanthal_ratio",
        "group": "eye_configuration",
        "tier": "dense_advisory",
        "base_weight": 1.05,
        "threshold": 0.08,
        "evidence_mode": "advisory",
        "provider": "facemesh_468",
        "instruction": "保持本人双眼内眼角间距与脸宽的关系。",
    },
    {
        "key": "dense_eye_width_ratio",
        "group": "eye_configuration",
        "tier": "dense_advisory",
        "base_weight": 0.90,
        "threshold": 0.09,
        "evidence_mode": "advisory",
        "provider": "facemesh_468",
        "instruction": "保持本人眼裂横向长度与脸宽的比例，不要放大或缩短眼睛。",
    },
    {
        "key": "dense_nose_width_ratio",
        "group": "nose_structure",
        "tier": "dense_advisory",
        "base_weight": 1.00,
        "threshold": 0.10,
        "evidence_mode": "advisory",
        "provider": "facemesh_468",
        "instruction": "保持本人鼻翼宽度与脸宽的比例。",
    },
    {
        "key": "dense_nose_length_ratio",
        "group": "nose_structure",
        "tier": "dense_advisory",
        "base_weight": 0.90,
        "threshold": 0.10,
        "evidence_mode": "advisory",
        "provider": "facemesh_468",
        "instruction": "保持本人鼻根到鼻尖的长度与中庭比例。",
    },
    {
        "key": "dense_mouth_width_ratio",
        "group": "mouth_structure",
        "tier": "conditional_advisory",
        "base_weight": 0.65,
        "threshold": 0.12,
        "evidence_mode": "advisory",
        "provider": "facemesh_468",
        "reliability_key": "dense_mouth_reliability",
        "instruction": "在不改变表情的前提下，保持本人嘴部横向宽度与脸宽比例。",
    },
)

FEATURE_SPECS = BASE_FEATURE_SPECS + FIVE_POINT_ADVISORY_SPECS + DENSE_FEATURE_SPECS


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def derive_face_metrics(face: Sequence[float]) -> dict[str, float] | None:
    """从 YuNet 人脸框和五点 landmark 提取粗粒度归一化指标。"""
    if len(face) < 14:
        return None
    values = [float(value) for value in face]
    _x, _y, width, height = values[:4]
    if width <= 0 or height <= 0:
        return None
    right_eye_x, right_eye_y = values[4:6]
    left_eye_x, left_eye_y = values[6:8]
    nose_x, nose_y = values[8:10]
    right_mouth_x, right_mouth_y = values[10:12]
    left_mouth_x, left_mouth_y = values[12:14]
    eye_distance = hypot(right_eye_x - left_eye_x, right_eye_y - left_eye_y) / width
    eye_mid_x = (right_eye_x + left_eye_x) / 2.0
    eye_mid_y = (right_eye_y + left_eye_y) / 2.0
    mouth_mid_x = (right_mouth_x + left_mouth_x) / 2.0
    mouth_mid_y = (right_mouth_y + left_mouth_y) / 2.0
    return {
        "eye_distance": eye_distance,
        "face_ratio": height / width,
        "nose_to_eye": hypot(nose_x - eye_mid_x, nose_y - eye_mid_y) / height,
        "mouth_width": hypot(right_mouth_x - left_mouth_x, right_mouth_y - left_mouth_y) / width,
        "nose_to_mouth": hypot(nose_x - mouth_mid_x, nose_y - mouth_mid_y) / height,
        # 以下只决定二维结构是否适合比较，不是身份锚点。
        "lateral_asymmetry": abs(nose_x - eye_mid_x) / width,
        "eye_line_tilt": abs(right_eye_y - left_eye_y) / max(eye_distance * width, 1e-9),
        "detector_confidence": values[14] if len(values) > 14 else 1.0,
    }


def build_identity_profile(metrics: Mapping[str, float]) -> dict[str, Any]:
    """建立参考身份档案；稠密指标可选，基础三个锚点必需。"""
    missing = [spec["key"] for spec in BASE_FEATURE_SPECS if spec["key"] not in metrics]
    if missing:
        raise ValueError("参考图缺少身份锚点：" + "、".join(missing))
    active_specs = [spec for spec in FEATURE_SPECS if spec["key"] in metrics]
    profile_metrics = {
        spec["key"]: _round(float(metrics[spec["key"]])) for spec in active_specs
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "identity_profile",
        "metrics": profile_metrics,
        "feature_groups": [
            {
                key: spec[key]
                for key in (
                    "key", "group", "tier", "base_weight", "threshold",
                    "evidence_mode", "provider",
                )
            }
            for spec in active_specs
        ],
        "capabilities": {
            "yunet_5point": True,
            "facemesh_468": any(spec["provider"] == "facemesh_468" for spec in active_specs),
        },
    }


def _observability(metrics: Mapping[str, float], face_area_ratio: float) -> dict[str, Any]:
    """决定二维几何是否可作为硬门槛，避免侧脸/小脸被误拒。"""
    reasons: list[str] = []
    area_score = _clip01(face_area_ratio / 0.02)
    detector_score = _clip01(float(metrics.get("detector_confidence", 1.0)))
    lateral = abs(float(metrics.get("lateral_asymmetry", 0.0)))
    tilt = abs(float(metrics.get("eye_line_tilt", 0.0)))
    pose_score = 1.0
    if lateral > 0.10:
        pose_score *= 0.55
        reasons.append("鼻尖明显偏离双眼中心，当前可能为侧脸或强透视")
    if tilt > 0.18:
        pose_score *= 0.75
        reasons.append("双眼连线倾斜较大，二维比例比较可信度下降")
    if face_area_ratio < 0.008:
        reasons.append("人脸过小")
    score = area_score * detector_score * pose_score
    if score < 0.45:
        status = "unreliable"
    elif score < 0.72:
        status = "limited"
    else:
        status = "reliable"
    return {
        "status": status,
        "score": _round(score),
        "face_area_ratio": _round(face_area_ratio),
        "lateral_asymmetry": _round(lateral),
        "eye_line_tilt": _round(tilt),
        "reasons": reasons,
    }


def _feature_observability(
    spec: Mapping[str, Any], metrics: Mapping[str, float], global_score: float
) -> float:
    score = global_score
    if spec["provider"] == "facemesh_468":
        score *= _clip01(float(metrics.get("dense_presence", 0.0)))
    reliability_key = spec.get("reliability_key")
    if reliability_key:
        score *= _clip01(float(metrics.get(str(reliability_key), 0.0)))
    return score


def _select_focus(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只选一个当前重点，避免修正指令重新形成注意力过载。"""
    for item in features:
        if float(item["severity"]) < 0.5 or float(item["observability"]) < 0.35:
            continue
        return [item]
    return []


def evaluate_attention(
    profile: Mapping[str, Any],
    candidate_metrics: Mapping[str, float] | None,
    face_area_ratio: float,
    model_disagreement: float = 0.0,
    identity_conflict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """比较候选与身份档案，输出可解释的注意力计划和验收结论。"""
    if not candidate_metrics:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "attention_report",
            "decision": "unreliable",
            "identity_attention_score": None,
            "observability": {"status": "unreliable", "score": 0.0, "reasons": ["无法提取候选身份锚点"]},
            "features": [],
            "next_focus": [],
            "instruction": "候选图无法可靠提取面部身份锚点；先生成脸更清晰、更正面的候选图。",
            "reject_reasons": [],
        }

    reference = profile.get("metrics") or {}
    missing_reference = [spec["key"] for spec in BASE_FEATURE_SPECS if spec["key"] not in reference]
    if missing_reference:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "attention_report",
            "decision": "unreliable",
            "identity_attention_score": None,
            "observability": {"status": "unreliable", "score": 0.0, "reasons": ["参考图身份档案不完整"]},
            "features": [],
            "next_focus": [],
            "instruction": "参考图缺少可用身份锚点，先更换或补充清晰正脸参考图。",
            "reject_reasons": [],
        }

    observability = _observability(candidate_metrics, face_area_ratio)
    disagreement_boost = 1.0 + min(max(float(model_disagreement), 0.0), 0.35)
    features: list[dict[str, Any]] = []
    weighted_penalty = 0.0
    total_weight = 0.0
    reject_reasons = []

    for spec in FEATURE_SPECS:
        key = spec["key"]
        ref_value = reference.get(key)
        cand_value = candidate_metrics.get(key)
        if ref_value is None or cand_value is None or not float(ref_value):
            continue
        deviation = (float(cand_value) - float(ref_value)) / float(ref_value)
        severity = abs(deviation) / float(spec["threshold"])
        feature_observability = _feature_observability(
            spec, candidate_metrics, float(observability["score"])
        )
        gate_eligible = (
            spec["evidence_mode"] == "gate"
            and observability["status"] == "reliable"
        )
        status = "matched"
        if feature_observability < 0.35:
            status = "unobservable"
        elif severity >= 1.0 and gate_eligible:
            status = "anchor_mismatch"
            reject_reasons.append(f"身份锚点 {key} 偏差 {abs(deviation):.1%} 超过允许范围")
        elif severity >= 1.0 and spec["evidence_mode"] == "advisory":
            status = "advisory_mismatch"
        elif severity >= 1.0:
            status = "needs_recheck"
        elif severity >= 0.5:
            status = "drifting"

        priority = (
            float(spec["base_weight"])
            * min(severity, 2.0)
            * feature_observability
            * disagreement_boost
        )
        if feature_observability >= 0.35:
            weighted_penalty += (
                float(spec["base_weight"])
                * min(severity, 1.5)
                * feature_observability
            )
            total_weight += float(spec["base_weight"]) * feature_observability
        features.append({
            "key": key,
            "group": spec["group"],
            "tier": spec["tier"],
            "provider": spec["provider"],
            "evidence_mode": spec["evidence_mode"],
            "reference": _round(float(ref_value)),
            "candidate": _round(float(cand_value)),
            "deviation": _round(deviation),
            "severity": _round(severity),
            "observability": _round(feature_observability),
            "priority": _round(priority),
            "status": status,
            "instruction": spec["instruction"],
        })

    # 图2身份竞争不是某个五官比例错误，而是身份来源错误。把它放进同一注意力
    # 队列，确保下一轮先隔离错误身份，而不是盲目修改眼鼻嘴。
    if identity_conflict and bool(identity_conflict.get("active")):
        source_lead = float(identity_conflict.get("source_lead", 0.0))
        conflict_reason = "候选身份疑似受到图2人物影响，图1没有形成足够身份领先"
        reject_reasons.append(conflict_reason)
        conflict_feature = {
            "key": "identity_source_conflict",
            "group": "identity_source",
            "tier": "identity_source_gate",
            "provider": "identity_comparator",
            "evidence_mode": "gate",
            "reference": _round(float(identity_conflict.get("source_similarity", 0.0))),
            "candidate": _round(float(identity_conflict.get("style_similarity", 0.0))),
            "deviation": _round(-source_lead),
            "severity": 2.0,
            "observability": 1.0,
            "priority": 10.0,
            "status": "identity_source_conflict",
            "instruction": "重新强调图1是唯一身份来源，禁止从图2遮挡区域猜测或继承任何脸型和五官。",
        }
        features.append(conflict_feature)
        weighted_penalty += 1.5
        total_weight += 1.0

    features.sort(key=lambda item: float(item["priority"]), reverse=True)
    next_focus = _select_focus(features)
    advisory_repair_recommended = any(
        item["tier"] == "dense_advisory"
        and float(item["severity"]) >= 1.25
        and float(item["observability"]) >= 0.70
        for item in features
    )
    attention_score = 1.0 - min(1.0, weighted_penalty / max(total_weight * 1.5, 1e-9))
    if observability["status"] == "unreliable":
        decision = "unreliable"
    elif reject_reasons:
        decision = "reject"
    else:
        decision = "pass"

    if decision == "unreliable":
        instruction = "当前姿态、清晰度或脸部尺寸不足以可靠比较身份锚点；不要据此改脸，先生成可评估候选。"
    elif next_focus:
        focus_text = "；".join(item["instruction"] for item in next_focus)
        instruction = "下一轮只优先修正以下身份锚点：" + focus_text
    else:
        instruction = "身份锚点稳定；下一轮保持当前锁脸段，不要额外堆叠新的五官描述。"

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "attention_report",
        "decision": decision,
        "identity_attention_score": _round(attention_score),
        "observability": observability,
        "model_disagreement": _round(float(model_disagreement)),
        "identity_conflict": dict(identity_conflict or {}),
        "advisory_repair_recommended": advisory_repair_recommended,
        "feature_sources": {
            "yunet_5point": sum(item["provider"] == "yunet_5point" for item in features),
            "facemesh_468": sum(item["provider"] == "facemesh_468" for item in features),
            "identity_comparator": sum(item["provider"] == "identity_comparator" for item in features),
        },
        "features": features,
        "next_focus": [
            {
                key: item[key]
                for key in (
                    "key", "group", "provider", "evidence_mode", "deviation",
                    "severity", "observability", "priority", "instruction",
                )
            }
            for item in next_focus
        ],
        "instruction": instruction,
        "reject_reasons": reject_reasons,
    }
