"""从 468 点 landmark 提取可解释的身份比例；不依赖具体推理框架。"""

from __future__ import annotations

from math import hypot
from typing import Sequence


LANDMARK_COUNT = 468


def _distance(points: Sequence[Sequence[float]], left: int, right: int) -> float:
    return hypot(
        float(points[left][0]) - float(points[right][0]),
        float(points[left][1]) - float(points[right][1]),
    )


def derive_dense_metrics(
    points: Sequence[Sequence[float]], presence: float = 1.0
) -> dict[str, float]:
    """提取不依赖像素尺寸的比例；动态嘴部状态只控制可靠性。"""
    if len(points) < LANDMARK_COUNT:
        raise ValueError(f"Face Mesh 至少需要 {LANDMARK_COUNT} 个点，实际只有 {len(points)} 个。")

    face_width = _distance(points, 234, 454)
    face_height = _distance(points, 10, 152)
    if face_width <= 1e-9 or face_height <= 1e-9:
        raise ValueError("Face Mesh 脸部基准宽高为零，无法计算身份比例。")

    right_eye_width = _distance(points, 33, 133)
    left_eye_width = _distance(points, 362, 263)
    mouth_openness = _distance(points, 13, 14) / face_height
    # 张嘴越明显，嘴宽越容易受表情拉伸；0.015 以下基本不降权，0.08 后不使用。
    mouth_reliability = 1.0 - max(0.0, mouth_openness - 0.015) / 0.065
    mouth_reliability = min(1.0, max(0.0, mouth_reliability))

    return {
        "dense_face_ratio": face_height / face_width,
        "dense_jaw_width_ratio": _distance(points, 172, 397) / face_width,
        "dense_intercanthal_ratio": _distance(points, 133, 362) / face_width,
        "dense_eye_width_ratio": ((right_eye_width + left_eye_width) / 2.0) / face_width,
        "dense_nose_width_ratio": _distance(points, 98, 327) / face_width,
        "dense_nose_length_ratio": _distance(points, 168, 2) / face_height,
        "dense_mouth_width_ratio": _distance(points, 61, 291) / face_width,
        "dense_mouth_openness": mouth_openness,
        "dense_mouth_reliability": mouth_reliability,
        "dense_presence": min(1.0, max(0.0, float(presence))),
    }
