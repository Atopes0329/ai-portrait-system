"""YuNet 检测结果去重；只合并同一张脸的重复框，不放松多人拦截。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    ax1, ay1, aw, ah = [float(value) for value in first[:4]]
    bx1, by1, bw, bh = [float(value) for value in second[:4]]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(aw * ah, 1e-9)
    second_area = max(bw * bh, 1e-9)
    union = first_area + second_area - intersection
    return intersection / max(union, 1e-9), intersection / min(first_area, second_area)


def _landmark_center(face: np.ndarray) -> np.ndarray:
    if len(face) >= 14:
        return face[4:14].reshape(5, 2).astype(np.float32).mean(axis=0)
    x, y, width, height = [float(value) for value in face[:4]]
    return np.array([x + width / 2.0, y + height / 2.0], dtype=np.float32)


def _is_duplicate(first: np.ndarray, second: np.ndarray) -> tuple[bool, dict[str, float]]:
    iou, containment = _overlap(first, second)
    landmark_distance = float(np.linalg.norm(_landmark_center(first) - _landmark_center(second)))
    first_diagonal = float(np.linalg.norm(first[2:4].astype(np.float32)))
    second_diagonal = float(np.linalg.norm(second[2:4].astype(np.float32)))
    normalized_landmark_distance = landmark_distance / max(
        min(first_diagonal, second_diagonal), 1e-9
    )
    duplicate = bool(
        normalized_landmark_distance <= 0.45
        and (containment >= 0.60 or iou >= 0.45)
    )
    return duplicate, {
        "iou": round(iou, 6),
        "smaller_box_containment": round(containment, 6),
        "normalized_landmark_distance": round(normalized_landmark_distance, 6),
    }


def merge_duplicate_faces(
    faces: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """按置信度保留代表框，最终仍按面积排序以兼容主脸选择。"""
    ordered = sorted(
        (np.asarray(face).copy() for face in faces),
        key=lambda face: float(face[14]) if len(face) > 14 else 0.0,
        reverse=True,
    )
    kept: list[np.ndarray] = []
    merged_pairs: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate_of = None
        duplicate_metrics = None
        for existing in kept:
            duplicate, metrics = _is_duplicate(existing, candidate)
            if duplicate:
                duplicate_of = existing
                duplicate_metrics = metrics
                break
        if duplicate_of is None:
            kept.append(candidate)
        else:
            merged_pairs.append({
                "kept_confidence": round(
                    float(duplicate_of[14]) if len(duplicate_of) > 14 else 0.0, 6
                ),
                "discarded_confidence": round(
                    float(candidate[14]) if len(candidate) > 14 else 0.0, 6
                ),
                **(duplicate_metrics or {}),
            })
    kept.sort(key=lambda face: float(face[2] * face[3]), reverse=True)
    return kept, {
        "schema_version": "1.0",
        "kind": "face_detection_deduplication",
        "raw_face_count": len(ordered),
        "unique_face_count": len(kept),
        "merged_duplicate_count": len(merged_pairs),
        "merged_pairs": merged_pairs,
    }
