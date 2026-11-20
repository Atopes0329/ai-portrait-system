"""候选图文件与像素完整性前置门，不评价身份或审美。"""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np


def _contiguous_edge_band(mask: np.ndarray, axis: int, reverse: bool) -> int:
    ratios = mask.mean(axis=axis)
    if reverse:
        ratios = ratios[::-1]
    count = 0
    for ratio in ratios:
        if float(ratio) < 0.995:
            break
        count += 1
    return count


def evaluate_image_integrity(
    image: np.ndarray,
    *,
    min_dimension: int = 256,
    black_value: int = 3,
    min_black_band_ratio: float = 0.02,
) -> dict[str, Any]:
    """拦截不可用图片；纯白棚拍背景不因边缘发白而被误拒。"""
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        return {
            "schema_version": "1.0",
            "kind": "candidate_integrity",
            "passed": False,
            "reasons": ["候选图像素数据为空或维度无效"],
        }
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    work = image if image.ndim == 2 else image[:, :, : min(channels, 3)]
    grayscale = work.astype(np.float32).mean(axis=2) if work.ndim == 3 else work.astype(np.float32)
    black_mask = (
        np.all(work <= black_value, axis=2) if work.ndim == 3 else work <= black_value
    )
    bands = {
        "top": _contiguous_edge_band(black_mask, axis=1, reverse=False),
        "bottom": _contiguous_edge_band(black_mask, axis=1, reverse=True),
        "left": _contiguous_edge_band(black_mask, axis=0, reverse=False),
        "right": _contiguous_edge_band(black_mask, axis=0, reverse=True),
    }
    min_vertical_band = max(8, int(ceil(height * min_black_band_ratio)))
    min_horizontal_band = max(8, int(ceil(width * min_black_band_ratio)))
    black_band_failure = bool(
        bands["top"] >= min_vertical_band
        or bands["bottom"] >= min_vertical_band
        or bands["left"] >= min_horizontal_band
        or bands["right"] >= min_horizontal_band
    )
    reasons: list[str] = []
    if min(height, width) < min_dimension:
        reasons.append(
            f"候选图尺寸过小：{width}×{height}，短边至少需要 {min_dimension} 像素"
        )
    grayscale_std = float(grayscale.std())
    if grayscale_std < 2.0:
        reasons.append("候选画面近乎空白或单色，不能进入身份评分")
    if black_band_failure:
        reasons.append("候选图边缘存在大面积连续纯黑边，疑似生成或下载不完整")
    return {
        "schema_version": "1.0",
        "kind": "candidate_integrity",
        "passed": not reasons,
        "reasons": reasons,
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "grayscale_std": round(grayscale_std, 6),
        "black_edge_bands": bands,
        "black_band_thresholds": {
            "vertical": min_vertical_band,
            "horizontal": min_horizontal_band,
        },
    }
