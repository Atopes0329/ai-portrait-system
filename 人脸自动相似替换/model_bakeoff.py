#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实样片身份指标校准工具。

它只输出各模型和各身份锚点的原始结果/排名，不把实验阈值变成生产硬门槛。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from face_guard import FaceEngine, imread
from identity_ensemble import IdentityEnsemble
from identity_lock import derive_face_metrics
from landmark_providers import get_facemesh_provider


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
BASE_KEYS = ("eye_distance", "face_ratio", "nose_to_eye")
DENSE_KEYS = (
    "dense_face_ratio",
    "dense_jaw_width_ratio",
    "dense_intercanthal_ratio",
    "dense_eye_width_ratio",
    "dense_nose_width_ratio",
    "dense_nose_length_ratio",
    "dense_mouth_width_ratio",
)


def _relative_deviations(reference: dict, candidate: dict, keys: tuple[str, ...]) -> dict:
    deviations = {}
    for key in keys:
        ref_value = reference.get(key)
        cand_value = candidate.get(key)
        if ref_value is None or cand_value is None or not float(ref_value):
            continue
        deviations[key] = abs((float(cand_value) - float(ref_value)) / float(ref_value))
    return deviations


def _mean_or_none(values: dict[str, float]) -> float | None:
    return mean(values.values()) if values else None


def _rank(valid: list[dict], key: str, reverse: bool = False) -> list[str]:
    available = [item for item in valid if item.get(key) is not None]
    return [item["name"] for item in sorted(available, key=lambda item: item[key], reverse=reverse)]


def main() -> int:
    parser = argparse.ArgumentParser(description="身份模型与可解释锚点真实样片校准")
    parser.add_argument("--reference", required=True, help="本人基准图")
    parser.add_argument("--candidate", required=True, nargs="+", help="同人/异人候选样片")
    parser.add_argument("--json", help="保存完整结果")
    args = parser.parse_args()

    detector = FaceEngine()
    ensemble = IdentityEnsemble()
    facemesh, facemesh_status = get_facemesh_provider(MODELS_DIR)

    reference_path = Path(args.reference).expanduser().resolve()
    reference_image = imread(reference_path)
    reference_faces = detector.detect_all(reference_image)
    if not reference_faces:
        raise ValueError("本人基准图未检测到人脸")
    reference_face = reference_faces[0]
    reference_sface = detector.feature(reference_image, reference_face)
    reference_ensemble = ensemble.features(reference_image, reference_face)
    reference_metrics = derive_face_metrics(reference_face) or {}
    reference_dense_status = dict(facemesh_status)
    if facemesh is not None:
        try:
            dense_measurement = facemesh.measure(reference_image, reference_face)
            reference_metrics.update(dense_measurement["metrics"])
            reference_dense_status.update(
                state="active",
                presence=dense_measurement["presence"],
                landmark_count=dense_measurement["landmark_count"],
            )
        except Exception as exc:
            reference_dense_status.update(state="inference_failed", reason=str(exc))

    results = []
    for raw_path in args.candidate:
        path = Path(raw_path).expanduser().resolve()
        image = imread(path)
        faces = detector.detect_all(image)
        if not faces:
            results.append({"name": path.stem, "path": str(path), "error": "未检测到人脸"})
            continue
        face = faces[0]
        candidate_sface = detector.feature(image, face)
        candidate_ensemble = ensemble.features(image, face)
        metrics = derive_face_metrics(face) or {}
        dense_status = dict(facemesh_status)
        if facemesh is not None and any(key in reference_metrics for key in DENSE_KEYS):
            try:
                measurement = facemesh.measure(image, face)
                metrics.update(measurement["metrics"])
                dense_status.update(
                    state="active",
                    presence=measurement["presence"],
                    landmark_count=measurement["landmark_count"],
                )
            except Exception as exc:
                dense_status.update(state="inference_failed", reason=str(exc))

        base_deviations = _relative_deviations(reference_metrics, metrics, BASE_KEYS)
        dense_deviations = _relative_deviations(reference_metrics, metrics, DENSE_KEYS)
        results.append({
            "name": path.stem,
            "path": str(path),
            "sface_cosine": round(detector.cosine(reference_sface, candidate_sface), 6),
            "arcface_cosine": round(
                ensemble.cosine(reference_ensemble["arcface"], candidate_ensemble["arcface"]), 6
            ),
            "adaface_cosine": round(
                ensemble.cosine(reference_ensemble["adaface"], candidate_ensemble["adaface"]), 6
            ),
            "base_anchor_mean_deviation": (
                round(_mean_or_none(base_deviations), 6) if base_deviations else None
            ),
            "dense_anchor_mean_deviation": (
                round(_mean_or_none(dense_deviations), 6) if dense_deviations else None
            ),
            "base_anchor_deviations": {key: round(value, 6) for key, value in base_deviations.items()},
            "dense_anchor_deviations": {key: round(value, 6) for key, value in dense_deviations.items()},
            "dense_provider": dense_status,
        })

    valid = [item for item in results if "error" not in item]
    report = {
        "schema_version": "2.0",
        "purpose": "offline_identity_calibration",
        "warning": "稠密指标仍是建议证据；本报告不自动修改生产阈值。",
        "reference": str(reference_path),
        "reference_metrics": {
            key: round(float(value), 6)
            for key, value in reference_metrics.items()
            if key in BASE_KEYS + DENSE_KEYS
        },
        "facemesh_provider": reference_dense_status,
        "results": results,
        "rankings": {
            "sface": _rank(valid, "sface_cosine", reverse=True),
            "arcface": _rank(valid, "arcface_cosine", reverse=True),
            "adaface": _rank(valid, "adaface_cosine", reverse=True),
            "base_anchors": _rank(valid, "base_anchor_mean_deviation"),
            "dense_anchors": _rank(valid, "dense_anchor_mean_deviation"),
        },
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json:
        Path(args.json).expanduser().resolve().write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
