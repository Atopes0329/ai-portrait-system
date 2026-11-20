#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""写真重 roll 的本地隐私护栏与人脸评分器。

这个脚本不负责生图。它负责建立身份任务、整理系统摄影风格、
在需要时遮蔽图 2 身份，并对候选图执行本地身份闭环评分。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

import cv2
import numpy as np

from candidate_quality import evaluate_image_integrity
from face_detection_utils import merge_duplicate_faces
from identity_ensemble import IdentityEnsemble
from identity_lock import (
    advance_iteration,
    build_identity_profile,
    compile_prompt_with_budget,
    compose_iteration_prompt,
    derive_face_metrics,
    evaluate_attention,
    extract_identity_anchor,
    new_iteration_state,
    select_final_candidate,
    strip_identity_blocks,
)
from identity_lock.job_contract import (
    add_identity_reference_arguments,
    job_mode,
    normalize_auxiliary_references,
    style_source_kind,
)
from landmark_providers import get_facemesh_provider
from style_library import (
    default_style_library,
    merge_style_prompt,
    resolve_library_preset,
)
from vision_judges import OpenAICompatibleVisionJudge, load_vision_judge_config


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
REC_MODEL = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
DEFAULT_OUTPUT = ROOT / "任务输出"
DEFAULT_STYLE_LIBRARY = default_style_library(ROOT)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def imread(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def imwrite(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(str(path))
    return path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FaceEngine:
    def __init__(self, threshold: float = 0.65):
        if not DET_MODEL.is_file() or not REC_MODEL.is_file():
            raise FileNotFoundError("models 目录中的 YuNet 或 SFace 模型缺失。")
        self.detector = cv2.FaceDetectorYN.create(
            str(DET_MODEL), "", (320, 320), threshold, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(REC_MODEL), "")
        self.last_detection_audit = {
            "raw_face_count": 0,
            "unique_face_count": 0,
            "merged_duplicate_count": 0,
            "merged_pairs": [],
        }

    def detect_all(self, image: np.ndarray) -> list[np.ndarray]:
        """检测全部人脸；在大图上缩放检测并映射回原尺寸。"""
        height, width = image.shape[:2]
        scale = min(1.0, 1800.0 / max(height, width))
        if scale < 1.0:
            work = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = image
        self.detector.setInputSize((work.shape[1], work.shape[0]))
        _, faces = self.detector.detect(work)
        if faces is None:
            self.last_detection_audit = {
                "schema_version": "1.0",
                "kind": "face_detection_deduplication",
                "raw_face_count": 0,
                "unique_face_count": 0,
                "merged_duplicate_count": 0,
                "merged_pairs": [],
            }
            return []
        result = []
        for face in faces:
            item = face.copy()
            if scale < 1.0:
                item[:14] /= scale
            result.append(item)
        unique, audit = merge_duplicate_faces(result)
        self.last_detection_audit = audit
        return unique

    def feature(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(image, face)
        return self.recognizer.feature(aligned).astype(np.float32)

    def cosine(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        return float(
            self.recognizer.match(
                feature_a, feature_b, cv2.FaceRecognizerSF_FR_COSINE
            )
        )


def _expanded_ellipse(
    shape: tuple[int, ...], face: np.ndarray, side: float, top: float, bottom: float
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = shape[:2]
    x, y, fw, fh = [float(v) for v in face[:4]]
    left = max(0, int(round(x - side * fw)))
    right = min(width - 1, int(round(x + fw + side * fw)))
    upper = max(0, int(round(y - top * fh)))
    lower = min(height - 1, int(round(y + fh + bottom * fh)))
    center = ((left + right) // 2, (upper + lower) // 2)
    axes = (max(1, (right - left) // 2), max(1, (lower - upper) // 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1, lineType=cv2.LINE_AA)

    # 椭圆两侧可能漏出耳前/下颌边缘；再覆盖原检测框的扩大矩形。
    core_left = max(0, int(round(x - 0.08 * fw)))
    core_right = min(width - 1, int(round(x + 1.08 * fw)))
    core_top = max(0, int(round(y - 0.10 * fh)))
    core_bottom = min(height - 1, int(round(y + 1.10 * fh)))
    cv2.rectangle(mask, (core_left, core_top), (core_right, core_bottom), 255, -1)
    return mask, {
        "left": left,
        "top": upper,
        "right": right,
        "bottom": lower,
    }


def _neutral_cover_color(image: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    kernel_size = max(9, int(round(min(image.shape[:2]) * 0.02)) | 1)
    dilated = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8))
    ring = (dilated > 0) & (mask == 0)
    if np.count_nonzero(ring) < 30:
        luminance = float(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    else:
        pixels = image[ring].astype(np.float32)
        # 用周围区域亮度，但去掉肤色/色相，避免遮罩携带身份特征。
        luminance = float(np.median(cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)))
    value = int(np.clip(luminance, 28, 210))
    return value, value, value


def opaque_mask_faces(
    image: np.ndarray,
    faces: Iterable[np.ndarray],
    side: float = 0.30,
    top: float = 0.45,
    bottom: float = 0.28,
) -> tuple[np.ndarray, list[dict[str, int]], np.ndarray]:
    """用完全不透明的无特征灰色遮罩覆盖全部脸，不使用模糊或马赛克。"""
    output = image.copy()
    union = np.zeros(image.shape[:2], dtype=np.uint8)
    regions: list[dict[str, int]] = []
    for face in faces:
        mask, region = _expanded_ellipse(image.shape, face, side, top, bottom)
        union = cv2.bitwise_or(union, mask)
        regions.append(region)
    color = _neutral_cover_color(image, union)
    output[union > 0] = color

    # 硬遮罩边缘做 3px 抗锯齿，但内部仍保持 100% 不透明。
    edge = cv2.GaussianBlur(union, (5, 5), 0).astype(np.float32) / 255.0
    alpha = edge[:, :, None]
    flat = np.full_like(image, color)
    output = np.clip(flat * alpha + image * (1.0 - alpha), 0, 255).astype(np.uint8)
    output[union >= 250] = color
    return output, regions, union


def coarse_mosaic_faces(
    image: np.ndarray,
    faces: Iterable[np.ndarray],
    side: float = 0.18,
    top: float = 0.22,
    bottom: float = 0.18,
    blocks_across: int = 4,
) -> tuple[np.ndarray, list[dict[str, int]], np.ndarray]:
    """用极粗马赛克破坏五官身份，同时保留头部朝向和局部明暗关系。"""
    output = image.copy()
    union = np.zeros(image.shape[:2], dtype=np.uint8)
    regions: list[dict[str, int]] = []
    for face in faces:
        mask, region = _expanded_ellipse(image.shape, face, side, top, bottom)
        union = cv2.bitwise_or(union, mask)
        regions.append(region)
        left, top_edge, right, bottom_edge = (
            region["left"], region["top"], region["right"], region["bottom"]
        )
        crop = image[top_edge : bottom_edge + 1, left : right + 1]
        crop_mask = mask[top_edge : bottom_edge + 1, left : right + 1]
        if crop.size == 0:
            continue

        # 先强模糊再压到约 4×N 个色块，避免马赛克仍能还原眼鼻嘴细节。
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=max(8.0, crop.shape[1] * 0.08))
        small_width = max(3, int(blocks_across))
        small_height = max(4, round(crop.shape[0] / crop.shape[1] * small_width))
        tiny = cv2.resize(blurred, (small_width, small_height), interpolation=cv2.INTER_AREA)
        mosaic = cv2.resize(tiny, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        active = crop_mask > 0
        target = output[top_edge : bottom_edge + 1, left : right + 1]
        target[active] = mosaic[active]
    return output, regions, union


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return cleaned.strip("_") or "任务"


def _measure_optional_dense_metrics(
    image: np.ndarray, face: np.ndarray, base_metrics: dict[str, float]
) -> tuple[dict[str, float], dict]:
    """尝试稠密测量；任何失败都只降级，不影响基础五点身份链路。"""
    provider, status = get_facemesh_provider(MODELS_DIR)
    if provider is None:
        return {}, status
    lateral = abs(float(base_metrics.get("lateral_asymmetry", 0.0)))
    tilt = abs(float(base_metrics.get("eye_line_tilt", 0.0)))
    if lateral > 0.10 or tilt > 0.18:
        degraded = dict(status)
        degraded.update(
            state="pose_unreliable",
            reason="当前姿态或滚转角不适合二维稠密比例比较，已退回五点身份锚点。",
        )
        return {}, degraded
    try:
        measurement = provider.measure(image, face)
    except Exception as exc:
        degraded = dict(status)
        degraded.update(state="inference_failed", reason=f"Face Mesh 推理失败，已降级：{exc}")
        return {}, degraded
    presence = float(measurement["presence"])
    if presence < 0.75:
        degraded = dict(status)
        degraded.update(
            state="low_presence",
            reason=f"Face Mesh presence={presence:.3f} 低于 0.750，已退回五点身份锚点。",
            presence=round(presence, 6),
        )
        return {}, degraded
    ready = dict(status)
    ready.update(
        state="active",
        reason="Face Mesh 稠密建议层已启用；本阶段不单独触发身份拒绝。",
        presence=round(presence, 6),
        landmark_count=int(measurement["landmark_count"]),
    )
    return dict(measurement["metrics"]), ready


def _validate_prepare_args(args: argparse.Namespace) -> None:
    """GUI 之外的 CLI 也必须遵守同一组安全边界。"""
    if not 0.75 <= float(args.threshold) <= 0.97:
        raise ValueError("身份门槛必须在 0.75 到 0.97 之间。")
    if not 0.0 <= float(args.leakage_threshold) <= 1.0:
        raise ValueError("图2串脸警戒线必须在 0 到 1 之间。")
    if not 0.0 <= float(args.identity_margin) <= 1.0:
        raise ValueError("图1领先值必须在 0 到 1 之间。")
    if not 1 <= int(args.max_rolls) <= 20:
        raise ValueError("最多生成次数必须在 1 到 20 之间。")
    if not 1 <= int(args.mosaic_blocks) <= 4:
        raise ValueError("马赛克横向色块数必须在 1 到 4 之间。")
    for name in ("mask_side", "mask_top", "mask_bottom"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} 必须在 0 到 1 之间。")


def prepare_job(args: argparse.Namespace) -> dict:
    _validate_prepare_args(args)
    source_path = Path(args.source).expanduser().resolve()
    style_value = getattr(args, "style", None)
    style_preset_value = getattr(args, "style_preset", None)
    if style_value and style_preset_value:
        raise ValueError("图2风格与系统摄影风格只能选择一种。")
    style_path = Path(style_value).expanduser().resolve() if style_value else None
    style_preset = (
        resolve_library_preset(style_preset_value, DEFAULT_STYLE_LIBRARY)
        if style_preset_value
        else None
    )
    raw_auxiliary_paths = list(getattr(args, "reference", []) or [])
    if style_path and raw_auxiliary_paths:
        raise ValueError("图片2替换模式固定只使用图1与已遮脸图2，不接收辅助真人参考图。")
    auxiliary_paths = [
        Path(path)
        for path in normalize_auxiliary_references(
            source_path, raw_auxiliary_paths, require_exists=True
        )
    ]
    if not source_path.is_file() or source_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"图1不是有效图片：{source_path}")
    if style_path and (
        not style_path.is_file() or style_path.suffix.lower() not in IMAGE_SUFFIXES
    ):
        raise ValueError(f"图2不是有效图片：{style_path}")
    for path in auxiliary_paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"辅助真人参考图不是有效图片：{path}")

    engine = FaceEngine()
    identity_ensemble = IdentityEnsemble()
    source_image = imread(source_path)
    style_image = imread(style_path) if style_path else None
    source_faces = engine.detect_all(source_image)
    source_detection_audit = dict(engine.last_detection_audit)
    style_faces = engine.detect_all(style_image) if style_image is not None else []
    style_detection_audit = (
        dict(engine.last_detection_audit)
        if style_image is not None
        else {
            "raw_face_count": 0,
            "unique_face_count": 0,
            "merged_duplicate_count": 0,
            "merged_pairs": [],
        }
    )
    if not source_faces:
        raise ValueError("图1没有检测到清晰人脸，不能建立身份基准。")
    if len(source_faces) != 1:
        raise ValueError(
            f"图1检测到 {len(source_faces)} 张脸。为避免身份混淆，请裁成只含本人一张脸的照片。"
        )
    if style_path and not style_faces:
        raise ValueError("图2没有检测到人脸；为安全起见已停止，避免未遮脸原图进入生图。")

    source_face = source_faces[0]
    source_metrics = derive_face_metrics(source_face)
    if not source_metrics:
        raise ValueError("图1人脸关键点不完整，不能建立身份注意力档案。")
    source_dense_metrics, source_dense_status = _measure_optional_dense_metrics(
        source_image, source_face, source_metrics
    )
    source_metrics.update(source_dense_metrics)
    identity_profile = build_identity_profile(source_metrics)
    identity_profile["dense_landmark_provider"] = source_dense_status
    source_feature = engine.feature(source_image, source_face)
    source_ensemble = identity_ensemble.features(source_image, source_face)
    if style_image is not None:
        style_features = np.concatenate(
            [engine.feature(style_image, face) for face in style_faces], axis=0
        ).astype(np.float32)
        style_ensemble = [identity_ensemble.features(style_image, face) for face in style_faces]
        style_arcface = np.stack([item["arcface"] for item in style_ensemble]).astype(np.float32)
        style_adaface = np.stack([item["adaface"] for item in style_ensemble]).astype(np.float32)
        source_style_max = max(
            engine.cosine(source_feature, feature.reshape(1, -1))
            for feature in style_features
        )
    else:
        style_features = np.empty((0, source_feature.shape[1]), dtype=np.float32)
        style_arcface = np.empty((0, source_ensemble["arcface"].shape[-1]), dtype=np.float32)
        style_adaface = np.empty((0, source_ensemble["adaface"].shape[-1]), dtype=np.float32)
        source_style_max = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.output).expanduser().resolve() if args.output else DEFAULT_OUTPUT
    session = base / f"{timestamp}_{_safe_name(args.name)}"
    session.mkdir(parents=True, exist_ok=False)
    (session / "candidates").mkdir()
    (session / "private").mkdir()

    source_staged = session / f"图片1_本人{source_path.suffix.lower()}"
    shutil.copy2(source_path, source_staged)
    auxiliary_staged = []
    for index, path in enumerate(auxiliary_paths, start=2):
        staged = session / f"真人参考图_{index:02d}{path.suffix.lower()}"
        shutil.copy2(path, staged)
        auxiliary_staged.append(staged)
    privacy_mode = "mosaic" if style_path else "not_applicable"
    mosaic_blocks_used = max(1, int(getattr(args, "mosaic_blocks", 4)))
    masked_path = None
    mask_path = None
    regions = []
    residual_faces = []
    if style_image is not None:
        masked_style, regions, union = coarse_mosaic_faces(
            style_image,
            style_faces,
            side=args.mask_side,
            top=args.mask_top,
            bottom=args.mask_bottom,
            blocks_across=mosaic_blocks_used,
        )
        residual_faces = engine.detect_all(masked_style)
        while residual_faces and mosaic_blocks_used > 1:
            # 只能继续加粗马赛克，不允许改用纯色/不透明遮挡。
            mosaic_blocks_used -= 1
            masked_style, regions, union = coarse_mosaic_faces(
                style_image,
                style_faces,
                side=args.mask_side,
                top=args.mask_top,
                bottom=args.mask_bottom,
                blocks_across=mosaic_blocks_used,
            )
            residual_faces = engine.detect_all(masked_style)
        if residual_faces:
            raise ValueError(
                "最强马赛克后仍能检测到人脸。任务已安全停止；不会退回不透明遮挡，"
                "请换一张更清晰或脸部更大的图2后重试。"
            )
        masked_path = imwrite(session / "图片2_已彻底遮脸.png", masked_style)
        mask_path = imwrite(session / "图片2_遮罩审计图.png", union)
    features_path = session / "private" / "identity_features.npz"
    np.savez_compressed(
        features_path,
        source=source_feature,
        style=style_features,
        source_style_max=np.array(
            [source_style_max if source_style_max is not None else -1.0], dtype=np.float32
        ),
        source_arcface=source_ensemble["arcface"],
        source_adaface=source_ensemble["adaface"],
        style_arcface=style_arcface,
        style_adaface=style_adaface,
    )

    prompt_text = args.prompt
    if getattr(args, "prompt_file", None):
        prompt_text = Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    if not prompt_text or not prompt_text.strip():
        raise ValueError("提示词不能为空。")
    if style_preset:
        prompt_text = merge_style_prompt(prompt_text, style_preset)
    prompt_path = session / "提示词.txt"
    prompt_path.write_text(prompt_text.strip(), encoding="utf-8")
    scene_prompt_text, scene_prompt_audit = strip_identity_blocks(prompt_text)
    scene_prompt_path = session / "场景与质量模板.txt"
    scene_prompt_path.write_text(scene_prompt_text, encoding="utf-8")
    style_prompt_path = None
    if style_preset:
        style_prompt_path = session / "系统摄影风格.json"
        style_prompt_path.write_text(
            json.dumps(style_preset.to_metadata(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    reference_analysis_path = None
    if style_image is not None:
        reference_analysis_path = session / "参考图分析.json"
        reference_analysis = {
            "schema_version": "1.0",
            "analysis_role": "style_reference_only",
            "source_image": str(masked_path),
            "source_is_face_mosaicked": True,
            "image": {
                "width": int(masked_style.shape[1]),
                "height": int(masked_style.shape[0]),
                "aspect_ratio": round(masked_style.shape[1] / masked_style.shape[0], 6),
                "orientation": "portrait" if masked_style.shape[0] >= masked_style.shape[1] else "landscape",
            },
            "privacy": {
                "face_identity_policy": "discard",
                "face_regions": regions,
                "allowed_from_image2": ["pose", "clothing", "composition", "lighting", "filter", "furniture", "background"],
                "forbidden_from_image2": [
                    "identity", "facial_proportions", "age", "ethnicity", "eyes", "nose", "mouth", "jawline"
                ],
            },
            "semantic_analysis": {
                "status": "pending_visual_analysis",
                "composition": {}, "camera": {}, "pose": {}, "clothing": {},
                "lighting": {}, "color_grade": {}, "background_and_props": {},
            },
            "generation_policy": {
                "mode": "compact_priority_prompt",
                "never_dump_entire_json_into_prompt": True,
                "priority_order": ["image1_identity", "high_priority_pose_and_clothing", "lighting_and_grade"],
                "max_style_constraints": 8,
            },
        }
        reference_analysis_path.write_text(
            json.dumps(reference_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    job = {
        "version": "2.3",
        "mode": job_mode(str(style_path) if style_path else None),
        "style_source": {
            "kind": style_source_kind(style_value, style_preset_value),
            "preset_id": style_preset.preset_id if style_preset else None,
            "preset_title": style_preset.safe_title if style_preset else None,
            "compatibility": style_preset.compatibility if style_preset else None,
            "compatibility_label": style_preset.compatibility_label if style_preset else None,
            "compatibility_reasons": list(style_preset.compatibility_reasons) if style_preset else [],
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_dir": str(session),
        "source_image": str(source_staged),
        "masked_style_image": str(masked_path) if masked_path else None,
        "mask_audit_image": str(mask_path) if mask_path else None,
        "generation_reference_images": [str(source_staged)] + [str(path) for path in auxiliary_staged],
        "generation_reference_sha256": [sha256(source_path)] + [sha256(path) for path in auxiliary_paths],
        "generation_reference_count": 1 + len(auxiliary_paths),
        "feature_file": str(features_path),
        "prompt_file": str(prompt_path),
        "scene_prompt_file": str(scene_prompt_path),
        "initial_prompt_budget": scene_prompt_audit,
        "style_prompt_file": str(style_prompt_path) if style_prompt_path else None,
        "reference_analysis_file": str(reference_analysis_path) if reference_analysis_path else None,
        "source_sha256": sha256(source_path),
        "style_sha256": sha256(style_path) if style_path else None,
        "style_preset_sha256": sha256(
            DEFAULT_STYLE_LIBRARY / style_preset.source_file
        ) if style_preset else None,
        "style_original_path_stored": False,
        "style_preset_original_path_stored": False,
        "source_faces": len(source_faces),
        "source_face_detection": source_detection_audit,
        "style_faces_masked": len(style_faces),
        "style_face_detection": style_detection_audit,
        "privacy_mode": privacy_mode,
        "mosaic_blocks_across": mosaic_blocks_used,
        "residual_faces_after_mask": len(residual_faces),
        "mask_regions": regions,
        "source_style_reference_similarity": round(source_style_max, 6) if source_style_max is not None else None,
        "identity_scoring": {
            "models": ["SFace", "ArcFace", "AdaFace"],
            "method": "SFace 严格门槛；ArcFace 与 AdaFace 独立交叉检查；最终必须视觉复核",
            "secondary_mismatch_floor": 0.55,
        },
        "identity_profile": identity_profile,
        "threshold": args.threshold,
        "leakage_threshold": args.leakage_threshold,
        "identity_margin": args.identity_margin,
        "max_rolls": args.max_rolls,
    }
    job_path = session / "job.json"
    job["job_file"] = str(job_path)
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def _geo_metrics(face: np.ndarray) -> dict | None:
    """兼容旧任务的函数名；实际策略统一交给身份扫描路径控制器。"""
    return derive_face_metrics(face)


def evaluate_candidate(job_path: Path, candidate_path: Path) -> dict:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    image = imread(candidate_path)
    integrity = evaluate_image_integrity(image)
    result = {
        "candidate": str(candidate_path.resolve()),
        "candidate_sha256": sha256(candidate_path),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "style_source": job.get("style_source", {"kind": "image_reference" if job.get("mode") == "replacement" else "none"}),
        "image_integrity": integrity,
        "face_detected": None,
        "detected_face_count": None,
        "passed": False,
    }
    if not integrity["passed"]:
        result.update(
            source_similarity=None,
            style_similarity=None,
            identity_margin=None,
            gates={
                "quality": False,
                "observability": False,
                "embedding_identity": False,
                "identity_source": False,
                "anchor": False,
            },
            reject_reasons=list(integrity["reasons"]),
        )
        return result

    engine = FaceEngine()
    identity_ensemble = IdentityEnsemble()
    faces = engine.detect_all(image)
    result.update(
        face_detected=bool(faces),
        detected_face_count=len(faces),
        face_detection=engine.last_detection_audit,
    )
    if not faces:
        result.update(
            source_similarity=None,
            style_similarity=None,
            identity_margin=None,
            gates={
                "quality": True,
                "observability": False,
                "embedding_identity": False,
                "identity_source": False,
                "anchor": False,
            },
            reject_reasons=["候选图没有检测到人脸"],
        )
        return result

    feature_data = np.load(job["feature_file"])
    source_feature = feature_data["source"].astype(np.float32)
    style_features = feature_data["style"].astype(np.float32)
    has_style_reference = bool(style_features.size and job.get("mode") != "portrait")
    source_style_max = (
        float(feature_data["source_style_max"][0]) if has_style_reference else None
    )
    candidate_feature = engine.feature(image, faces[0])
    sface_source_similarity = engine.cosine(source_feature, candidate_feature)
    sface_style_similarity = (
        max(
            engine.cosine(feature.reshape(1, -1), candidate_feature)
            for feature in style_features
        )
        if has_style_reference
        else None
    )
    candidate_ensemble = identity_ensemble.features(image, faces[0])

    # 2.0 任务直接读取预计算特征；老任务仍可评分，但串脸检查只沿用 SFace。
    source_ensemble_available = all(
        key in feature_data for key in ("source_arcface", "source_adaface")
    )
    style_ensemble_available = has_style_reference and all(
        key in feature_data
        for key in ("style_arcface", "style_adaface")
    )
    if source_ensemble_available:
        source_model_scores = {
            "sface": sface_source_similarity,
            "arcface": identity_ensemble.cosine(
                feature_data["source_arcface"], candidate_ensemble["arcface"]
            ),
            "adaface": identity_ensemble.cosine(
                feature_data["source_adaface"], candidate_ensemble["adaface"]
            ),
        }
        if style_ensemble_available:
            style_model_scores = {
                "sface": sface_style_similarity,
                "arcface": max(
                    identity_ensemble.cosine(feature, candidate_ensemble["arcface"])
                    for feature in feature_data["style_arcface"]
                ),
                "adaface": max(
                    identity_ensemble.cosine(feature, candidate_ensemble["adaface"])
                    for feature in feature_data["style_adaface"]
                ),
            }
        else:
            style_model_scores = {}
        source_similarity = sface_source_similarity
        style_similarity = sface_style_similarity
        model_spread = float(max(source_model_scores.values()) - min(source_model_scores.values()))
        style_model_spread = (
            float(max(style_model_scores.values()) - min(style_model_scores.values()))
            if style_model_scores
            else 0.0
        )
    else:
        source_model_scores = {"sface": sface_source_similarity}
        style_model_scores = {"sface": sface_style_similarity} if has_style_reference else {}
        source_similarity = sface_source_similarity
        style_similarity = sface_style_similarity
        model_spread = 0.0
        style_model_spread = 0.0
    margin = source_similarity - style_similarity if style_similarity is not None else None

    requested_threshold = float(job["threshold"])
    threshold = requested_threshold
    leakage_threshold = float(job["leakage_threshold"])
    required_margin = float(job["identity_margin"])
    reasons: list[str] = []
    sface_gate = source_similarity >= threshold
    if not sface_gate:
        reasons.append(f"图1 SFace 身份相似度 {source_similarity:.3f} 低于 {threshold:.3f}")
    auxiliary_mismatch = (
        source_ensemble_available
        and source_model_scores["arcface"] < 0.55
        and source_model_scores["adaface"] < 0.55
    )
    if auxiliary_mismatch:
        reasons.append(
            "ArcFace 与 AdaFace 同时判为明显身份不符，不能只依赖 SFace 虚高分"
        )

    # 如果图1和图2本来就是同一人，无法用二者差值判断串脸，只做图1门槛。
    leakage_check_applicable = bool(
        has_style_reference and source_style_max is not None and source_style_max < 0.70
    )
    if leakage_check_applicable:
        leaked = style_similarity >= leakage_threshold and margin < required_margin
        if leaked:
            reasons.append(
                f"疑似串入图2身份：图2相似度 {style_similarity:.3f}，"
                f"图1领先仅 {margin:.3f}"
            )
    else:
        leaked = False

    x, y, width, height = [float(v) for v in faces[0][:4]]
    face_area_ratio = (width * height) / float(image.shape[0] * image.shape[1])
    face_size_ok = face_area_ratio >= 0.008
    if not face_size_ok:
        reasons.append("候选图人脸过小，评分可信度不足")
    significant_secondary = False
    if len(faces) > 1:
        primary_area = float(faces[0][2] * faces[0][3])
        significant_secondary = any(
            float(face[2] * face[3]) >= max(primary_area * 0.20, image.shape[0] * image.shape[1] * 0.002)
            for face in faces[1:]
        )
        if significant_secondary:
            reasons.append(f"候选图检测到 {len(faces)} 张明显人脸，存在双人或重影风险")

    # 身份扫描路径控制器：注意力结果正式参与拒绝与最终排序，不能再只是 Agent 提示。
    profile = job.get("identity_profile")
    if not profile:
        try:
            src_image = imread(job["source_image"])
            src_faces = engine.detect_all(src_image)
            if src_faces:
                src_metrics = derive_face_metrics(src_faces[0])
                if src_metrics:
                    profile = build_identity_profile(src_metrics)
        except Exception:
            profile = None
    candidate_metrics = derive_face_metrics(faces[0])
    candidate_dense_status = {
        "provider": "facemesh_468",
        "state": "reference_unavailable",
        "reason": "图1档案没有可比较的稠密指标，本轮使用五点身份锚点。",
        "experimental": True,
        "evidence_mode": "advisory",
    }
    reference_has_dense = any(
        str(key).startswith("dense_") for key in (profile or {}).get("metrics", {})
    )
    if candidate_metrics and reference_has_dense:
        candidate_dense_metrics, candidate_dense_status = _measure_optional_dense_metrics(
            image, faces[0], candidate_metrics
        )
        candidate_metrics.update(candidate_dense_metrics)
    attention = evaluate_attention(
        profile or {"metrics": {}},
        candidate_metrics,
        face_area_ratio,
        model_spread,
        identity_conflict={
            "active": leaked,
            "source_similarity": round(source_similarity, 6),
            "style_similarity": round(style_similarity, 6) if style_similarity is not None else None,
            "source_lead": round(margin, 6) if margin is not None else None,
        },
    )
    attention["dense_landmark_provider"] = {
        "reference": (profile or {}).get("dense_landmark_provider", {}),
        "candidate": candidate_dense_status,
    }
    if attention["decision"] == "reject":
        reasons.extend(
            reason for reason in attention["reject_reasons"] if reason not in reasons
        )
    elif attention["decision"] == "unreliable":
        # “不可可靠”不是“不是本人”，但也不能把无法验明身份的候选当作通过。
        # 把原因保留在 attention_report 中，便于下一轮优先改善构图/姿态，而不是改脸。
        reasons.append("身份锚点暂不可可靠判断，不能作为锁脸通过结果")

    gates = {
        "quality": True,
        "observability": bool(
            face_size_ok and not significant_secondary and attention["decision"] != "unreliable"
        ),
        "embedding_identity": bool(sface_gate and not auxiliary_mismatch),
        "identity_source": not leaked,
        "anchor": attention["decision"] != "reject",
    }

    result.update(
        source_similarity=round(source_similarity, 6),
        style_similarity=round(style_similarity, 6) if style_similarity is not None else None,
        requested_threshold=round(requested_threshold, 6),
        effective_sface_threshold=round(threshold, 6),
        identity_models={key: round(value, 6) for key, value in source_model_scores.items()},
        identity_model_spread=round(model_spread, 6),
        style_identity_models={key: round(value, 6) for key, value in style_model_scores.items()},
        style_identity_model_spread=round(style_model_spread, 6),
        scoring_mode=(
            "纯人像：SFace 门槛 + ArcFace/AdaFace 交叉检查 + 强制视觉复核"
            if not has_style_reference and source_ensemble_available
            else "换脸：SFace 门槛 + ArcFace/AdaFace 交叉检查 + 图2串脸拦截 + 强制视觉复核"
            if source_ensemble_available
            else "旧任务 SFace 兼容模式"
        ),
        identity_margin=round(margin, 6) if margin is not None else None,
        source_style_reference_similarity=(
            round(source_style_max, 6) if source_style_max is not None else None
        ),
        leakage_check_applicable=leakage_check_applicable,
        suspected_style_identity_leak=leaked,
        face_area_ratio=round(face_area_ratio, 6),
        gates=gates,
        reject_reasons=reasons,
        passed=all(gates.values()),
    )
    result["attention_priority"] = attention
    return result


def score_command(args: argparse.Namespace) -> dict:
    job_path = Path(args.job).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    result = evaluate_candidate(job_path, candidate_path)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else candidate_path.with_suffix(candidate_path.suffix + ".score.json")
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["score_file"] = str(output_path)
    return result


def _default_iteration_state_path(job: dict) -> Path:
    return Path(job["session_dir"]) / "iteration_state.json"


def _default_identity_anchor(job: dict | None = None) -> str:
    if (job or {}).get("mode") == "replacement":
        suffix = "图2只提供动作、服装、构图与光线，不得提供任何身份特征。"
    else:
        suffix = "系统摄影风格只提供画面约束，不得重新定义人物身份。"
    return "以图1人物为唯一身份来源，保持其整体脸型、五官空间关系和真实年龄感；" + suffix


def iterate_command(args: argparse.Namespace) -> dict:
    """评分候选并推进一次闭环；不绑定任何具体生图模型。"""
    job_path = Path(args.job).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    session = Path(job["session_dir"]).resolve()
    if not candidate_path.is_file():
        raise ValueError(f"候选图片不存在：{candidate_path}")

    state_path = (
        Path(args.state).expanduser().resolve()
        if args.state
        else _default_iteration_state_path(job)
    )
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = new_iteration_state(
            max_rounds=int(job.get("max_rolls", 6)),
            max_focus_attempts=int(args.max_focus_attempts),
        )

    score = evaluate_candidate(job_path, candidate_path)
    previous = (state.get("best") or {}).get("candidate")
    previous_path = Path(previous) if previous and Path(previous).is_file() else None
    focus = state.get("current_focus") or {}
    vision_judge = None
    if args.vision_config:
        try:
            judge_config = load_vision_judge_config(args.vision_config)
            provider = OpenAICompatibleVisionJudge(judge_config)
            vision_judge = provider.judge(
                job["source_image"],
                candidate_path,
                str(focus.get("instruction") or "整体身份"),
                previous_path,
            )
        except Exception as exc:
            vision_judge = {
                "kind": "vision_identity_judgement",
                "status": "unavailable",
                "error": str(exc),
                "hard_gate_enabled": False,
                "gate_mode": "local_fallback",
            }
    if vision_judge is not None:
        score["vision_judge"] = vision_judge

    next_state = advance_iteration(state, score, vision_judge)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(next_state, ensure_ascii=False, indent=2), encoding="utf-8")

    score_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else candidate_path.with_suffix(candidate_path.suffix + ".score.json")
    )
    score["iteration_decision"] = {
        "round": next_state["round"],
        "status": next_state["status"],
        "next_action": next_state.get("next_action"),
        "current_focus": next_state.get("current_focus"),
        "decision_reasons": next_state.get("decision_reasons", []),
    }
    score_path.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")

    next_prompt_path = None
    next_prompt_audit_path = None
    if next_state["status"] == "active":
        identity_anchor = _default_identity_anchor(job)
        if args.identity_anchor_file:
            identity_text = Path(args.identity_anchor_file).expanduser().resolve().read_text(
                encoding="utf-8"
            )
            identity_anchor = extract_identity_anchor(identity_text, identity_anchor)
        scene_prompt_value = job.get("scene_prompt_file")
        scene_prompt_file = Path(scene_prompt_value) if scene_prompt_value else None
        base_prompt_file = (
            scene_prompt_file
            if scene_prompt_file is not None and scene_prompt_file.is_file()
            else Path(job["prompt_file"])
        )
        base_prompt = base_prompt_file.read_text(encoding="utf-8")
        prompt, prompt_audit = compile_prompt_with_budget(
            base_prompt, identity_anchor, next_state
        )
        prompt_audit.update(
            round=next_state["round"],
            source_template=str(base_prompt_file),
            current_focus=(next_state.get("current_focus") or {}).get("key"),
        )
        next_prompt_path = session / "下一轮提示词.txt"
        next_prompt_path.write_text(prompt, encoding="utf-8")
        next_prompt_audit_path = session / "下一轮提示词.audit.json"
        next_prompt_audit_path.write_text(
            json.dumps(prompt_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return {
        "score_file": str(score_path),
        "state_file": str(state_path),
        "next_prompt_file": str(next_prompt_path) if next_prompt_path else None,
        "next_prompt_audit_file": (
            str(next_prompt_audit_path) if next_prompt_audit_path else None
        ),
        "status": next_state["status"],
        "next_action": next_state.get("next_action"),
        "current_focus": next_state.get("current_focus"),
        "best_candidate": (next_state.get("best") or {}).get("candidate"),
        "decision_reasons": next_state.get("decision_reasons", []),
        "vision_judge": vision_judge,
    }


def finalize_command(args: argparse.Namespace) -> dict:
    job_path = Path(args.job).expanduser().resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    session = Path(job["session_dir"])
    score_files = sorted((session / "candidates").glob("*.score.json"))
    if not score_files:
        raise ValueError("没有找到任何候选评分文件。")
    scores = [json.loads(path.read_text(encoding="utf-8")) for path in score_files]
    state_path = session / "iteration_state.json"
    iteration_state = None
    state_load_issue = None
    if state_path.is_file():
        try:
            iteration_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            state_load_issue = f"迭代状态文件不可用：{exc}"

    selection = select_final_candidate(
        scores,
        iteration_state=iteration_state,
        preferred_candidate=args.prefer,
    )
    best = selection["selected"]
    if state_load_issue and selection["selection_source"] == "machine_fallback":
        selection["fallback_reason"] = state_load_issue
    source = Path(best["candidate"])
    if not source.is_file():
        raise ValueError(f"最终候选图片不存在：{source}")
    final_path = session / f"最终结果{source.suffix.lower()}"
    shutil.copy2(source, final_path)
    report = {
        "status": "达标" if best.get("passed") else "未达标，保留最佳候选",
        "selection_mode": selection["selection_mode"],
        "selection_source": selection["selection_source"],
        "selection_fallback_reason": selection["fallback_reason"],
        "iteration_state_file": str(state_path) if state_path.is_file() else None,
        "selected": best,
        "all_scores": scores,
        "final_image": str(final_path),
        "job_file": str(job_path),
        "style_source": job.get("style_source", {"kind": "none"}),
    }
    report_path = session / "最终报告.json"
    report["report_file"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _inside_session(path: Path, session: Path) -> bool:
    try:
        path.resolve().relative_to(session.resolve())
        return True
    except ValueError:
        return False


def cleanup_command(args: argparse.Namespace) -> dict:
    """删除当前任务中可重建的私密派生数据，绝不触碰任务外的用户原图。"""
    job_path = Path(args.job).expanduser().resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    session = Path(job["session_dir"]).resolve()
    if job_path.parent != session:
        raise ValueError("job.json 不在其声明的任务目录中，拒绝清理。")

    removable = []
    for value in (
        job.get("source_image"),
        job.get("masked_style_image"),
        job.get("mask_audit_image"),
        job.get("feature_file"),
        job.get("prompt_file"),
        job.get("scene_prompt_file"),
        job.get("style_prompt_file"),
        job.get("reference_analysis_file"),
    ):
        if value:
            removable.append(Path(value))
    removable.extend(Path(path) for path in job.get("generation_reference_images", [])[1:])
    removable.extend(
        (
            session / "identity_锁脸段.txt",
            session / "下一轮提示词.txt",
            session / "下一轮提示词.audit.json",
            session / "candidates",
        )
    )
    removed = []
    for path in removable:
        if not path or not _inside_session(path, session) or not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path.relative_to(session)))
    private_dir = session / "private"
    if private_dir.exists() and _inside_session(private_dir, session):
        shutil.rmtree(private_dir)
        removed.append("private/")

    report_path = session / "最终报告.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["privacy_cleanup"] = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "removed": removed,
            "preserved": ["最终结果.*", "最终报告.json", "job.json", "运行日志.txt"],
            "note": "未删除任务外的原图；已删除任务内复制件、身份向量、身份提示词、遮脸参考与候选图。",
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"session_dir": str(session), "removed": removed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="人脸遮蔽、相似度评分与串脸拦截")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="建立纯人像任务，或遮蔽图2后建立换脸任务")
    add_identity_reference_arguments(prepare)
    prompt_group = prepare.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="本次生图提示词")
    prompt_group.add_argument("--prompt-file", help="UTF-8 提示词文件")
    prepare.add_argument("--name", default="写真重roll", help="任务名称")
    prepare.add_argument("--output", help="任务输出根目录")
    prepare.add_argument("--threshold", type=float, default=0.90, help="图1身份相似度门槛")
    prepare.add_argument("--leakage-threshold", type=float, default=0.70, help="图2串脸警戒线")
    prepare.add_argument("--identity-margin", type=float, default=0.08, help="图1相对图2最低领先值")
    prepare.add_argument("--max-rolls", type=int, default=6, help="最多生成次数")
    prepare.add_argument(
        "--privacy-mode", choices=("mosaic",), default="mosaic", help="固定为超粗马赛克"
    )
    prepare.add_argument("--mosaic-blocks", type=int, default=4, help="马赛克横向色块数")
    prepare.add_argument("--mask-side", type=float, default=0.30, help="遮罩左右扩张比例")
    prepare.add_argument("--mask-top", type=float, default=0.45, help="遮罩向上扩张比例")
    prepare.add_argument("--mask-bottom", type=float, default=0.28, help="遮罩向下扩张比例")

    score = subparsers.add_parser("score", help="评分一个候选图")
    score.add_argument("--job", required=True, help="job.json 路径")
    score.add_argument("--candidate", required=True, help="候选图片路径")
    score.add_argument("--output", help="评分 JSON 输出路径")

    iterate = subparsers.add_parser("iterate", help="评分候选并推进单重点闭环")
    iterate.add_argument("--job", required=True, help="job.json 路径")
    iterate.add_argument("--candidate", required=True, help="本轮候选图片路径")
    iterate.add_argument("--state", help="迭代状态 JSON；默认写入任务目录")
    iterate.add_argument("--output", help="本轮评分 JSON 输出路径")
    iterate.add_argument("--identity-anchor-file", help="P0/P1/P2 分级身份基线文字文件")
    iterate.add_argument("--vision-config", help="含 vision_judge 配置的 JSON；不填则纯本地评分")
    iterate.add_argument("--max-focus-attempts", type=int, default=3, help="同一重点最多尝试次数")

    finalize = subparsers.add_parser("finalize", help="选出最高分候选并写报告")
    finalize.add_argument("--job", required=True, help="job.json 路径")
    finalize.add_argument("--prefer", help="视觉复核后指定候选图；仍保留机器分数与是否达标")
    cleanup = subparsers.add_parser("cleanup", help="清理当前任务内可重建的私密派生数据")
    cleanup.add_argument("--job", required=True, help="已完成任务的 job.json 路径")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_job(args)
        elif args.command == "score":
            result = score_command(args)
        elif args.command == "iterate":
            result = iterate_command(args)
        elif args.command == "finalize":
            result = finalize_command(args)
        else:
            result = cleanup_command(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
