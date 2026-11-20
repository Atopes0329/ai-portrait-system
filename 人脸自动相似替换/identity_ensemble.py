#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArcFace + AdaFace 的本地身份交叉检查。

两种模型都只接收已经由 YuNet 找到并五点对齐的人脸；所有推理均在本机完成。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
ARCFACE_MODEL = ROOT / "models" / "arcfaceresnet100-11-int8.onnx"
ADAFACE_MODEL = ROOT / "models" / "adaface_ir_18.onnx"
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def ordered_landmarks(face: np.ndarray) -> np.ndarray:
    """将 YuNet 五点统一为左眼、右眼、鼻、左嘴角、右嘴角。"""
    points = face[4:14].reshape(5, 2).astype(np.float32)
    eyes = points[:2][np.argsort(points[:2, 0])]
    mouth = points[3:5][np.argsort(points[3:5, 0])]
    return np.vstack((eyes, points[2], mouth)).astype(np.float32)


def align_face(image: np.ndarray, face: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(
        ordered_landmarks(face), ARCFACE_TEMPLATE, method=cv2.LMEDS
    )
    if matrix is None:
        raise ValueError("人脸五点对齐失败")
    return cv2.warpAffine(
        image,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _normalize(feature: np.ndarray) -> np.ndarray:
    feature = feature.astype(np.float32).reshape(-1)
    return feature / max(float(np.linalg.norm(feature)), 1e-9)


class IdentityEnsemble:
    """同时运行 ArcFace 与 AdaFace，避免单一模型误判。"""

    def __init__(self):
        missing = [path for path in (ARCFACE_MODEL, ADAFACE_MODEL) if not path.is_file()]
        if missing:
            raise FileNotFoundError("身份交叉检查模型缺失：" + "、".join(str(path) for path in missing))
        self.arcface = ort.InferenceSession(
            str(ARCFACE_MODEL), providers=["CPUExecutionProvider"]
        )
        self.adaface = ort.InferenceSession(
            str(ADAFACE_MODEL), providers=["CPUExecutionProvider"]
        )

    def features(self, image: np.ndarray, face: np.ndarray) -> dict[str, np.ndarray]:
        aligned = align_face(image, face)

        # ONNX Model Zoo 的 LResNet100E-IR 示例输入是 0..255 RGB CHW，
        # 模型本身已经包含所需的输入归一化。
        arc_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        arc_blob = arc_rgb.astype(np.float32)
        arc_blob = np.transpose(arc_blob, (2, 0, 1))[None, ...]
        arc_input = self.arcface.get_inputs()[0].name
        arc_output = self.arcface.get_outputs()[0].name
        arc_feature = self.arcface.run([arc_output], {arc_input: arc_blob})[0][0]

        # AdaFace 官方权重使用 BGR；不能与 ArcFace 一样交换通道。
        ada_blob = cv2.dnn.blobFromImage(
            aligned,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
        )
        ada_input = self.adaface.get_inputs()[0].name
        ada_output = self.adaface.get_outputs()[0].name
        ada_feature = self.adaface.run([ada_output], {ada_input: ada_blob})[0][0]
        return {"arcface": _normalize(arc_feature), "adaface": _normalize(ada_feature)}

    @staticmethod
    def cosine(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.dot(first.reshape(-1), second.reshape(-1)))
