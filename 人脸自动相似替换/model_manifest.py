"""人脸评分模型的唯一清单。

下载、校验和初始化都从这里读取哈希，防止三份清单逐渐失真。
"""

from __future__ import annotations

from pathlib import Path
import hashlib


RUNTIME_MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "size_range": (200_000, 300_000),
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    },
    "face_recognition_sface_2021dec.onnx": {
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size_range": (35_000_000, 40_000_000),
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    },
    "arcfaceresnet100-11-int8.onnx": {
        "sha256": "c625ca68a422418c48aa84f73341337e0a92b111f327909005d1eec07c95f936",
        "size_range": (60_000_000, 70_000_000),
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11-int8.onnx",
    },
    "adaface_ir_18.onnx": {
        "sha256": "6b6a35772fb636cdd4fa86520c1a259d0c41472a76f70f802b351837a00d9870",
        "size_range": (90_000_000, 100_000_000),
        "url": "https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_18.onnx",
    },
}

EXPERIMENTAL_MODELS = {
    "face_mesh_Nx3x192x192.onnx": {
        "sha256": "3ca77cf59c18e4da0eccb46695bf604683fa564253e3385892981a5c274fb10f",
        "size_range": (2_000_000, 3_000_000),
        "manual_instruction": "可选稠密建议模型：从 https://github.com/yakhyo/uniface 获取；用于身份注意力增强与 model_bakeoff.py 校准。",
    },
}


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def invalid_runtime_models(models_dir: Path) -> list[str]:
    """返回缺失、大小异常或哈希不匹配的运行模型。"""
    invalid = []
    for name, spec in RUNTIME_MODELS.items():
        path = models_dir / name
        if not path.is_file() or not (spec["size_range"][0] <= path.stat().st_size <= spec["size_range"][1]):
            invalid.append(name)
            continue
        if compute_sha256(path) != spec["sha256"]:
            invalid.append(name)
    return invalid
