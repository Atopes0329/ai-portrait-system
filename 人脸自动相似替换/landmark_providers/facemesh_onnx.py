"""旧 Skill 的 Face Mesh ONNX 实验适配器。

第三方依赖只在真正加载模型时导入。模型缺失或异常时返回结构化状态，调用方
可以无损退回 YuNet 五点机制。
"""

from __future__ import annotations

from functools import lru_cache
from math import exp
from pathlib import Path
from typing import Any

from identity_lock import derive_dense_metrics
from model_manifest import EXPERIMENTAL_MODELS, compute_sha256


MODEL_NAME = "face_mesh_Nx3x192x192.onnx"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _status(state: str, model_path: Path, reason: str) -> dict[str, Any]:
    return {
        "provider": "facemesh_468",
        "state": state,
        "reason": reason,
        "model": str(model_path),
        "experimental": True,
        "evidence_mode": "advisory",
    }


class FaceMeshProvider:
    """只负责推理和坐标还原，身份含义由 identity_lock 解释。"""

    def __init__(self, model_path: Path):
        import cv2
        import numpy as np
        import onnxruntime as ort

        self.cv2 = cv2
        self.np = np
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_names = [item.name for item in self.session.get_outputs()]
        shape = model_input.shape
        size = shape[-1] if shape and isinstance(shape[-1], int) else 192
        self.size = int(size)

    @staticmethod
    def _ordered_eyes(face: Any) -> Any:
        points = face[4:8].reshape(2, 2).astype("float32")
        return points[points[:, 0].argsort()]

    @staticmethod
    def _presence(raw_value: float) -> float:
        value = max(-60.0, min(60.0, float(raw_value)))
        return 1.0 / (1.0 + exp(-value))

    def landmarks(self, image: Any, face: Any) -> tuple[Any, float]:
        cv2 = self.cv2
        np = self.np
        x, y, width, height = [float(value) for value in face[:4]]
        if width <= 0 or height <= 0:
            raise ValueError("Face Mesh 输入人脸框无效。")
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        side = 1.5 * max(width, height)
        eyes = self._ordered_eyes(face)
        eye_delta = eyes[1] - eyes[0]
        angle = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))

        matrix = cv2.getRotationMatrix2D(
            (center_x, center_y), angle, self.size / side
        )
        matrix[0, 2] += self.size / 2.0 - center_x
        matrix[1, 2] += self.size / 2.0 - center_y
        crop = cv2.warpAffine(image, matrix, (self.size, self.size))
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        outputs = self.session.run(self.output_names, {self.input_name: tensor})

        landmark_arrays = [array for array in outputs if int(array.size) >= 468 * 3]
        if not landmark_arrays:
            raise ValueError("Face Mesh 输出中没有找到 468×3 landmark。")
        raw_points = min(landmark_arrays, key=lambda array: int(array.size))
        points = raw_points.reshape(-1, 3)[:468].astype(np.float64)

        scalar_outputs = [array for array in outputs if int(array.size) < 468 * 3]
        presence = 1.0
        if scalar_outputs and int(scalar_outputs[0].size):
            presence = self._presence(float(scalar_outputs[0].reshape(-1)[0]))

        inverse = cv2.invertAffineTransform(matrix)
        points[:, :2] = points[:, :2] @ inverse[:, :2].T + inverse[:, 2]
        points[:, 2] *= side / self.size
        return points.astype(np.float32), presence

    def measure(self, image: Any, face: Any) -> dict[str, Any]:
        points, presence = self.landmarks(image, face)
        metrics = derive_dense_metrics(points, presence)
        return {
            "metrics": metrics,
            "presence": round(float(presence), 6),
            "landmark_count": int(len(points)),
        }


@lru_cache(maxsize=4)
def _load_provider(model_path_text: str) -> FaceMeshProvider:
    return FaceMeshProvider(Path(model_path_text))


def get_facemesh_provider(
    models_dir: Path | None = None,
) -> tuple[FaceMeshProvider | None, dict[str, Any]]:
    """安全加载可选模型；任何失败都变成可解释降级状态。"""
    directory = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    model_path = directory / MODEL_NAME
    spec = EXPERIMENTAL_MODELS[MODEL_NAME]
    if not model_path.is_file():
        return None, _status("missing", model_path, "可选 Face Mesh 模型未安装，继续使用 YuNet 五点身份锚点。")
    size = model_path.stat().st_size
    if not spec["size_range"][0] <= size <= spec["size_range"][1]:
        return None, _status("invalid", model_path, "Face Mesh 模型大小异常，已拒绝加载并降级。")
    if compute_sha256(model_path) != spec["sha256"]:
        return None, _status("invalid", model_path, "Face Mesh 模型 SHA256 不匹配，已拒绝加载并降级。")
    try:
        provider = _load_provider(str(model_path.resolve()))
    except (ImportError, ModuleNotFoundError) as exc:
        return None, _status("dependency_missing", model_path, f"Face Mesh 运行依赖缺失：{exc}")
    except Exception as exc:
        return None, _status("load_failed", model_path, f"Face Mesh 加载失败：{exc}")
    return provider, _status("ready", model_path, "Face Mesh 稠密建议层已启用。")
