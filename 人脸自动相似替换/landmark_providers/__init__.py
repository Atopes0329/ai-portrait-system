"""具体 landmark 模型的可选适配器。"""

from .facemesh_onnx import FaceMeshProvider, get_facemesh_provider

__all__ = ["FaceMeshProvider", "get_facemesh_provider"]
