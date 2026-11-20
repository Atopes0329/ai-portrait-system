"""模型无关的锁人脸核心。"""

from .attention import build_identity_profile, derive_face_metrics, evaluate_attention
from .dense_metrics import derive_dense_metrics
from .iteration import (
    advance_iteration,
    compose_iteration_prompt,
    extract_identity_anchor,
    new_iteration_state,
)
from .selection import select_final_candidate
from .prompt_budget import compile_prompt_with_budget, strip_identity_blocks

__all__ = [
    "build_identity_profile",
    "advance_iteration",
    "compose_iteration_prompt",
    "derive_dense_metrics",
    "derive_face_metrics",
    "evaluate_attention",
    "extract_identity_anchor",
    "new_iteration_state",
    "select_final_candidate",
    "compile_prompt_with_budget",
    "strip_identity_blocks",
]
