"""任务入口的模型无关参数契约。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_AUXILIARY_REFERENCES = 7


def add_identity_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="图片1：本人正脸")
    parser.add_argument(
        "--reference", action="append", default=[], help="纯人像辅助真人参考图，可重复传入"
    )
    style_group = parser.add_mutually_exclusive_group()
    style_group.add_argument("--style", help="可选图2：动作/服饰/滤镜参考；不填即纯人像模式")
    style_group.add_argument(
        "--style-preset", help="可选系统摄影风格 Markdown；与图2互斥"
    )


def job_mode(style: str | None) -> str:
    return "replacement" if style else "portrait"


def style_source_kind(style: str | None, style_preset: str | None) -> str:
    if style:
        return "image_reference"
    if style_preset:
        return "prompt_library"
    return "none"


def normalize_auxiliary_references(
    source: str | Path | None,
    references: Sequence[str | Path],
    *,
    require_exists: bool = True,
    max_references: int = MAX_AUXILIARY_REFERENCES,
) -> list[str]:
    """规范化辅助真人参考；自动去重并排除图1本身。"""
    source_path = (
        Path(source).expanduser().resolve(strict=False) if source else None
    )
    result: list[str] = []
    seen: set[str] = set()
    for raw in references:
        if not str(raw).strip():
            continue
        path = Path(raw).expanduser().resolve(strict=False)
        if source_path is not None and path == source_path:
            continue
        normalized = str(path)
        if normalized in seen:
            continue
        if require_exists and not path.is_file():
            raise ValueError(f"辅助真人参考图不存在：{path}")
        seen.add(normalized)
        result.append(normalized)
    if len(result) > max_references:
        raise ValueError(f"辅助真人参考图最多允许 {max_references} 张")
    return result


def generation_attachments(job: Mapping[str, Any]) -> list[str]:
    """返回 Agent 可接收的图片；替换模式固定守住两图隐私边界。"""
    style_kind = (job.get("style_source") or {}).get("kind")
    source = job.get("source_image")
    if style_kind == "image_reference" or job.get("mode") == "replacement":
        return [str(path) for path in (source, job.get("masked_style_image")) if path]
    references = job.get("generation_reference_images") or [source]
    return [str(path) for path in references if path]
