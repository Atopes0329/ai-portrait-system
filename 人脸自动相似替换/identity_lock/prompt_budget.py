"""把完整首轮提示词编译为不累积身份信息的单重点迭代提示词。"""

from __future__ import annotations

import re
from typing import Any, Mapping


IDENTITY_BLOCK_PREFIXES = (
    "【P0/P1/P2 身份基线】",
    "【P0/P1/P2身份基线】",
    "【身份基线】",
    "## 首轮身份基线",
    "### 首轮身份基线",
    "基线锁脸段成品：",
    "锁脸段成品：",
    "身份锚点成品：",
    "身份基线：",
    "本轮唯一身份修正目标：",
    "身份优先规则：",
)

FINAL_IDENTITY_RULE = (
    "身份优先规则：真人参考图是唯一身份来源；固定身份基线不得重写。"
    "除上述当前目标外，不新增其他五官要求，不得为了美化改变已经正确的脸型与五官关系。"
)


def _blocks(text: str) -> list[str]:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [block.strip() for block in re.split(r"\n[ \t]*\n+", normalized) if block.strip()]


def _is_identity_block(block: str) -> bool:
    compact = block.lstrip()
    return any(compact.startswith(prefix) for prefix in IDENTITY_BLOCK_PREFIXES)


def strip_identity_blocks(prompt: str) -> tuple[str, dict[str, Any]]:
    """只剥离带明确结构标记的身份块，普通自由文本保持原样。"""
    kept: list[str] = []
    removed: list[str] = []
    for block in _blocks(prompt):
        (removed if _is_identity_block(block) else kept).append(block)
    scene_prompt = "\n\n".join(kept)
    original = str(prompt or "").strip()
    return scene_prompt, {
        "schema_version": "1.0",
        "kind": "prompt_scene_separation",
        "original_chars": len(original),
        "scene_chars": len(scene_prompt),
        "removed_identity_block_count": len(removed),
        "removed_identity_chars": sum(len(block) for block in removed),
        "removed_block_prefixes": [block.splitlines()[0][:80] for block in removed],
    }


def compile_prompt_with_budget(
    base_prompt: str,
    identity_anchor: str,
    state: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """从无历史状态的场景模板编译一轮提示词，并给出预算审计。"""
    scene_prompt, separation = strip_identity_blocks(base_prompt)
    anchor = str(identity_anchor or "").strip()
    focus = state.get("current_focus") or {}
    focus_instruction = str(focus.get("instruction") or "").strip()
    focus_block = (
        "本轮唯一身份修正目标：" + focus_instruction if focus_instruction else ""
    )
    sections = [anchor, focus_block, scene_prompt, FINAL_IDENTITY_RULE]
    prompt = "\n\n".join(section for section in sections if section)
    anchor_occurrences = prompt.count(anchor) if anchor else 0
    focus_occurrences = prompt.count("本轮唯一身份修正目标：")
    audit = {
        "schema_version": "1.0",
        "kind": "prompt_budget_report",
        "total_chars": len(prompt),
        "section_chars": {
            "identity_anchor": len(anchor),
            "current_focus": len(focus_block),
            "scene_and_quality": len(scene_prompt),
            "identity_rule": len(FINAL_IDENTITY_RULE),
        },
        "identity_anchor_occurrences": anchor_occurrences,
        "focus_occurrences": focus_occurrences,
        "within_single_focus_budget": bool(
            anchor_occurrences == (1 if anchor else 0)
            and focus_occurrences == (1 if focus_instruction else 0)
        ),
        "warnings": [
            warning
            for warning, active in (
                ("身份基线超过 500 字；本轮仅去重，不自动截短已验证基线", len(anchor) > 500),
                ("提示词仍超过 900 字；建议进入提示词密度 A/B 校准", len(prompt) > 900),
            )
            if active
        ],
        "scene_separation": separation,
    }
    return prompt, audit
