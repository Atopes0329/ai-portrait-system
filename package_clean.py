#!/usr/bin/env python3
"""生成不含密钥、人物档案、照片、成片与运行记录的干净分发目录。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "迭代进化.分发模板.md"
EXCLUDED_NAMES = {
    ".git",
    ".DS_Store",
    ".venv",
    "__pycache__",
    "config.json",
    "任务输出",
    "本地设置.json",
    "tasks",
}
PRIVATE_RELATIVE_DIRS = {
    Path("ai人像摄影技能包/特征库"),
    Path("ai人像摄影技能包/出图"),
    Path("ai人像摄影技能包/示例"),
}
PRIVATE_RELATIVE_FILES = {
    Path("迭代进化.md"),
}
SECRET_PATTERN = re.compile(rb"sk-[A-Za-z0-9_-]{20,}")


def ignore(directory: str, names: list[str]) -> set[str]:
    current = Path(directory).resolve()
    ignored = {name for name in names if name in EXCLUDED_NAMES or name.endswith(".pyc")}
    for name in names:
        source = current / name
        try:
            relative = source.relative_to(ROOT)
        except ValueError:
            continue
        if relative in PRIVATE_RELATIVE_DIRS or relative in PRIVATE_RELATIVE_FILES:
            ignored.add(name)
    return ignored


def verify_clean(destination: Path) -> dict:
    violations = []
    for path in destination.rglob("*"):
        relative = path.relative_to(destination)
        if path.name in EXCLUDED_NAMES or path.name.endswith(".pyc"):
            violations.append(str(relative))
        if path.is_file() and path.stat().st_size <= 5 * 1024 * 1024:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if SECRET_PATTERN.search(data):
                violations.append(f"possible-secret:{relative}")
    for relative in PRIVATE_RELATIVE_DIRS:
        directory = destination / relative
        if any(path.is_file() for path in directory.rglob("*")):
            violations.append(f"private-data:{relative}")
    evolution = destination / "迭代进化.md"
    if not evolution.is_file() or evolution.read_text(encoding="utf-8") != TEMPLATE.read_text(encoding="utf-8"):
        violations.append("迭代进化.md 不是空白分发模板")
    if violations:
        raise ValueError("干净分发校验失败：" + "; ".join(violations))
    return {
        "status": "clean",
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "file_count": sum(1 for path in destination.rglob("*") if path.is_file()),
        "excluded": sorted(EXCLUDED_NAMES),
        "private_directories_emptied": sorted(str(path) for path in PRIVATE_RELATIVE_DIRS),
    }


def build(destination: Path) -> dict:
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"目标已存在，拒绝覆盖：{destination}")
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("目标不能是源目录或源目录内部。")
    shutil.copytree(ROOT, destination, ignore=ignore)
    for relative in PRIVATE_RELATIVE_DIRS:
        (destination / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE, destination / "迭代进化.md")
    report = verify_clean(destination)
    audit_path = destination / "DISTRIBUTION_AUDIT.json"
    report["audit_file"] = audit_path.name
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 AI 人像系统干净分发目录")
    parser.add_argument("destination", help="必须是尚不存在的新目录")
    args = parser.parse_args()
    try:
        report = build(Path(args.destination))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
