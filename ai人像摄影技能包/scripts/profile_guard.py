#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人物文字档案的状态、哈希与黄金回归标尺管理。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(profile: str | Path) -> Path:
    path = Path(profile).expanduser().resolve()
    return path.with_suffix(path.suffix + ".profile.json")


def _asset(path: str | Path, role: str) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"文件不存在：{resolved}")
    return {
        "name": resolved.name,
        "sha256": sha256(resolved),
        "role": role,
    }


def load_metadata(profile: str | Path) -> dict | None:
    metadata_path = sidecar_path(profile)
    if not metadata_path.is_file():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def check_profile(profile: str | Path) -> dict:
    profile_path = Path(profile).expanduser().resolve()
    if not profile_path.is_file():
        raise ValueError(f"人物档案不存在：{profile_path}")
    metadata = load_metadata(profile_path)
    if not metadata:
        return {
            "ok": True,
            "status": "untracked",
            "profile": str(profile_path),
            "message": "档案没有状态记录，只能按 provisional 使用。",
        }
    expected = metadata.get("profile_sha256")
    actual = sha256(profile_path)
    intact = bool(expected and expected == actual)
    return {
        "ok": intact,
        "status": metadata.get("status", "provisional"),
        "profile": str(profile_path),
        "profile_intact": intact,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "metadata_file": str(sidecar_path(profile_path)),
        "golden_regression": metadata.get("golden_regression"),
    }


def guard_output(profile: str | Path, replace_provisional: bool = False) -> None:
    """在调用视觉接口前拦截覆盖，避免花费额度后才发现档案受保护。"""
    profile_path = Path(profile).expanduser().resolve()
    if not profile_path.exists():
        return
    report = check_profile(profile_path)
    if report["status"] == "validated":
        if not report.get("profile_intact"):
            raise ValueError("已验证人物档案的内容或哈希记录发生变化，拒绝继续覆盖。")
        raise ValueError(
            "目标是 validated 人物档案，自动重分析不得覆盖。请换一个 --out 路径保存候选档案。"
        )
    if not replace_provisional:
        raise ValueError(
            "目标档案已存在。为避免静默覆盖，请换一个 --out 路径；"
            "确实只想替换 provisional 档案时显式加 --replace-provisional。"
        )


def write_provisional_metadata(
    profile: str | Path,
    references: Iterable[str | Path],
    model: str,
) -> dict:
    profile_path = Path(profile).expanduser().resolve()
    metadata = {
        "schema_version": "1.0",
        "status": "provisional",
        "profile_sha256": sha256(profile_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "vision_model": model,
        "identity_source_policy": "real_reference_photos_only",
        "references": [_asset(path, "real_identity_source") for path in references],
        "golden_regression": None,
    }
    sidecar_path(profile_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def validate_profile(
    profile: str | Path,
    references: Iterable[str | Path],
    golden_output: str | Path | None,
    note: str | None = None,
) -> dict:
    profile_path = Path(profile).expanduser().resolve()
    refs = list(references)
    if not profile_path.is_file():
        raise ValueError(f"人物档案不存在：{profile_path}")
    if not refs:
        raise ValueError("validated 档案必须登记至少一张真人参考照片。")
    golden = _asset(golden_output, "regression_only") if golden_output else None
    if golden:
        golden.update(identity_source=False, validated_by="human_confirmation")
    metadata = {
        "schema_version": "1.0",
        "status": "validated",
        "profile_sha256": sha256(profile_path),
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "validated_by": "human_confirmation",
        "identity_source_policy": "real_reference_photos_only",
        "references": [_asset(path, "real_identity_source") for path in refs],
        "golden_regression": golden,
        "note": note or "",
    }
    metadata_path = sidecar_path(profile_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["metadata_file"] = str(metadata_path)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="人物档案状态与黄金回归标尺管理")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="校验人物档案状态与哈希")
    check.add_argument("--profile", required=True)
    validate = subparsers.add_parser("validate", help="经真人确认后把档案标记为 validated")
    validate.add_argument("--profile", required=True)
    validate.add_argument("--reference", action="append", required=True)
    validate.add_argument("--golden-output", help="已确认相似的成片；仅登记为回归标尺")
    validate.add_argument("--note")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "check":
            result = check_profile(args.profile)
        else:
            result = validate_profile(
                args.profile, args.reference, args.golden_output, args.note
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok", True) else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
