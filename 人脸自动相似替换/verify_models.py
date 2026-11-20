#!/usr/bin/env python3
"""
验证 models/ 目录中的所有模型文件
"""
from pathlib import Path
import sys

from model_manifest import EXPERIMENTAL_MODELS, RUNTIME_MODELS, compute_sha256

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

EXPECTED_MODELS = {**RUNTIME_MODELS, **EXPERIMENTAL_MODELS}


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f} MB"


def main():
    print("=" * 70)
    print("模型文件验证")
    print("=" * 70)
    print()

    if not MODELS_DIR.exists():
        print(f"✗ models/ 目录不存在: {MODELS_DIR}")
        print()
        print("请先运行: .venv/bin/python download_models.py")
        return 1

    all_valid = True
    missing_critical = False

    for model_name, expected in EXPECTED_MODELS.items():
        model_path = MODELS_DIR / model_name
        print(f"[{model_name}]")

        # 检查文件是否存在
        if not model_path.exists():
            print(f"  ✗ 文件不存在")
            all_valid = False
            if model_name in RUNTIME_MODELS:
                missing_critical = True
            print()
            continue

        # 检查文件大小
        file_size = model_path.stat().st_size
        size_ok = expected["size_range"][0] <= file_size <= expected["size_range"][1]
        print(f"  大小: {format_size(file_size)} {'✓' if size_ok else '✗ 异常'}")

        if not size_ok:
            all_valid = False
            print(f"  预期范围: {format_size(expected['size_range'][0])} - {format_size(expected['size_range'][1])}")

        # 计算并验证 SHA256
        print(f"  计算 SHA256...")
        actual_hash = compute_sha256(model_path)
        hash_ok = actual_hash == expected["sha256"]

        if hash_ok:
            print(f"  SHA256: ✓ 匹配")
        else:
            print(f"  SHA256: ✗ 不匹配")
            print(f"    预期: {expected['sha256']}")
            print(f"    实际: {actual_hash}")
            all_valid = False

        print()

    # 总结
    print("=" * 70)

    critical_models = list(RUNTIME_MODELS)

    critical_ok = all(
        (MODELS_DIR / name).exists()
        and compute_sha256(MODELS_DIR / name) == EXPECTED_MODELS[name]["sha256"]
        for name in critical_models
    )

    if all_valid:
        print("✓ 所有模型验证通过")
        return 0
    elif critical_ok:
        print("✓ 所有运行时模型验证通过")
        print("  (部分可选模型缺失或损坏)")
        return 0
    else:
        print("✗ 验证失败")
        if missing_critical:
            print("  核心模型缺失，请运行: .venv/bin/python download_models.py")
        else:
            print("  部分模型损坏，请重新下载")
        return 1


if __name__ == "__main__":
    sys.exit(main())
