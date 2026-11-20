#!/usr/bin/env python3
"""
自动下载所有必需的 ONNX 模型到 models/ 目录
"""
from http.client import HTTPException
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

from model_manifest import EXPERIMENTAL_MODELS, RUNTIME_MODELS, compute_sha256

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# 运行时模型的上游下载地址和哈希由 model_manifest.py 统一维护。
MODELS = [
    {"name": name, "url": spec["url"], "sha256": spec["sha256"],
     "size_range": spec["size_range"]}
    for name, spec in RUNTIME_MODELS.items()
    if spec.get("url")
]

# GitHub 直连在部分地区会被限速，准备镜像源自动兜底
GH_PROXY_PREFIXES = [
    "https://gh-proxy.com/",
    "https://ghfast.top/",
    "https://ghproxy.net/",
]

MANUAL_MODELS = [
    {"name": name, "sha256": spec["sha256"],
     "instruction": spec.get("manual_instruction", "请重新获取该模型。"),
     "required": name in RUNTIME_MODELS}
    for name, spec in {**RUNTIME_MODELS, **EXPERIMENTAL_MODELS}.items()
    if not spec.get("url")
]


def download_file(model, destination: Path) -> bool:
    """先验证下载文件，成功后再替换 models 目录中的副本。"""
    url = model["url"]
    candidates = [url]
    if "github.com/" in url:
        candidates.extend(prefix + url for prefix in GH_PROXY_PREFIXES)
    for candidate in candidates:
        print(f"  下载中: {candidate}")
        try:
            with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                pending = Path(temporary) / destination.name
                with urllib.request.urlopen(candidate, timeout=30) as response, pending.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                low, high = model["size_range"]
                if not low <= pending.stat().st_size <= high:
                    raise ValueError("文件大小不符")
                if not verify_sha256(pending, model["sha256"]):
                    raise ValueError("SHA-256 不符")
                pending.replace(destination)
            return True
        except (HTTPException, OSError, ValueError) as error:
            print(f"  源失败（{error}），尝试下一个…")
    return False


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """验证文件的 SHA256。"""
    return compute_sha256(file_path) == expected_hash


def main():
    # 创建 models 目录
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("人脸自动相似替换 - 模型下载工具")
    print("=" * 70)
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    # 下载自动模型
    for model in MODELS:
        dest_path = MODELS_DIR / model["name"]
        print(f"[{model['name']}]")

        if dest_path.exists():
            print(f"  文件已存在，验证中...")
            if verify_sha256(dest_path, model["sha256"]):
                print(f"  ✓ 校验通过，跳过下载")
                skip_count += 1
                print()
                continue
            print("  ✗ 校验失败，重新下载")

        if download_file(model, dest_path):
            print("  ✓ SHA-256 校验通过")
            success_count += 1
        else:
            print("  ✗ 所有下载源均未提供符合校验的文件")
            fail_count += 1
        print()

    # 检查不能自动下载的模型
    print("-" * 70)
    print("需要手动下载的模型:")
    print("-" * 70)

    for model in MANUAL_MODELS:
        dest_path = MODELS_DIR / model["name"]
        print(f"\n[{model['name']}]")

        if dest_path.exists():
            print(f"  文件已存在，验证中...")
            if verify_sha256(dest_path, model["sha256"]):
                print(f"  ✓ 校验通过")
                skip_count += 1
            else:
                print(f"  ✗ 校验失败")
                print(f"  {model['instruction']}")
        else:
            print("  ✗ 文件不存在")
            print(f"  {model['instruction']}")
            print(f"  下载后放入: {dest_path}")

    # 总结
    print()
    print("=" * 70)
    print(f"下载完成: {success_count} 成功, {skip_count} 跳过, {fail_count} 失败")

    all_models = MODELS + MANUAL_MODELS
    total_required = len(all_models)
    existing_valid = sum(1 for m in all_models
                        if (MODELS_DIR / m["name"]).exists()
                        and verify_sha256(MODELS_DIR / m["name"], m["sha256"]))
    missing_runtime = [m["name"] for m in MANUAL_MODELS if m["required"]
                       and not ((MODELS_DIR / m["name"]).exists()
                                and verify_sha256(MODELS_DIR / m["name"], m["sha256"]))]

    print(f"模型就绪: {existing_valid}/{total_required}")

    if not missing_runtime and fail_count == 0:
        print()
        print("✓ 所有运行时模型已就绪，可以开始使用")
        if existing_valid < total_required:
            print("  (Face Mesh 是身份注意力增强与校准使用的可选模型；缺失时自动退回五点机制)")
    else:
        print()
        print("✗ 运行时模型未就绪，GUI 不能启动。")

    print("=" * 70)

    return 0 if not missing_runtime and fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
