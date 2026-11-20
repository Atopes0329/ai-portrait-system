#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI人像完整技能包 · 一键初始化（跨平台）

用法:
  python3 setup.py          # macOS / Linux
  python setup.py           # Windows
  或双击 初始化.command（macOS）/ 初始化.bat（Windows）

自动完成:
  1. 校验目录关系（两个 skill 必须在同一父目录）
  2. 检查 Python 3.8+
  3. 生成 ai人像摄影技能包/config.json 空白模板（缺才生成）
  4. 建 人脸自动相似替换/.venv 并装依赖（缺才建）
  5. 校验模型（缺才用 download_models.py 补）
  6. 打印当前平台的正确命令
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
SK1 = os.path.join(ROOT, "ai人像摄影技能包")
SK2 = os.path.join(ROOT, "人脸自动相似替换")
IS_WIN = os.name == "nt"
REQUIRED_MODELS = [
    "face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx",
    "arcfaceresnet100-11-int8.onnx",
    "adaface_ir_18.onnx",
]


def invalid_runtime_models():
    """校验模型尺寸与 SHA256；不能把“文件存在”当作“模型可用”。"""
    if SK2 not in sys.path:
        sys.path.insert(0, SK2)
    from model_manifest import invalid_runtime_models as verify

    return verify(Path(SK2) / "models")


def verify_runtime_load(vpy):
    """确认当前虚拟环境真的能加载 OpenCV、ONNX Runtime 与全部运行模型。"""
    check = (
        "from face_guard import FaceEngine; "
        "from identity_ensemble import IdentityEnsemble; "
        "FaceEngine(); IdentityEnsemble(); print('runtime models loaded')"
    )
    subprocess.run([vpy, "-c", check], cwd=SK2, check=True)


def venv_python(skill_dir):
    """返回 .venv 里的 python 解释器路径（平台自适应）。"""
    if IS_WIN:
        return os.path.join(skill_dir, ".venv", "Scripts", "python.exe")
    return os.path.join(skill_dir, ".venv", "bin", "python")


def venv_pip(skill_dir):
    """返回 .venv 里的 pip 调用方式。Windows 用 python -m pip，类 unix 用 bin/pip。"""
    if IS_WIN:
        return [venv_python(skill_dir), "-m", "pip"]
    return [os.path.join(skill_dir, ".venv", "bin", "pip")]


def main():
    print("=" * 60)
    print("  AI人像完整技能包 · 一键初始化")
    print("=" * 60)
    print()

    # 1. 目录关系校验
    if not (os.path.isdir(SK1) and os.path.isdir(SK2)):
        print("✗ 目录关系不对。本系统要求：")
        print("  ai人像摄影技能包/  与  人脸自动相似替换/")
        print("  必须与 setup.py 在同一父目录下（保持本包原样即可）。")
        return 1
    print("✓ 目录关系正确（两个 skill 在同一父目录）")

    # 2. Python 检查
    py = sys.executable
    if sys.version_info < (3, 8):
        print(f"✗ 需要 Python 3.8+，当前 {sys.version.split()[0]}")
        return 1
    print(f"✓ Python: {sys.version.split()[0]}")

    # 3. config 模板
    conf = os.path.join(SK1, "config.json")
    if os.path.exists(conf):
        print("✓ ai人像摄影技能包/config.json 已存在")
    else:
        example = os.path.join(SK1, "config.example.json")
        shutil.copy(example, conf)
        print("→ 已生成 ai人像摄影技能包/config.json 空白模板")
        print("  请把「把你的key填在这里」换成你的 key / host / model，")
        print("  或让 AI 助手按「首次配置问卷.md」的问卷帮你填。")

    # 4. venv + 依赖
    vpy = venv_python(SK2)
    if os.path.exists(vpy):
        print("✓ 人脸自动相似替换/.venv 已存在")
    else:
        print("→ 创建虚拟环境并安装依赖（首次 1-3 分钟）…")
        try:
            subprocess.run([py, "-m", "venv", os.path.join(SK2, ".venv")], check=True)
            subprocess.run(venv_pip(SK2) + ["install", "-q", "--upgrade", "pip"], check=False)
            subprocess.run(venv_pip(SK2) + ["install", "-q", "-r",
                                            os.path.join(SK2, "requirements.txt")], check=True)
        except subprocess.CalledProcessError as e:
            print(f"✗ 依赖安装失败：{e}")
            print("  请检查网络后重试，或手动运行：")
            print("  " + " ".join(venv_pip(SK2) + ["install", "-r", "requirements.txt"]))
            return 1
        print("✓ 依赖安装完成")

    # 5. 模型校验
    invalid = invalid_runtime_models()
    if invalid:
        print(f"→ {len(invalid)} 个运行模型缺失或校验失败，尝试修复…")
        print("  " + "、".join(invalid))
        dl = os.path.join(SK2, "download_models.py")
        subprocess.run([py, dl], cwd=SK2, check=False)
        invalid = invalid_runtime_models()
    if invalid:
        print("✗ 以下运行模型未就绪，停止初始化，GUI 不会被错误标记为可用：")
        print("  " + "、".join(invalid))
        print("  可按 人脸自动相似替换/models/README.md 的链接手动下载，放入 models 后重试。")
        return 1
    print("✓ 4 个运行模型哈希校验通过")
    try:
        verify_runtime_load(vpy)
    except subprocess.CalledProcessError as e:
        print(f"✗ 运行时无法加载依赖或模型：{e}")
        print("  请重新运行初始化；若仍失败，请把完整报错交给 AI 助手。")
        return 1
    print("✓ OpenCV / ONNX Runtime 与模型加载成功")

    # 6. 平台命令
    print()
    print("=" * 60)
    if IS_WIN:
        print("  Windows 命令：")
        print("    启动 GUI  : 双击 人脸自动相似替换/启动.bat")
        print("    虚拟环境  : 人脸自动相似替换\\.venv\\Scripts\\python.exe")
        print("    手动出图  : 让 AI 读 ai人像摄影技能包/SKILL.md")
    else:
        print("  macOS / Linux 命令：")
        print("    启动 GUI  : 双击 人脸自动相似替换/启动.command（macOS）")
        print("    虚拟环境  : .venv/bin/python")
        print("    手动出图  : 让 AI 读 ai人像摄影技能包/SKILL.md")
    print("  下一步配置：编辑 ai人像摄影技能包/config.json 填你的 API（或让 AI 按问卷填）")
    print("=" * 60)
    print()
    print("初始化完成。详细使用看 README.md，系统架构看 SYSTEM.md。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
