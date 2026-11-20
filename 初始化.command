#!/bin/zsh
# AI人像完整技能包 · 一键初始化（macOS 双击入口）
# 实际逻辑在 setup.py（跨平台同一份），这里只负责调用并停留显示结果。
cd "$(dirname "$0")"
python3 setup.py
status=$?
echo ""
read "?按回车关闭窗口…"
exit $status
