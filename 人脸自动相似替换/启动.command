#!/bin/zsh
cd "${0:A:h}"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "未找到项目虚拟环境 .venv。请先在终端运行："
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  read "?按回车关闭窗口…"
  exit 1
fi
"$PY" app.py
status=$?
if [ $status -ne 0 ]; then
  echo ""
  echo "启动失败，退出代码：$status"
  echo "如提示缺少依赖，请运行：.venv/bin/pip install -r requirements.txt"
  read "?按回车关闭窗口…"
fi
exit $status
