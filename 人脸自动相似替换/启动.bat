@echo off
rem 人脸自动相似替换 · Windows 启动入口（macOS 用 启动.command）
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 未找到虚拟环境 .venv，请先双击根目录的「初始化.bat」完成初始化。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
  echo.
  echo 启动失败。如提示缺少依赖，请重新运行根目录的「初始化.bat」。
  pause
)
