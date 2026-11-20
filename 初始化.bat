@echo off
rem AI人像完整技能包 · 一键初始化（Windows 双击入口）
rem 实际逻辑在 setup.py（跨平台同一份），这里只负责调用并停留显示结果。
cd /d "%~dp0"
python setup.py
set "status=%errorlevel%"
echo.
pause

exit /b %status%
