@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "GRADIO_ANALYTICS_ENABLED=False"
set "TRAIN_PYTHON=%~dp0..\training-preparation\.venv\Scripts\python.exe"
if not exist "%TRAIN_PYTHON%" (
  echo 未找到共用训练环境：%TRAIN_PYTHON%
  echo 请按 docs\webui-zh.md 配置环境后运行 python webui.py。
  pause
  exit /b 1
)
"%TRAIN_PYTHON%" -X utf8 webui.py --open
if errorlevel 1 pause
