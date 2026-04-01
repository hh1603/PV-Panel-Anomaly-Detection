@echo off
setlocal

python -m venv .venv || exit /b 1
.\.venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1

echo Virtual environment ready: .venv
echo Use .\.venv\Scripts\python.exe to run project scripts.
