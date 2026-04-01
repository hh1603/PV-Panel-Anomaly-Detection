@echo off
setlocal
cd /d "%~dp0.."
set ROOT=%CD%
set PYTHON=%ROOT%\.venv\Scripts\python.exe
if not exist "%PYTHON%" (
  echo Virtual environment not found: %PYTHON%
  echo Run scripts\setup_venv.cmd first.
  pause
  exit /b 1
)
echo.
echo Starting local demo server...
echo URL: http://127.0.0.1:7860
echo Keep this window open while you use the page.
echo Press Ctrl+C here to stop the server.
echo.
"%PYTHON%" "%ROOT%\scripts\web_demo.py" --host 127.0.0.1 --port 7860 --device cpu
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Demo exited with code %EXIT_CODE%.
  echo If the page still cannot open, copy the error in this window and send it over.
  pause
)
