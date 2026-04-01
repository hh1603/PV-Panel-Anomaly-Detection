param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

& $PythonExe -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Virtual environment ready: .venv"
Write-Host "Use .\\.venv\\Scripts\\python.exe to run project scripts."
