@echo off
rem sotto for Windows (developer shortcut): first run creates .\venv and installs
rem the pinned Windows alpha set (+ CUDA extras when nvidia-smi is present).
rem Testers: follow docs\windows-alpha.md instead.
cd /d "%~dp0.."
if not exist venv\Scripts\python.exe (
    echo Creating venv and installing requirements...
    python -m venv venv || exit /b 1
    venv\Scripts\python -m pip install pip==26.2.1 || exit /b 1
    where nvidia-smi >nul 2>nul && (
        venv\Scripts\python -m pip install -r requirements-alpha-windows-cuda.txt || exit /b 1
    ) || (
        venv\Scripts\python -m pip install -r requirements-alpha-windows.txt || exit /b 1
    )
)
venv\Scripts\python sotto_win.py %*
