@echo off
setlocal
cd /d "%~dp0"
title Office Posture - Acer Lightweight Development

if not exist ".venv\Scripts\python.exe" (
    echo .venv was not found. Starting the one-time lightweight setup...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_laptop_dev.ps1"
    if errorlevel 1 (
        echo Setup did not finish. Read the message above and fix the reported item.
        pause
        exit /b 1
    )
)

echo.
echo Running quick health check. This does not open the camera or download data.
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\laptop_health_check.py"
echo.
if exist ".git" (
    git status --short --branch
) else (
    echo Git repository: not configured in this checkout yet.
)
echo.
echo Environment is ready. Type commands here, or run scripts\run_laptop_tests.py.
call "%~dp0.venv\Scripts\activate.bat"
cmd /k
