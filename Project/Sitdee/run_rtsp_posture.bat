@echo off
setlocal
cd /d "%~dp0"
if not exist ".env" if not defined POSTURE_RTSP_URL if not defined POSTURE_CAMERA_HOST (
  echo RTSP configuration is missing.
  echo Create a local .env from .env.example or set environment variables.
  exit /b 2
)
if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment was not found.
  exit /b 2
)
".venv\Scripts\python.exe" scripts\run_rtsp_posture.py --source rtsp --profile HIKVISION_MAIN --preset RTSP_DESKTOP
exit /b %ERRORLEVEL%
