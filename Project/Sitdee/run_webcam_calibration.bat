@echo off
cd /d "%~dp0"
echo Starting Office Posture calibration view...
echo This view shows technical geometry values. Press Q or ESC to exit.
"%~dp0.venv\Scripts\python.exe" scripts\run_webcam_demo.py --source 0 --validation --calibration
echo.
echo The session files are under outputs\live\
pause
