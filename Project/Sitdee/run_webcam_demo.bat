@echo off
cd /d "%~dp0"
echo Starting Office Posture Live Webcam Prototype V1...
echo If the camera is not index 0, run check_camera.bat first.
"%~dp0.venv\Scripts\python.exe" scripts\run_webcam_demo.py --source 0 --validation
echo.
echo The session files are under outputs\live\
pause
