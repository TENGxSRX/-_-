@echo off
cd /d "%~dp0"
echo Checking camera indices 0, 1, and 2...
"%~dp0.venv\Scripts\python.exe" scripts\list_cameras.py --indices 0 1 2
pause

