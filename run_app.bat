@echo off
echo ========================================
echo   UPRYT - Posture Analysis System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import cv2, numpy, torch, mediapipe, PIL" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Dependencies may not be installed
    echo Running pip install...
    pip install -r requirements.txt
)

REM Launch GUI application
echo Starting UPRYT GUI...
python "%~dp0gui_app.py"

pause
