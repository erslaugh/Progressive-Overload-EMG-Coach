@echo off
setlocal
cd /d "%~dp0"

echo.
echo EMG + IMU Workout Dashboard
echo ===========================
echo Installing/checking Python packages...
echo.

py -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo Installation failed.
    echo Make sure Python is installed and the "py" command works.
    pause
    exit /b 1
)

echo.
echo Starting dashboard...
echo.
py "%~dp0workout_dashboard.py"

if errorlevel 1 (
    echo.
    echo The dashboard closed with an error.
    pause
)
endlocal
