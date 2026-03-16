@echo off
setlocal
cd /d "%~dp0"
title SLCE Setup

echo.
echo [SLCE] Setting up this project for Windows...
echo.

call :resolve_python
if errorlevel 1 goto :fail

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto :fail
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :fail

echo Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist "models\baseline_model.joblib" (
  echo Baseline model missing. Training a fallback model...
  python -m src.baseline_model --train
)

echo.
echo [SLCE] Setup complete.
echo Next step: double-click RUN_SLCE_WINDOWS.bat
echo.
pause
exit /b 0

:resolve_python
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY_CMD=py"
  exit /b 0
)
where python >nul 2>nul
if %errorlevel%==0 (
  set "PY_CMD=python"
  exit /b 0
)
echo Python was not found on this PC.
echo Install Python 3.10+ and enable "Add Python to PATH", then run this file again.
pause
exit /b 1

:fail
echo.
echo [SLCE] Setup failed.
echo Please review the messages above and try again.
echo.
pause
exit /b 1
