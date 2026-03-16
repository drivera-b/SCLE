@echo off
setlocal
cd /d "%~dp0"
title SLCE Launcher

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup is required.
  echo Running SETUP_WINDOWS.bat...
  call "%~dp0SETUP_WINDOWS.bat"
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Could not activate the project environment.
  echo Run SETUP_WINDOWS.bat and try again.
  pause
  exit /b 1
)

python -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo Streamlit is not installed in this environment.
  echo Running dependency setup...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
  )
)

echo.
echo [SLCE] Starting app at http://localhost:8501
echo Keep this window open while presenting.
echo.
python -m streamlit run app.py
