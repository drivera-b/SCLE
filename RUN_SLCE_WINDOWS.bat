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
echo [SLCE] Launching app...
echo Keep this window open while presenting.
echo.

call :run_streamlit 8501
if errorlevel 1 (
  echo.
  echo Port 8501 failed. Retrying on port 8502...
  call :run_streamlit 8502
  if errorlevel 1 (
    echo.
    echo SLCE could not start. Please keep this window and share this error output.
    pause
    exit /b 1
  )
)
exit /b 0

:run_streamlit
set "SLCE_PORT=%~1"
echo Opening http://127.0.0.1:%SLCE_PORT%
start "" "http://127.0.0.1:%SLCE_PORT%"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port %SLCE_PORT% --browser.gatherUsageStats false
if errorlevel 1 exit /b 1
exit /b 0
