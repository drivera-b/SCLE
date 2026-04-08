#!/bin/bash
set -e

cd "$(dirname "$0")"

echo
echo "[SLCE] Setting up this project for macOS..."
echo

PY_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PY_CMD="python"
else
  echo "Python was not found."
  echo "Install Python 3.10+ and run this file again."
  read -r -p "Press Enter to close..."
  exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  "$PY_CMD" -m venv .venv
fi

source ".venv/bin/activate"

echo "Installing dependencies..."
python -m pip install --upgrade pip
if ! python -m pip install -r requirements.txt; then
  echo "Core requirements install failed. Trying fallback package install..."
  python -m pip install streamlit numpy pandas scikit-learn matplotlib joblib
fi

if [ ! -f "models/baseline_model.joblib" ]; then
  echo "Baseline model missing. Training fallback model..."
  python -m src.baseline_model --train || true
fi

echo
echo "[SLCE] Setup complete."
echo "Next step: double-click RUN_SLCE_MAC.command"
echo
read -r -p "Press Enter to close..."
