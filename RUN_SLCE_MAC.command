#!/bin/bash

cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
  echo "First-time setup is required."
  echo "Running SETUP_MAC.command..."
  bash "./SETUP_MAC.command" || exit 1
fi

source ".venv/bin/activate"

if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "Streamlit not found in this environment."
  echo "Running setup..."
  bash "./SETUP_MAC.command" || exit 1
  source ".venv/bin/activate"
fi

run_on_port() {
  local port="$1"
  echo
  echo "Open this URL in Safari/Chrome:"
  echo "http://127.0.0.1:${port}"
  echo
  (
    sleep 3
    open "http://127.0.0.1:${port}"
  ) &
  python -m streamlit run app.py \
    --server.address 127.0.0.1 \
    --server.port "${port}" \
    --server.headless true \
    --browser.serverAddress 127.0.0.1 \
    --browser.serverPort "${port}" \
    --browser.gatherUsageStats false
}

echo "[SLCE] Launching app..."
echo "Keep this terminal window open while presenting."

if ! run_on_port 8501; then
  echo
  echo "Port 8501 failed. Retrying on 8502..."
  run_on_port 8502 || {
    echo
    echo "SLCE could not start. Please share this terminal output."
    read -r -p "Press Enter to close..."
    exit 1
  }
fi

echo
echo "[SLCE] App stopped."
read -r -p "Press Enter to close..."
