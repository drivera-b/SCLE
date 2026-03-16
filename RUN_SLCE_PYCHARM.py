"""PyCharm-friendly launcher for SLCE Streamlit app.

Run this file with PyCharm's green Run button.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _pick_port() -> int:
    """Pick a localhost port that is likely available."""
    for port in (8501, 8502, 8503):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 8501


def _open_browser(url: str) -> None:
    try:
        webbrowser.open_new(url)
    except Exception:
        # Browser auto-open is optional; the URL is printed either way.
        pass


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    app_path = project_dir / "app.py"
    if not app_path.exists():
        print(f"Could not find app file: {app_path}")
        return 1

    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError:
        print("Streamlit is not installed in this interpreter.")
        print("In PyCharm: Python Packages -> install from requirements.txt")
        print("If needed, install Streamlit directly from Python Packages and rerun.")
        return 1

    os.chdir(project_dir)
    port = _pick_port()
    url = f"http://127.0.0.1:{port}"
    print(f"Starting SLCE at {url}")
    print("Keep this run process active while presenting.")

    # Delay browser open slightly so Streamlit has time to boot.
    threading.Timer(2.0, _open_browser, args=(url,)).start()

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.serverAddress",
        "127.0.0.1",
        "--browser.serverPort",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
