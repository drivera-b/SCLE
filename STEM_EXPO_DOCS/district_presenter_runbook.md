# SLCE District Presenter Runbook

This guide is for teammates presenting SLCE when the original developer is not present.

## 1) Best Reliability Strategy
Use this order of preference:

1. Personal laptop (recommended) using PyCharm and `RUN_SLCE_PYCHARM.py`.
2. Windows local launch using `SETUP_WINDOWS.bat` and `RUN_SLCE_WINDOWS.bat`.
3. Streamlit Community Cloud backup URL (no local install).

School-managed computers may block local servers (`127.0.0.1`) by policy.

## 2) One-Day-Before Checklist

1. Test on the exact laptop that will be used for presentation.
2. Confirm app opens and all 3 pages work:
   - Dashboard
   - Optimize Plan
   - Weekly Log
3. Confirm optimization returns 3 distinct plans.
4. Run one export for logbook artifacts.
5. Keep a backup ZIP of the repo on a USB drive.

## 3) Teammate Quick Start (No Terminal, PyCharm)

1. Download the latest repo ZIP from GitHub and extract it.
2. Open folder in PyCharm.
3. Set interpreter to Python 3.10 or 3.11.
4. Install packages from `requirements.txt` in PyCharm Python Packages.
5. Run `RUN_SLCE_PYCHARM.py` using the green Run button.
6. Open the shown URL (`127.0.0.1` on port 8501/8502/8503).

## 4) Windows Batch Quick Start (No Terminal)

1. Double-click `SETUP_WINDOWS.bat`.
2. After setup completes, double-click `RUN_SLCE_WINDOWS.bat`.
3. Wait for the `Local URL` line.
4. Open `http://127.0.0.1:8501` or `http://127.0.0.1:8502`.

## 5) Fast Troubleshooting

### Browser says "can't reach this page" on 127.0.0.1
- Keep launcher/PyCharm run process open.
- Wait for `Local URL` message.
- Disable Proxy and VPN on Windows.
- Allow Python through firewall if prompted.

### Install fails in PyCharm
- Use Python 3.10 or 3.11 interpreter.
- Install only `requirements.txt` first.
- Optional POC UIs use `requirements_optional_ui.txt`.

### School laptop still blocks localhost
- Use Streamlit Cloud backup deployment and present from a web URL.

## 6) Demo Flow (3 Minutes)

1. Dashboard:
   - Choose `High Stress Student`
   - Run simulation
   - Explain uncertainty fan chart and summary cards
2. Optimize Plan:
   - Set realistic constraints
   - Run optimizer
   - Compare top 3 plan cards
3. Weekly Log:
   - Enter improved week values
   - Show before/after personalization weights

## 7) Safety Statement

Always state:

**Educational tool only. Not medical advice or diagnosis.**
