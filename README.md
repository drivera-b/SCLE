# SLCE (STEM Expo Computer Science Entry)

SLCE is a Python + Streamlit app that demonstrates **health/lifestyle risk planning under uncertainty** using a two-layer approach:

1. **Layer 1 (Baseline ML):** logistic regression trained on a real public heart disease dataset (UCI Heart Disease when available).
2. **Layer 2 (Stochastic Simulation):** a Monte Carlo model simulates many possible health trajectories over time and evaluates habit plans under constraints.

## Safety
**Educational tool only. Not medical advice or diagnosis.**

## Features
- Streamlit app with 3 pages:
  - `Dashboard`
  - `Optimize Plan`
  - `Weekly Log`
- Monte Carlo fan charts (risk + latent health state)
- Optimization of lifestyle plans under time/ramp constraints
- Adaptive personalization weights from weekly logs
- Robust validation + graceful fallbacks
- Demo profiles for reliable expo presentation

## Project Structure
- `app.py` - Streamlit UI
- `src/` - core modules
- `data/` - dataset cache and demo fallback sample
- `models/` - saved baseline model + metadata
- `tests/` - unit tests
- `STEM_EXPO_DOCS/` - expo support docs

## Setup
```bash
pip install -r requirements.txt
```

## Windows (No Terminal Needed)
If you cannot use terminal on the Windows laptop:

1. Download this repo as ZIP from GitHub and extract it.
2. Double-click `SETUP_WINDOWS.bat` once.
3. After setup finishes, double-click `RUN_SLCE_WINDOWS.bat` to launch the app.

Notes:
- Keep the launcher window open while presenting.
- The app opens at `http://127.0.0.1:8501` (or `8502` if 8501 is busy).
- Wait for the launcher to print the `Local URL` line before opening the browser.

## Run the app
```bash
streamlit run app.py
```

## NiceGUI Dashboard POC (optional)
This repo includes a side-by-side UI experiment using NiceGUI so you can compare app feel.

Run:
```bash
python3 nicegui_poc.py
```

Then open the local URL shown in terminal (typically `http://localhost:8080`).

## Reflex Dashboard POC (optional)
This repo also includes a side-by-side Reflex prototype without changing the Streamlit app.

Run:
```bash
reflex run
```

If port 3000/8000 is busy:
```bash
reflex run --frontend-port 3001 --backend-port 8001
```

## Train/retrain the baseline ML model
```bash
python -m src.baseline_model --train
```

Notes:
- The code attempts to download the UCI Heart Disease dataset automatically.
- If download fails, place a compatible CSV at `data/heart.csv`.
- The app can still run using `data/demo_sample.csv` fallback.

## Demo Mode (Reliable Expo Use)
Use the built-in demo profiles:
- `Balanced Student`
- `High Stress Student`
- `Inconsistent Sleeper`

Suggested live demo flow:
1. Open `Dashboard`, load `High Stress Student`, run simulation.
2. Open `Optimize Plan`, find best plans under a 30-45 min/day budget.
3. Open `Weekly Log`, enter an improved week and show weight adaptation.

## What judges should notice
- Real dataset baseline (Layer 1)
- Uncertainty bands, not a single prediction (Layer 2 Monte Carlo)
- Constraint-aware optimization
- Adaptive personalization from user logs
- Error handling and validation for live reliability

## Testing
Run unit tests:
```bash
pytest -q
```

## Screenshots (placeholders)
- `docs/screenshots/dashboard.png` (add later)
- `docs/screenshots/optimizer.png` (add later)
- `docs/screenshots/weekly_log.png` (add later)
