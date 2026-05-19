# SLCE: Stochastic Lifestyle Control Engine

SLCE is a decision-support app that combines supervised learning, stochastic simulation, and constrained optimization to estimate long-term health risk under uncertainty.

This repository is intentionally packaged for reliable first-run behavior on new machines and live demos.

## Quant/Engineering Highlights
- Two-layer modeling system with baseline ML plus stochastic state simulation
- Baseline risk model from real public data (`LogisticRegression` on UCI Cleveland heart dataset features)
- Weekly stochastic state model (`H_t`) with Monte Carlo uncertainty propagation
- Constrained plan optimization with explicit objective and feasibility rules
- Adaptive personalization from weekly logs (weight updates based on observed trend direction)
- Defensive runtime design with input validation, clamping, and fallback logic
- Graceful fallback to local demo data/model paths
- Cached model/dataset status and friendly error messaging
- Verified test suite (`pytest`) for core validation and stochastic logic

## Modeling Overview
- Latent health state update (weekly):
`H_{t+1} = clamp(H_t + drift(features, plan) + noise, 0, 100)`
- Weekly risk mapping:
`p_t = sigmoid(alpha + beta * (100 - H_t)/100 + baseline_logit)`
- Optimizer objective:
`objective = expected_mean_risk + lambda_time * time_cost - lambda_adherence * adherence_score`

## Data and Artifacts
- Real dataset included: `/data/heart.csv` (UCI Cleveland format, cleaned, 303 rows)
- Fallback dataset: `/data/demo_sample.csv`
- Pretrained baseline artifacts: `/models/baseline_model.joblib`, `/models/baseline_metadata.json`

Current baseline holdout metrics (from metadata):
- Accuracy: `0.8684`
- ROC-AUC: `0.9310`
- Confusion matrix: `[[35, 6], [4, 31]]`

## Quick Start (No Terminal Required)

### macOS
1. Download ZIP from GitHub and extract.
2. Double-click `/SETUP_MAC.command` (first time only).
3. Double-click `/RUN_SLCE_MAC.command`.

### Windows
1. Download ZIP from GitHub and extract.
2. Double-click `/SETUP_WINDOWS.bat` (first time only).
3. Double-click `/RUN_SLCE_WINDOWS.bat`.

### PyCharm (Mac/Windows)
1. Open project folder in PyCharm.
2. Use Python `3.10` or `3.11` interpreter.
3. Install from `/requirements.txt`.
4. Run `/RUN_SLCE_PYCHARM.py`.

If school-managed laptops block localhost (`127.0.0.1`), use the Streamlit Cloud backup flow in `/STEM_EXPO_DOCS/streamlit_cloud_backup.md`.

## CLI Setup (Developer Path)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproducibility Commands
Train/retrain baseline model:
```bash
python -m src.baseline_model --train
```

Run tests:
```bash
pytest -q
```

## Repository Layout
- `/app.py` Streamlit application (Dashboard, Optimize Plan, Weekly Log)
- `/src/` core modeling modules
- `/data/` bundled real and fallback datasets
- `/models/` model artifact and metadata
- `/tests/` unit tests
- `/STEM_EXPO_DOCS/` runbook, testing checklist, logbook, tri-board layout, data provenance

## Reliability Guarantees in This Repo
- Works without internet using bundled `/data/heart.csv` and `/models/baseline_model.joblib`
- Auto-fallback to `/data/demo_sample.csv` if needed
- Heuristic baseline fallback if model artifact is unavailable
- Explicit validation and friendly user-facing failures instead of raw stack traces

## Safety
Educational tool only. Not medical advice or diagnosis.

## Dataset Provenance
See `/STEM_EXPO_DOCS/dataset_provenance.md` for source, DOI, and licensing details.
