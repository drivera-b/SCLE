# VITA STEM Expo Testing Checklist

## Pre-demo startup
- `pip install -r requirements.txt` completes with no errors.
- `streamlit run app.py` launches and opens the app locally.
- App loads without crashing even if `models/baseline_model.joblib` is missing.
- Safety disclaimer is visible: **Educational tool only. Not medical advice or diagnosis.**

## Dashboard manual test cases
1. Balanced demo profile (1-year horizon)
- Load `Balanced`.
- Click `Run Simulation`.
- Expected: metrics render, 3 plots render, no stack trace.

2. High-Stress Student demo profile (5-year horizon)
- Load `High-Stress Student`.
- Click `Run Simulation`.
- Expected: wider uncertainty band than balanced profile; results still render.

3. Validation error handling
- Enter invalid values (example: stress score 0 via manual edit if possible, or unsupported horizon by code edit test).
- Expected: user-friendly error messages, app remains usable.

4. Missing model fallback
- Temporarily rename `models/baseline_model.joblib` (if present).
- Run simulation.
- Expected: app shows fallback/auto-train note and still computes results.

## Optimize Plan manual test cases
1. Standard constraints
- Use `Balanced` or current Dashboard inputs.
- Set `Max minutes/day` to 45 and `Optimization Monte Carlo paths` to 300.
- Click `Find Best Plans`.
- Expected: top 3 plans table appears + tradeoff plot + ramp preview.

2. Overly strict constraints
- Set `Max minutes/day` to 0 and exercise cap to 0.
- Click `Find Best Plans`.
- Expected: either feasible minimal plans or a clear warning if no feasible plan exists.

3. Runtime stability
- Use 5-year horizon and 300 optimization paths.
- Expected: app completes within demo-friendly time (few seconds to under ~20s depending on laptop).

## Weekly Log manual test cases
1. Valid 7-day entry
- Enter 7 days of sleep/stress/exercise/nutrition.
- Click `Submit Weekly Log & Adapt Model`.
- Expected: previous vs new weights table and explanation shown.

2. Invalid entry length/data
- Delete a row or enter invalid numeric value if UI allows.
- Expected: friendly validation errors, no crash.

3. Repeat adaptation
- Submit a second week with noticeably different stress/sleep.
- Expected: log history table grows and weights adjust incrementally.

## Final demo readiness checks
- Laptop charger connected.
- Browser zoom set to 100%.
- Internet not required for demo mode (works with `data/demo_sample.csv`).
- Backup screenshot(s) ready in case of venue Wi-Fi/power issues.

