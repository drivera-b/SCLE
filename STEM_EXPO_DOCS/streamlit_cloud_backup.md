# Streamlit Cloud Backup Plan (For Restricted School Computers)

Use this when local installs are blocked by school security policy.

## Goal
Run SLCE from a public/private web URL so presenters only need a browser.

## Steps

1. Ensure the latest code is pushed to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Click **New app**.
4. Select repository: `drivera-b/SCLE`.
5. Set branch: `main`.
6. Set app file path: `app.py`.
7. Deploy.
8. After deployment, append `?demo=research` to the app URL and bookmark that version.

## Notes

- First deploy may take a few minutes.
- If deployment fails, check package versions from `requirements.txt`.
- Share the app URL with teammates before presentation day.
- The repository already includes `.streamlit/config.toml`, bundled datasets, and a trained model artifact; no secrets are required.

## Presentation-Day Use

1. Open the bookmarked URL ending in `?demo=research`.
2. Confirm the Dashboard already shows the measured adult profile and simulation results.
3. Run Optimize Plan once and confirm three plan cards appear.
4. Keep `reports/screenshots/simulation_charts.png` available in case internet is unstable.

## Required Preflight

1. Deploy at least one day before the presentation.
2. Test the exact shared URL in a private/incognito browser window.
3. Confirm the app wakes from sleep and loads within two minutes.
4. Send the tested URL to every presenter and the instructor.
