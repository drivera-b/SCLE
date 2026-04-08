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

## Notes

- First deploy may take a few minutes.
- If deployment fails, check package versions from `requirements.txt`.
- Share the app URL with teammates before presentation day.

## Presentation-Day Use

1. Open the app URL in browser.
2. Use demo profiles for consistent live results.
3. Keep a backup screenshot set in case internet is unstable.
