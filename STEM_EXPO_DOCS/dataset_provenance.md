# SLCE Dataset Provenance

## Primary Baseline Dataset
- Name: UCI Heart Disease (Cleveland subset)
- DOI: `10.24432/C52P4X`
- License: CC BY 4.0
- Local file in repo: `data/heart.csv`

## Why this matters
SLCE uses this real public dataset for Layer 1 baseline logistic regression training.

## Reliability design
To keep demos stable on restricted networks:
- `data/heart.csv` is bundled in the repo.
- `data/demo_sample.csv` is kept as fallback.
- If model artifacts are missing, SLCE can retrain from local data.
