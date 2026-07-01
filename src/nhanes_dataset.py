from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .dataset import data_dir


NHANES_CYCLE = "2017-2018"
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
NHANES_FILES = {
    "DEMO_J": f"{NHANES_BASE_URL}/DEMO_J.XPT",
    "SLQ_J": f"{NHANES_BASE_URL}/SLQ_J.XPT",
    "PAQ_J": f"{NHANES_BASE_URL}/PAQ_J.XPT",
    "PAQY_J": f"{NHANES_BASE_URL}/PAQY_J.XPT",
    "BPX_J": f"{NHANES_BASE_URL}/BPX_J.XPT",
    "BMX_J": f"{NHANES_BASE_URL}/BMX_J.XPT",
    "TCHOL_J": f"{NHANES_BASE_URL}/TCHOL_J.XPT",
    "HDL_J": f"{NHANES_BASE_URL}/HDL_J.XPT",
    "GHB_J": f"{NHANES_BASE_URL}/GHB_J.XPT",
    "GLU_J": f"{NHANES_BASE_URL}/GLU_J.XPT",
}


def default_output_path() -> Path:
    return data_dir() / "nhanes_lifestyle_biomarkers.csv"


def default_metadata_path() -> Path:
    return data_dir() / "nhanes_lifestyle_biomarkers_metadata.json"


def default_cache_dir() -> Path:
    return data_dir() / "nhanes_raw"


def _download(url: str, destination: Path, timeout: int = 45) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response:
        destination.write_bytes(response.read())


def download_nhanes_modules(cache_dir: Path | None = None) -> dict[str, Path]:
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    errors: list[str] = []
    for name, url in NHANES_FILES.items():
        destination = cache / f"{name}.XPT"
        paths[name] = destination
        if destination.exists() and destination.stat().st_size > 0:
            continue
        try:
            _download(url, destination)
        except (URLError, TimeoutError, OSError) as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("Could not download all NHANES modules: " + "; ".join(errors))
    return paths


def _read_modules(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing NHANES modules: {missing}")
    return {name: pd.read_sas(path, format="xport") for name, path in paths.items()}


def _valid_numeric(series: pd.Series, low: float, high: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values.between(low, high))


def _activity_component(
    frame: pd.DataFrame,
    *,
    yes_column: str,
    days_column: str,
    minutes_column: str,
) -> tuple[pd.Series, pd.Series]:
    yes = pd.to_numeric(frame[yes_column], errors="coerce")
    days = _valid_numeric(frame[days_column], 1, 7)
    minutes = _valid_numeric(frame[minutes_column], 1, 1440)
    active_days = days.where(yes.eq(1), 0.0).where(yes.isin([1, 2]))
    weekly_minutes = (days * minutes).where(yes.eq(1), 0.0).where(yes.isin([1, 2]))
    return active_days, weekly_minutes


def build_nhanes_lifestyle_dataset(
    *,
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    download: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = cache_dir or default_cache_dir()
    paths = {name: cache / f"{name}.XPT" for name in NHANES_FILES}
    if download:
        paths = download_nhanes_modules(cache)
    modules = _read_modules(paths)

    demo = modules["DEMO_J"][["SEQN", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]].copy()
    sleep = modules["SLQ_J"][["SEQN", "SLD012", "SLD013"]].copy()
    adult_activity = modules["PAQ_J"][["SEQN", "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675", "PAD680"]].copy()
    youth_activity = modules["PAQY_J"][["SEQN", "PAQ706"]].copy()
    blood_pressure = modules["BPX_J"][[
        "SEQN", "BPXPLS", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"
    ]].copy()
    body = modules["BMX_J"][["SEQN", "BMXBMI"]].copy()
    cholesterol = modules["TCHOL_J"][["SEQN", "LBXTC"]].copy()
    hdl = modules["HDL_J"][["SEQN", "LBDHDD"]].copy()
    hba1c = modules["GHB_J"][["SEQN", "LBXGH"]].copy()
    glucose = modules["GLU_J"][["SEQN", "LBXGLU"]].copy()

    frame = demo
    for module in (sleep, adult_activity, youth_activity, blood_pressure, body, cholesterol, hdl, hba1c, glucose):
        frame = frame.merge(module, on="SEQN", how="left")

    weekday_sleep = _valid_numeric(frame["SLD012"], 3, 14)
    weekend_sleep = _valid_numeric(frame["SLD013"], 3, 14)
    frame["sleep_mean_hours"] = (5.0 * weekday_sleep + 2.0 * weekend_sleep) / 7.0
    frame["sleep_mean_hours"] = frame["sleep_mean_hours"].fillna(weekday_sleep).fillna(weekend_sleep)
    frame["sleep_variability_hours"] = (weekend_sleep - weekday_sleep).abs()

    vigorous_days, vigorous_minutes = _activity_component(
        frame, yes_column="PAQ650", days_column="PAQ655", minutes_column="PAD660"
    )
    moderate_days, moderate_minutes = _activity_component(
        frame, yes_column="PAQ665", days_column="PAQ670", minutes_column="PAD675"
    )
    adult_days = pd.concat([vigorous_days, moderate_days], axis=1).max(axis=1, skipna=True)
    adult_weekly_minutes = vigorous_minutes.add(moderate_minutes, fill_value=0.0)
    youth_days = _valid_numeric(frame["PAQ706"], 0, 7)
    is_youth = pd.to_numeric(frame["RIDAGEYR"], errors="coerce").lt(18)
    frame["exercise_days_per_week"] = adult_days.where(~is_youth, youth_days)
    frame["exercise_minutes_per_week"] = adult_weekly_minutes.where(~is_youth, youth_days * 60.0)

    systolic_columns = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]
    diastolic_columns = ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]
    frame["systolic_bp"] = frame[systolic_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    frame["diastolic_bp"] = frame[diastolic_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    output = pd.DataFrame(
        {
            "participant_id": pd.to_numeric(frame["SEQN"], errors="coerce").astype("Int64"),
            "age": pd.to_numeric(frame["RIDAGEYR"], errors="coerce"),
            "sex": frame["RIAGENDR"].map({1.0: "Male", 2.0: "Female"}),
            "survey_weight": pd.to_numeric(frame["WTMEC2YR"], errors="coerce"),
            "survey_psu": pd.to_numeric(frame["SDMVPSU"], errors="coerce"),
            "survey_stratum": pd.to_numeric(frame["SDMVSTRA"], errors="coerce"),
            "sleep_mean_hours": frame["sleep_mean_hours"],
            "sleep_variability_hours": frame["sleep_variability_hours"],
            "exercise_days_per_week": frame["exercise_days_per_week"],
            "exercise_minutes_per_week": frame["exercise_minutes_per_week"],
            "sedentary_minutes_per_day": _valid_numeric(frame["PAD680"], 0, 1440),
            "resting_hr": _valid_numeric(frame["BPXPLS"], 30, 220),
            "systolic_bp": _valid_numeric(frame["systolic_bp"], 60, 260),
            "diastolic_bp": _valid_numeric(frame["diastolic_bp"], 30, 160),
            "bmi": _valid_numeric(frame["BMXBMI"], 8, 90),
            "total_cholesterol": _valid_numeric(frame["LBXTC"], 50, 700),
            "hdl_cholesterol": _valid_numeric(frame["LBDHDD"], 5, 200),
            "hba1c": _valid_numeric(frame["LBXGH"], 2, 25),
            "fasting_glucose": _valid_numeric(frame["LBXGLU"], 30, 600),
        }
    )
    output = output.loc[output["age"].between(16, 80)].reset_index(drop=True)
    research_fields = [
        "sleep_mean_hours", "exercise_days_per_week", "resting_hr", "systolic_bp", "bmi",
        "total_cholesterol", "hdl_cholesterol", "hba1c", "fasting_glucose",
    ]
    output["available_measurement_count"] = output[research_fields].notna().sum(axis=1)
    numeric_columns = output.select_dtypes(include=[np.number]).columns
    output[numeric_columns] = output[numeric_columns].round(3)

    destination = output_path or default_output_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    metadata = {
        "dataset": "NHANES 2017-2018 lifestyle and biomarker research extract",
        "source": "CDC/NCHS NHANES",
        "cycle": NHANES_CYCLE,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(output)),
        "columns": list(output.columns),
        "source_files": NHANES_FILES,
        "merge_key": "SEQN (exported as participant_id)",
        "minimum_age": 16,
        "notes": [
            "Survey weights, PSU, and strata are retained for research analyses.",
            "Adult exercise days use the maximum of reported vigorous/moderate recreation days to avoid double-counting unknown overlap.",
            "Youth exercise minutes approximate each active day as 60 minutes.",
            "This extract is for research and population-percentile context, not diagnosis.",
        ],
    }
    default_metadata_path().write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the SLCE NHANES lifestyle/biomarker research extract.")
    parser.add_argument("--build", action="store_true", help="Download, merge, clean, and save the extract.")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--no-download", action="store_true", help="Use already-downloaded XPT files.")
    args = parser.parse_args(argv)
    if not args.build:
        parser.print_help()
        return 0
    try:
        frame, metadata = build_nhanes_lifestyle_dataset(
            cache_dir=args.cache_dir,
            output_path=args.output,
            download=not args.no_download,
        )
    except Exception as exc:
        print(f"NHANES build failed: {exc}")
        return 1
    print(f"Built {len(frame)} rows at {args.output}")
    print(f"Available columns: {', '.join(metadata['columns'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
