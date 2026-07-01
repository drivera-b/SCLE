from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import data_dir


BIOMARKER_LABELS = {
    "systolic_bp": "Systolic blood pressure",
    "total_cholesterol": "Total cholesterol",
    "fasting_glucose": "Fasting glucose",
    "hba1c": "HbA1c",
    "bmi": "BMI",
}


def _is_present(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def input_evidence_summary(
    profile: dict[str, Any],
    *,
    weekly_log_count: int = 0,
    training_age_range: tuple[float, float] = (29.0, 77.0),
) -> dict[str, Any]:
    """Score input coverage, not predictive or clinical certainty."""
    score = 15.0
    strengths: list[str] = []
    limitations: list[str] = []

    age = float(profile.get("age", 18))
    if training_age_range[0] <= age <= training_age_range[1]:
        score += 15.0
        strengths.append("Age is within the UCI training range.")
    else:
        limitations.append(
            f"Age is outside the UCI training range ({training_age_range[0]:.0f}-{training_age_range[1]:.0f})."
        )

    if str(profile.get("sex", "Unknown")) not in {"Unknown", "Other", ""}:
        score += 5.0
        strengths.append("Sex input is available for the baseline model.")
    else:
        limitations.append("Sex is unknown and uses a simplified fallback mapping.")

    if _is_present(profile.get("resting_hr")):
        score += 5.0
        strengths.append("Resting heart rate is measured or user-provided.")

    direct_weights = {
        "systolic_bp": 10.0,
        "total_cholesterol": 10.0,
        "fasting_glucose": 10.0,
    }
    measured_fields: list[str] = []
    for field, weight in direct_weights.items():
        if _is_present(profile.get(field)) and bool(profile.get("use_biomarkers", False)):
            score += weight
            measured_fields.append(field)
    if measured_fields:
        strengths.append(
            "Measured clinical inputs replace model medians: "
            + ", ".join(BIOMARKER_LABELS[field] for field in measured_fields)
            + "."
        )
    else:
        limitations.append("Blood pressure, cholesterol, and glucose use proxy/imputed model values.")

    supplemental_fields = [field for field in ("hba1c", "bmi") if _is_present(profile.get(field))]
    if bool(profile.get("use_biomarkers", False)) and supplemental_fields:
        score += min(10.0, 5.0 * len(supplemental_fields))
        strengths.append("Supplemental biomarkers improve population-reference context.")

    history_points = min(10.0, max(0, int(weekly_log_count)) * 3.0)
    score += history_points
    if weekly_log_count:
        strengths.append(f"Personalization includes {weekly_log_count} weekly log(s).")
    else:
        limitations.append("No longitudinal weekly history is available yet.")

    score = float(np.clip(score, 0.0, 100.0))
    level = "High" if score >= 80 else "Moderate" if score >= 55 else "Limited"
    return {
        "score": score,
        "level": level,
        "strengths": strengths,
        "limitations": limitations,
        "measured_fields": measured_fields,
        "label": "Input Evidence Score",
        "disclaimer": "Measures data coverage and model applicability, not medical certainty.",
    }


def default_reference_path() -> Path:
    return data_dir() / "nhanes_lifestyle_biomarkers.csv"


@lru_cache(maxsize=4)
def _load_reference_cached(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


def population_percentile_context(
    profile: dict[str, Any],
    *,
    reference_path: Path | None = None,
) -> list[dict[str, Any]]:
    path = reference_path or default_reference_path()
    if not path.exists() or not bool(profile.get("use_biomarkers", False)):
        return []

    frame = _load_reference_cached(str(path.resolve()))
    age = float(profile.get("age", 18))
    comparison = frame.loc[frame["age"].between(max(16.0, age - 10.0), min(80.0, age + 10.0))].copy()
    sex = str(profile.get("sex", "Unknown"))
    if sex in {"Male", "Female"}:
        same_sex = comparison.loc[comparison["sex"] == sex]
        if len(same_sex) >= 50:
            comparison = same_sex
    if len(comparison) < 50:
        comparison = frame

    rows: list[dict[str, Any]] = []
    for field, label in BIOMARKER_LABELS.items():
        if not _is_present(profile.get(field)) or field not in comparison.columns:
            continue
        metric_frame = pd.DataFrame(
            {
                "value": pd.to_numeric(comparison[field], errors="coerce"),
                "weight": pd.to_numeric(comparison.get("survey_weight"), errors="coerce")
                if "survey_weight" in comparison
                else np.nan,
            }
        ).dropna(subset=["value"])
        if len(metric_frame) < 30:
            continue
        value = float(profile[field])
        valid_weights = metric_frame["weight"].notna() & metric_frame["weight"].gt(0)
        if valid_weights.sum() >= 30:
            weighted = metric_frame.loc[valid_weights]
            percentile = float(
                weighted.loc[weighted["value"] <= value, "weight"].sum() / weighted["weight"].sum() * 100.0
            )
            reference_method = "NHANES survey-weighted"
            sample_size = int(len(weighted))
        else:
            percentile = float((metric_frame["value"] <= value).mean() * 100.0)
            reference_method = "Unweighted fallback"
            sample_size = int(len(metric_frame))
        rows.append(
            {
                "Biomarker": label,
                "Your value": round(value, 2),
                "Population percentile": round(percentile, 1),
                "Reference sample": sample_size,
                "Method": reference_method,
            }
        )
    return rows
