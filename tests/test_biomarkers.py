from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.baseline_model as baseline_model
from src.baseline_model import MODEL_FEATURES, _profile_to_model_row
from src.biomarkers import input_evidence_summary, population_percentile_context
from src.nhanes_dataset import default_output_path


def _profile(*, measured: bool) -> dict[str, object]:
    return {
        "age": 45,
        "sex": "Male",
        "resting_hr": 70,
        "stress_score": 5,
        "exercise_days_per_week": 3,
        "nutrition_score": 6,
        "use_biomarkers": measured,
        "systolic_bp": 132.0,
        "total_cholesterol": 205.0,
        "fasting_glucose": 125.0,
        "hba1c": 5.7,
        "bmi": 26.0,
    }


def test_measured_biomarkers_replace_proxy_model_fields():
    metadata = {"feature_medians": {feature: 1.0 for feature in MODEL_FEATURES}}
    row, provenance = _profile_to_model_row(_profile(measured=True), metadata)

    assert row.loc[0, "trestbps"] == 132.0
    assert row.loc[0, "chol"] == 205.0
    assert row.loc[0, "fbs"] == 1.0
    assert {"trestbps", "chol", "fbs"}.issubset(provenance["observed"])


def test_evidence_score_rewards_measured_inputs_and_age_coverage():
    measured = input_evidence_summary(_profile(measured=True), weekly_log_count=2)
    unmeasured_profile = _profile(measured=False)
    unmeasured_profile["age"] = 18
    unmeasured = input_evidence_summary(unmeasured_profile, weekly_log_count=0)

    assert measured["score"] > unmeasured["score"]
    assert measured["level"] == "High"
    assert unmeasured["level"] == "Limited"


def test_population_percentiles_use_age_and_sex_reference(tmp_path):
    values = np.arange(100, 200, dtype=float)
    reference = pd.DataFrame(
        {
            "age": [45.0] * len(values),
            "sex": ["Male"] * len(values),
            "systolic_bp": values,
            "total_cholesterol": values + 50.0,
            "fasting_glucose": values - 40.0,
            "hba1c": np.linspace(4.5, 7.0, len(values)),
            "bmi": np.linspace(18.0, 35.0, len(values)),
        }
    )
    path = tmp_path / "reference.csv"
    reference.to_csv(path, index=False)

    rows = population_percentile_context(_profile(measured=True), reference_path=path)
    systolic = next(row for row in rows if row["Biomarker"] == "Systolic blood pressure")

    assert systolic["Reference sample"] == 100
    assert 30.0 <= systolic["Population percentile"] <= 35.0


def test_bundled_nhanes_extract_has_research_columns():
    frame = pd.read_csv(default_output_path())
    expected = {
        "age",
        "sleep_mean_hours",
        "exercise_days_per_week",
        "systolic_bp",
        "total_cholesterol",
        "hba1c",
        "fasting_glucose",
        "survey_weight",
    }

    assert len(frame) > 5000
    assert expected.issubset(frame.columns)


def test_biomarker_impact_compares_same_profile_without_measurements(monkeypatch):
    def fake_predict(profile):
        probability = 0.22 if profile.get("use_biomarkers") else 0.30
        return {"probability": probability, "feature_provenance": {"observed": ["trestbps", "chol"]}}

    monkeypatch.setattr(baseline_model, "predict_baseline_risk", fake_predict)
    measured_output = fake_predict(_profile(measured=True))
    impact = baseline_model.compare_biomarker_impact(
        _profile(measured=True), measured_output=measured_output
    )

    assert impact is not None
    assert impact["absolute_difference"] == pytest.approx(-0.08)
    assert impact["observed_model_features"] == ["trestbps", "chol"]
    assert impact["context_only_fields"] == ["hba1c", "bmi"]
