from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import TRAINING_COLUMNS, load_heart_dataset, project_root
from .validate import sex_to_numeric

try:
    import joblib

    JOBLIB_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    joblib = None  # type: ignore[assignment]
    JOBLIB_IMPORT_ERROR = exc

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    SKLEARN_IMPORT_ERROR = exc
    SimpleImputer = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    confusion_matrix = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    train_test_split = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]


MODEL_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def models_dir() -> Path:
    path = project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path() -> Path:
    return models_dir() / "baseline_model.joblib"


def metadata_path() -> Path:
    return models_dir() / "baseline_metadata.json"


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _require_sklearn() -> None:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-learn is required for training/loading the baseline model. "
            f"Import error: {SKLEARN_IMPORT_ERROR}"
        )


def _require_joblib() -> None:
    if JOBLIB_IMPORT_ERROR is not None:
        raise RuntimeError(f"joblib is required to save/load model artifacts. Import error: {JOBLIB_IMPORT_ERROR}")


def build_pipeline() -> "Pipeline":
    _require_sklearn()
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1500, solver="lbfgs")),
        ]
    )


def _clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TRAINING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    clean = df.copy()
    for col in TRAINING_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean["target"] = clean["target"].fillna(0).astype(int)
    # Keep rows with target and at least a few valid feature values.
    clean = clean.dropna(subset=["target"])
    non_nan_counts = clean[MODEL_FEATURES].notna().sum(axis=1)
    clean = clean.loc[non_nan_counts >= 5].reset_index(drop=True)
    return clean


def train_baseline_model(
    *,
    try_download: bool = True,
    allow_demo_fallback: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    _require_sklearn()
    _require_joblib()
    df, source_info = load_heart_dataset(try_download=try_download, allow_demo_fallback=allow_demo_fallback)
    df = _clean_training_frame(df)
    if len(df) < 20:
        raise ValueError("Dataset is too small to train a stable baseline model (need at least 20 rows).")

    X = df[MODEL_FEATURES]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_test.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    medians = {feature: float(df[feature].median(skipna=True)) for feature in MODEL_FEATURES}
    feature_ranges = {
        feature: {
            "min": float(df[feature].min(skipna=True)),
            "max": float(df[feature].max(skipna=True)),
        }
        for feature in MODEL_FEATURES
    }
    metadata = {
        "project": "SLCE",
        "model_type": "logistic_regression",
        "features": MODEL_FEATURES,
        "metrics": metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": source_info,
        "feature_medians": medians,
        "feature_ranges": feature_ranges,
    }

    joblib.dump(pipeline, model_path())
    metadata_path().write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"pipeline": pipeline, "metadata": metadata}


def load_baseline_artifacts() -> tuple[Any | None, dict[str, Any] | None]:
    pipeline = None
    metadata = None
    if metadata_path().exists():
        try:
            metadata = json.loads(metadata_path().read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metadata = None
    if model_path().exists():
        try:
            if joblib is not None:
                pipeline = joblib.load(model_path())
        except Exception:
            pipeline = None
    return pipeline, metadata


def ensure_baseline_artifacts() -> tuple[Any | None, dict[str, Any] | None]:
    pipeline, metadata = load_baseline_artifacts()
    if pipeline is not None and metadata is not None:
        return pipeline, metadata
    if SKLEARN_IMPORT_ERROR is not None or JOBLIB_IMPORT_ERROR is not None:
        return pipeline, metadata
    try:
        trained = train_baseline_model(try_download=True, allow_demo_fallback=True)
        return trained["pipeline"], trained["metadata"]
    except Exception:
        return pipeline, metadata


def _estimated_thalach_from_resting_hr(age: float, resting_hr: float) -> float:
    # Rough heuristic to populate the UCI `thalach` feature when only resting HR is known.
    est = 208.0 - 0.7 * age - 0.30 * (resting_hr - 60.0)
    return float(np.clip(est, 80.0, 205.0))


def _profile_to_model_row(
    profile: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    medians = dict((metadata or {}).get("feature_medians", {}))
    row = {feature: float(medians.get(feature, 0.0)) for feature in MODEL_FEATURES}
    observed: set[str] = set()
    proxy: set[str] = set()

    age = float(profile.get("age", 18))
    sex = sex_to_numeric(profile.get("sex", "Unknown"))
    resting_hr = float(profile.get("resting_hr", 70))
    stress_score = float(profile.get("stress_score", 5))
    exercise_days = float(profile.get("exercise_days_per_week", 3))

    row["age"] = age
    row["sex"] = float(sex)
    observed.add("age")
    if str(profile.get("sex", "Unknown")) in {"Male", "Female"}:
        observed.add("sex")
    else:
        proxy.add("sex")
    row["thalach"] = _estimated_thalach_from_resting_hr(age, resting_hr)
    proxy.add("thalach")

    use_biomarkers = bool(profile.get("use_biomarkers", False))
    systolic_bp = profile.get("systolic_bp")
    total_cholesterol = profile.get("total_cholesterol")
    fasting_glucose = profile.get("fasting_glucose")

    if use_biomarkers and systolic_bp is not None:
        row["trestbps"] = float(np.clip(float(systolic_bp), 70.0, 250.0))
        observed.add("trestbps")
    elif "trestbps" in row:
        row["trestbps"] = float(np.clip(row["trestbps"] + 1.8 * (stress_score - 5.0), 80.0, 220.0))
        proxy.add("trestbps")

    if use_biomarkers and total_cholesterol is not None:
        row["chol"] = float(np.clip(float(total_cholesterol), 80.0, 500.0))
        observed.add("chol")
    elif "chol" in row:
        nutrition_score = float(profile.get("nutrition_score", 6))
        row["chol"] = float(np.clip(row["chol"] - 3.5 * (nutrition_score - 6.0), 100.0, 450.0))
        proxy.add("chol")

    if use_biomarkers and fasting_glucose is not None:
        row["fbs"] = 1.0 if float(fasting_glucose) > 120.0 else 0.0
        observed.add("fbs")

    if "oldpeak" in row:
        row["oldpeak"] = float(np.clip(row["oldpeak"] + 0.08 * max(0.0, 2.0 - exercise_days), 0.0, 6.5))
        proxy.add("oldpeak")

    imputed = set(MODEL_FEATURES) - observed - proxy
    provenance = {
        "observed": sorted(observed),
        "proxy_derived": sorted(proxy),
        "imputed_from_training_median": sorted(imputed),
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES), provenance


def heuristic_baseline_risk(profile: dict[str, Any]) -> dict[str, Any]:
    age = float(profile.get("age", 18))
    sex = sex_to_numeric(profile.get("sex", "Unknown"))
    resting_hr = float(profile.get("resting_hr", 70))
    stress = float(profile.get("stress_score", 5))
    exercise = float(profile.get("exercise_days_per_week", 3))
    sleep = float(profile.get("sleep_mean_hours", 7))
    nutrition = float(profile.get("nutrition_score", 6))

    logit = (
        -2.0
        + 0.030 * (age - 18.0)
        + 0.18 * sex
        + 0.020 * (resting_hr - 70.0)
        + 0.22 * (stress - 5.0)
        - 0.18 * (exercise - 3.0)
        - 0.12 * (sleep - 7.5)
        - 0.10 * (nutrition - 6.0)
    )
    p = _sigmoid(logit)
    return {
        "probability": p,
        "logit": _logit(p),
        "source": "heuristic_fallback",
        "model_loaded": False,
        "metadata": None,
        "note": "Using heuristic fallback because a trained baseline model was unavailable.",
    }


def predict_baseline_risk(profile: dict[str, Any]) -> dict[str, Any]:
    pipeline, metadata = ensure_baseline_artifacts()
    if pipeline is None or metadata is None:
        return heuristic_baseline_risk(profile)

    try:
        row, provenance = _profile_to_model_row(profile, metadata)
        prob = float(pipeline.predict_proba(row)[:, 1][0])
        return {
            "probability": prob,
            "logit": _logit(prob),
            "source": "trained_model",
            "model_loaded": True,
            "metadata": metadata,
            "feature_provenance": provenance,
        }
    except Exception as exc:
        fallback = heuristic_baseline_risk(profile)
        fallback["note"] = (
            "Fell back to heuristic baseline because prediction using the trained model failed: "
            f"{exc.__class__.__name__}"
        )
        return fallback


def compare_biomarker_impact(
    profile: dict[str, Any],
    *,
    measured_output: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compare measured labs with the same profile using proxy/imputed lab features."""
    if not bool(profile.get("use_biomarkers", False)):
        return None
    measured = measured_output or predict_baseline_risk(profile)
    proxy_profile = dict(profile)
    proxy_profile["use_biomarkers"] = False
    proxy = predict_baseline_risk(proxy_profile)
    measured_probability = float(measured["probability"])
    proxy_probability = float(proxy["probability"])
    provenance = measured.get("feature_provenance", {})
    return {
        "measured_probability": measured_probability,
        "proxy_probability": proxy_probability,
        "absolute_difference": measured_probability - proxy_probability,
        "observed_model_features": provenance.get("observed", []),
        "context_only_fields": [
            field for field in ("hba1c", "bmi") if profile.get(field) not in (None, "")
        ],
        "interpretation": (
            "Difference caused by replacing proxy/imputed model inputs with supplied measurements; "
            "not a causal effect or a diagnosis."
        ),
    }


def _print_metrics(metadata: dict[str, Any] | None) -> None:
    if not metadata:
        print("No metadata available.")
        return
    metrics = metadata.get("metrics", {})
    print("Training complete.")
    print(f"Features: {', '.join(metadata.get('features', []))}")
    print(f"Accuracy: {metrics.get('accuracy')}")
    print(f"ROC-AUC: {metrics.get('roc_auc')}")
    print(f"Confusion matrix: {metrics.get('confusion_matrix')}")
    print(f"Artifacts: {model_path()} and {metadata_path()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or inspect the SLCE baseline model.")
    parser.add_argument("--train", action="store_true", help="Train and save the baseline model artifacts.")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt dataset download.")
    args = parser.parse_args(argv)

    if args.train:
        try:
            result = train_baseline_model(try_download=not args.no_download, allow_demo_fallback=True)
            _print_metrics(result["metadata"])
            return 0
        except Exception as exc:
            print(f"Training failed: {exc}")
            return 1

    pipeline, metadata = load_baseline_artifacts()
    print(f"Model file exists: {model_path().exists()} (loaded: {pipeline is not None})")
    print(f"Metadata file exists: {metadata_path().exists()} (loaded: {metadata is not None})")
    if metadata:
        _print_metrics(metadata)
    else:
        print("Run `python -m src.baseline_model --train` to generate artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
