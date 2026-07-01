from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.baseline_model import MODEL_FEATURES, _clean_training_frame, build_pipeline, predict_baseline_risk
from src.dataset import load_heart_dataset, project_root
from src.monte_carlo import run_monte_carlo


FIGURE_DIR = project_root() / "reports" / "figures"
METRICS_PATH = project_root() / "reports" / "research_metrics.json"


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#fbfbfd")


def generate_baseline_evaluation() -> dict[str, object]:
    data, _ = load_heart_dataset(try_download=False, allow_demo_fallback=False)
    clean = _clean_training_frame(data)
    X = clean[MODEL_FEATURES]
    y = clean["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    roc_auc = float(roc_auc_score(y_test, probabilities))
    matrix = confusion_matrix(y_test, predictions)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    observed, predicted = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    fig.patch.set_facecolor("white")
    axes[0].plot(false_positive_rate, true_positive_rate, color="#0f766e", linewidth=2.2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
    axes[0].set_title(f"Holdout ROC (AUC={roc_auc:.3f})")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    _style_axis(axes[0])

    axes[1].plot(predicted, observed, marker="o", color="#b45309", linewidth=2.0)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
    axes[1].set_title("Holdout Calibration")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed positive rate")
    _style_axis(axes[1])

    ConfusionMatrixDisplay(matrix).plot(ax=axes[2], colorbar=False, cmap="Blues")
    axes[2].set_title(f"Confusion Matrix (accuracy={accuracy:.3f})")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "baseline_evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.tolist(),
        "test_rows": int(len(X_test)),
    }


def generate_nhanes_coverage() -> dict[str, object]:
    path = project_root() / "data" / "nhanes_lifestyle_biomarkers.csv"
    frame = pd.read_csv(path)
    fields = [
        "sleep_mean_hours",
        "exercise_days_per_week",
        "resting_hr",
        "systolic_bp",
        "bmi",
        "total_cholesterol",
        "hdl_cholesterol",
        "hba1c",
        "fasting_glucose",
    ]
    labels = [field.replace("_", " ").title() for field in fields]
    coverage = frame[fields].notna().mean().mul(100.0)

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    fig.patch.set_facecolor("white")
    bars = ax.barh(labels, coverage.values, color="#0f766e", alpha=0.88)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Records with measurement (%)")
    ax.set_title(f"NHANES Research Extract Coverage (n={len(frame):,})")
    _style_axis(ax)
    for bar, value in zip(bars, coverage.values):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "nhanes_measurement_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "rows": int(len(frame)),
        "coverage_percent": {field: float(coverage[field]) for field in fields},
    }


def generate_monte_carlo_convergence() -> dict[str, object]:
    profile = {
        "age": 45,
        "sex": "Male",
        "resting_hr": 72,
        "sleep_mean_hours": 6.8,
        "sleep_variability_hours": 1.1,
        "exercise_days_per_week": 3,
        "stress_score": 6,
        "nutrition_score": 6,
        "use_biomarkers": True,
        "systolic_bp": 128.0,
        "total_cholesterol": 198.0,
        "fasting_glucose": 96.0,
        "hba1c": 5.5,
        "bmi": 25.4,
    }
    baseline = predict_baseline_risk(profile)
    path_counts = [100, 250, 500, 1000, 2000]
    estimates: list[list[float]] = []
    for path_count in path_counts:
        repetitions = []
        for seed in (7, 17, 27, 37, 47):
            result = run_monte_carlo(
                profile,
                baseline,
                horizon_years=1,
                n_paths=path_count,
                seed=seed,
                threshold=0.6,
            )
            repetitions.append(float(result["expected_mean_risk"]))
        estimates.append(repetitions)

    estimate_array = np.asarray(estimates)
    means = estimate_array.mean(axis=1)
    standard_deviations = estimate_array.std(axis=1, ddof=1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor("white")
    ax.errorbar(
        path_counts,
        means,
        yerr=standard_deviations,
        marker="o",
        capsize=4,
        linewidth=2.0,
        color="#b45309",
    )
    ax.set_xscale("log")
    ax.set_xticks(path_counts, labels=[str(value) for value in path_counts])
    ax.set_xlabel("Monte Carlo paths")
    ax.set_ylabel("Expected mean risk")
    ax.set_title("Monte Carlo Convergence Across Five Random Seeds")
    _style_axis(ax)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "monte_carlo_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "path_counts": path_counts,
        "mean_estimates": means.tolist(),
        "standard_deviations": standard_deviations.tolist(),
        "seeds": [7, 17, 27, 37, 47],
    }


def main() -> int:
    metrics = {
        "baseline": generate_baseline_evaluation(),
        "nhanes": generate_nhanes_coverage(),
        "monte_carlo_convergence": generate_monte_carlo_convergence(),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Generated figures in {FIGURE_DIR}")
    print(f"Saved metrics to {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
