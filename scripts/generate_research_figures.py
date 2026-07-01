from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src.baseline_model import MODEL_FEATURES, _clean_training_frame, build_pipeline, predict_baseline_risk
from src.dataset import load_heart_dataset, project_root
from src.monte_carlo import run_monte_carlo
from src.optimizer import optimize_habit_plans


FIGURE_DIR = project_root() / "reports" / "figures"
METRICS_PATH = project_root() / "reports" / "research_metrics.json"
WORKED_EXAMPLE_PATH = project_root() / "reports" / "worked_example.json"
MODEL_CARD_PATH = project_root() / "research" / "model_card.md"
DATA_CARD_PATH = project_root() / "research" / "data_card.md"


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#fbfbfd")


def _research_profile() -> dict[str, object]:
    return {
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


def _bootstrap_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric: str,
    iterations: int = 1000,
) -> list[float]:
    rng = np.random.default_rng(2026)
    estimates: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        sampled_probabilities = probabilities[indices]
        if metric == "roc_auc":
            if np.unique(sampled_y).size < 2:
                continue
            estimates.append(float(roc_auc_score(sampled_y, sampled_probabilities)))
        elif metric == "accuracy":
            estimates.append(float(accuracy_score(sampled_y, sampled_probabilities >= 0.5)))
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return [float(low), float(high)]


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
    brier = float(brier_score_loss(y_test, probabilities))
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
        "brier_score": brier,
        "confusion_matrix": matrix.tolist(),
        "test_rows": int(len(X_test)),
    }


def generate_model_benchmark() -> dict[str, object]:
    data, _ = load_heart_dataset(try_download=False, allow_demo_fallback=False)
    clean = _clean_training_frame(data)
    X = clean[MODEL_FEATURES]
    y = clean["target"].astype(int).to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Logistic Regression": build_pipeline(),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=4,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Histogram Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        max_leaf_nodes=12,
                        learning_rate=0.05,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results: dict[str, dict[str, object]] = {}
    for name, model in models.items():
        probabilities = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
        accuracy = float(accuracy_score(y, probabilities >= 0.5))
        roc_auc = float(roc_auc_score(y, probabilities))
        brier = float(brier_score_loss(y, probabilities))
        results[name] = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "brier_score": brier,
        }
        if name == "Logistic Regression":
            results[name]["bootstrap_95pct_ci"] = {
                "accuracy": _bootstrap_interval(y, probabilities, metric="accuracy"),
                "roc_auc": _bootstrap_interval(y, probabilities, metric="roc_auc"),
            }

    names = list(results)
    auc_values = [float(results[name]["roc_auc"]) for name in names]
    brier_values = [float(results[name]["brier_score"]) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    fig.patch.set_facecolor("white")
    axes[0].bar(names, auc_values, color=["#0f766e", "#b45309", "#2563eb"])
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("5-fold out-of-fold ROC-AUC")
    axes[0].set_title("Discrimination")
    axes[0].tick_params(axis="x", rotation=15)
    _style_axis(axes[0])
    axes[1].bar(names, brier_values, color=["#0f766e", "#b45309", "#2563eb"])
    axes[1].set_ylabel("Brier score (lower is better)")
    axes[1].set_title("Probability Error")
    axes[1].tick_params(axis="x", rotation=15)
    _style_axis(axes[1])
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "model_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "evaluation": "5-fold stratified out-of-fold predictions",
        "rows": int(len(X)),
        "models": results,
        "selection_note": (
            "Logistic regression remains the production baseline for interpretability and stable probability behavior; "
            "nonlinear models are reported as benchmarks rather than silently substituted."
        ),
    }


def _group_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = probabilities >= 0.5
    result: dict[str, object] = {
        "rows": int(len(y)),
        "positive_rows": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "mean_predicted_probability": float(np.mean(probabilities)),
        "calibration_gap": float(np.mean(probabilities) - np.mean(y)),
    }
    if np.unique(y).size > 1:
        result["roc_auc"] = float(roc_auc_score(y, probabilities))
        result["roc_auc_bootstrap_95pct_ci"] = _bootstrap_interval(
            y, probabilities, metric="roc_auc", iterations=600
        )
    else:
        result["roc_auc"] = None
        result["roc_auc_bootstrap_95pct_ci"] = None
    return result


def generate_subgroup_evaluation() -> dict[str, object]:
    data, _ = load_heart_dataset(try_download=False, allow_demo_fallback=False)
    clean = _clean_training_frame(data)
    X = clean[MODEL_FEATURES]
    y = clean["target"].astype(int).to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(build_pipeline(), X, y, cv=folds, method="predict_proba")[:, 1]

    age = pd.to_numeric(clean["age"], errors="coerce")
    sex = pd.to_numeric(clean["sex"], errors="coerce")
    masks = {
        "Sex: Female": sex.eq(0).to_numpy(),
        "Sex: Male": sex.eq(1).to_numpy(),
        "Age: 29-49": age.between(29, 49).to_numpy(),
        "Age: 50-59": age.between(50, 59).to_numpy(),
        "Age: 60-77": age.between(60, 77).to_numpy(),
    }
    groups = {name: _group_metrics(y[mask], probabilities[mask]) for name, mask in masks.items()}

    labels = list(groups)
    auc_values = [float(groups[label]["roc_auc"]) for label in labels]
    brier_values = [float(groups[label]["brier_score"]) for label in labels]
    colors = ["#0f766e", "#0f766e", "#b45309", "#b45309", "#b45309"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    axes[0].barh(labels, auc_values, color=colors, alpha=0.88)
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xlabel("Out-of-fold ROC-AUC")
    axes[0].set_title("Discrimination by Subgroup")
    axes[1].barh(labels, brier_values, color=colors, alpha=0.88)
    axes[1].set_xlabel("Brier score (lower is better)")
    axes[1].set_title("Probability Error by Subgroup")
    for ax in axes:
        _style_axis(ax)
    fig.suptitle("Subgroup Evaluation (descriptive; small samples)")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "subgroup_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "evaluation": "Shared five-fold out-of-fold predictions, sliced after prediction",
        "groups": groups,
        "warning": "Subgroup sample sizes are small; differences are descriptive and not evidence of fairness or transportability.",
    }


def generate_missing_feature_ablation() -> dict[str, object]:
    data, _ = load_heart_dataset(try_download=False, allow_demo_fallback=False)
    clean = _clean_training_frame(data)
    X = clean[MODEL_FEATURES]
    y = clean["target"].astype(int).to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ablations = {
        "Full feature set": [],
        "No BP/cholesterol/glucose": ["trestbps", "chol", "fbs"],
        "No age/sex": ["age", "sex"],
        "No exercise-test signals": ["thalach", "exang", "oldpeak"],
        "No diagnostic categories": ["cp", "restecg", "slope", "ca", "thal"],
    }
    results: dict[str, dict[str, object]] = {}
    for name, removed_features in ablations.items():
        variant = X.drop(columns=removed_features) if removed_features else X.copy()
        probabilities = cross_val_predict(
            build_pipeline(), variant, y, cv=folds, method="predict_proba"
        )[:, 1]
        results[name] = {
            "removed_features": removed_features,
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "accuracy": float(accuracy_score(y, probabilities >= 0.5)),
            "brier_score": float(brier_score_loss(y, probabilities)),
        }

    baseline_auc = float(results["Full feature set"]["roc_auc"])
    baseline_brier = float(results["Full feature set"]["brier_score"])
    for metrics in results.values():
        metrics["roc_auc_change"] = float(metrics["roc_auc"]) - baseline_auc
        metrics["brier_score_change"] = float(metrics["brier_score"]) - baseline_brier

    labels = list(results)[1:]
    auc_changes = [float(results[label]["roc_auc_change"]) for label in labels]
    brier_changes = [float(results[label]["brier_score_change"]) for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].barh(labels, auc_changes, color="#0f766e", alpha=0.88)
    axes[0].axvline(0.0, color="#334155", linewidth=1)
    axes[0].set_xlabel("Change in ROC-AUC vs. full model")
    axes[0].set_title("Discrimination Impact")
    axes[1].barh(labels, brier_changes, color="#b45309", alpha=0.88)
    axes[1].axvline(0.0, color="#334155", linewidth=1)
    axes[1].set_xlabel("Change in Brier score vs. full model")
    axes[1].set_title("Probability Error Impact")
    for ax in axes:
        _style_axis(ax)
    fig.suptitle("Missing-Feature Group Ablation")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "missing_feature_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "evaluation": "Five-fold out-of-fold predictions with selected feature groups removed before fitting",
        "results": results,
        "interpretation": "Ablations measure predictive dependence in this dataset, not causal importance.",
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
    profile = _research_profile()
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


def generate_scenario_sensitivity() -> dict[str, object]:
    profile = _research_profile()
    baseline = predict_baseline_risk(profile)
    baseline_result = run_monte_carlo(
        profile,
        baseline,
        horizon_years=1,
        n_paths=1500,
        seed=42,
        threshold=0.6,
    )
    baseline_risk = float(baseline_result["expected_mean_risk"])
    scenarios = {
        "Sleep +1 hour": {"sleep_mean_hours": min(12.0, float(profile["sleep_mean_hours"]) + 1.0)},
        "Sleep variability -0.6h": {
            "sleep_variability_hours": max(0.0, float(profile["sleep_variability_hours"]) - 0.6)
        },
        "Exercise +2 days": {
            "exercise_days_per_week": min(7.0, float(profile["exercise_days_per_week"]) + 2.0)
        },
        "Stress -2 points": {"stress_score": max(1.0, float(profile["stress_score"]) - 2.0)},
        "Nutrition +2 points": {"nutrition_score": min(10.0, float(profile["nutrition_score"]) + 2.0)},
    }
    changes: dict[str, float] = {}
    scenario_risks: dict[str, float] = {}
    for name, updates in scenarios.items():
        scenario = dict(profile)
        scenario.update(updates)
        result = run_monte_carlo(
            scenario,
            baseline,
            horizon_years=1,
            n_paths=1500,
            seed=42,
            threshold=0.6,
        )
        scenario_risk = float(result["expected_mean_risk"])
        scenario_risks[name] = scenario_risk
        changes[name] = scenario_risk - baseline_risk

    ordered = sorted(changes.items(), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    values = [item[1] * 100.0 for item in ordered]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    fig.patch.set_facecolor("white")
    colors = ["#0f766e" if value <= 0 else "#b91c1c" for value in values]
    bars = ax.barh(labels, values, color=colors, alpha=0.88)
    ax.axvline(0.0, color="#334155", linewidth=1.0)
    ax.set_xlabel("Change in expected mean risk (percentage points)")
    ax.set_title("Common-Seed One-at-a-Time Scenario Sensitivity")
    _style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(
            value / 2.0,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha="center",
            color="white",
            fontweight="bold",
        )
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.18, top=0.88)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "scenario_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "baseline_expected_mean_risk": baseline_risk,
        "scenario_expected_mean_risk": scenario_risks,
        "change_from_baseline": changes,
        "simulation_paths": 1500,
        "common_seed": 42,
        "interpretation": "Scenario associations under SLCE assumptions; not causal-effect estimates.",
    }


def generate_worked_example() -> dict[str, object]:
    profile = _research_profile()
    profile["time_budget_minutes_per_day"] = 40
    baseline = predict_baseline_risk(profile)
    simulation = run_monte_carlo(
        profile,
        baseline,
        horizon_years=5,
        n_paths=1500,
        seed=42,
        threshold=0.6,
    )
    constraints = {
        "max_minutes_per_day": 40,
        "max_exercise_days_per_week": 6,
        "max_sleep_increase_per_week": 0.5,
        "max_stress_reduction_per_week": 1.0,
        "nutrition_improvement_cap_per_week": 1.0,
        "optimization_paths": 300,
    }
    optimization = optimize_habit_plans(
        profile,
        baseline,
        constraints,
        horizon_years=5,
        seed=123,
    )
    selected_fields = [
        "rank",
        "name",
        "target_sleep_mean",
        "target_exercise_days",
        "target_stress_score",
        "target_nutrition_score",
        "target_sleep_variability_hours",
        "expected_mean_risk",
        "expected_risk_reduction",
        "time_cost_minutes_per_day",
        "adherence_score",
    ]
    result = {
        "profile": profile,
        "baseline": {
            "probability": float(baseline["probability"]),
            "model_source": baseline.get("source", "unknown"),
        },
        "simulation": {
            "horizon_years": 5,
            "paths": 1500,
            "expected_mean_risk": float(simulation["expected_mean_risk"]),
            "final_risk_mean": float(simulation["final_risk_mean"]),
            "final_risk_p05": float(np.percentile(simulation["final_risk"], 5)),
            "final_risk_p95": float(np.percentile(simulation["final_risk"], 95)),
            "final_health_mean": float(simulation["final_health_mean"]),
        },
        "optimization": {
            "constraints": constraints,
            "candidates_evaluated": int(optimization["candidate_count"]),
            "baseline_expected_risk": float(optimization["baseline_expected_risk"]),
            "top_plans": [
                {field: plan[field] for field in selected_fields}
                for plan in optimization["top_plans"]
            ],
        },
        "interpretation": "Illustrative scenario output under SLCE assumptions; not a causal or clinical estimate.",
    }
    WORKED_EXAMPLE_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_research_cards(metrics: dict[str, object]) -> None:
    benchmark = metrics["model_benchmark"]
    logistic = benchmark["models"]["Logistic Regression"]
    subgroup_groups = metrics["subgroup_evaluation"]["groups"]
    subgroup_rows = "\n".join(
        f"| {name} | {values['rows']} | {values['roc_auc']:.3f} | {values['brier_score']:.3f} |"
        for name, values in subgroup_groups.items()
    )
    model_card = f"""# SLCE Baseline Model Card

## Model details
- **Model:** median imputation, standardization, logistic regression
- **Version:** generated with `python -m scripts.generate_research_figures`
- **Training data:** UCI Cleveland Heart Disease, 303 rows
- **Output:** binary heart-disease baseline probability used to initialize SLCE scenarios

## Intended use
Educational demonstration of transparent baseline modeling, uncertainty propagation, and constrained decision support. The probability is an input to a research simulation, not a diagnosis or treatment recommendation.

## Out-of-scope use
- Clinical diagnosis, screening, triage, or treatment selection
- Predictions for populations not represented by the training data
- Causal claims about lifestyle or laboratory changes

## Aggregate evaluation
Five-fold stratified out-of-fold results:

| Metric | Value |
|---|---:|
| ROC-AUC | {logistic['roc_auc']:.3f} |
| Accuracy | {logistic['accuracy']:.3f} |
| Brier score | {logistic['brier_score']:.3f} |
| ROC-AUC bootstrap 95% interval | {logistic['bootstrap_95pct_ci']['roc_auc'][0]:.3f}-{logistic['bootstrap_95pct_ci']['roc_auc'][1]:.3f} |

## Subgroup evaluation
These slices reuse shared out-of-fold predictions. Small samples make differences descriptive, not proof of fairness.

| Slice | Rows | ROC-AUC | Brier score |
|---|---:|---:|---:|
{subgroup_rows}

## Inputs and provenance
The app records whether each model feature is observed, proxy-derived, or median-imputed. Measured systolic blood pressure, total cholesterol, and fasting glucose can replace corresponding proxy/imputed fields. HbA1c and BMI remain context-only because this classifier was not trained with them.

## Limitations and risks
- Training ages span 29-77; teen outputs are out-of-distribution.
- The dataset is small, historical, single-center data and lacks broad demographic metadata.
- Many clinical fields are not available from the consumer form and require imputation.
- Calibration and subgroup estimates have substantial sampling uncertainty.
- The stochastic layer contains transparent assumptions, not learned treatment effects.

## Monitoring and reproducibility
- CI runs unit and Streamlit smoke tests.
- `reports/research_metrics.json` stores machine-readable evaluation output.
- `reports/figures/subgroup_performance.png` and `reports/figures/missing_feature_ablation.png` expose slice and missingness behavior.
"""
    MODEL_CARD_PATH.write_text(model_card, encoding="utf-8")

    coverage = metrics["nhanes"]["coverage_percent"]
    coverage_rows = "\n".join(
        f"| {field} | {value:.1f}% |" for field, value in coverage.items()
    )
    data_card = f"""# SLCE Data Card

## Dataset roles

### UCI Cleveland Heart Disease
- **Role:** supervised binary baseline model
- **Rows:** 303
- **License:** CC BY 4.0
- **Local artifact:** `data/heart.csv`
- **DOI:** `10.24432/C52P4X`

### CDC NHANES 2017-2018
- **Role:** survey-weighted population context for lifestyle and laboratory measurements
- **Processed rows:** {metrics['nhanes']['rows']:,}
- **Local artifact:** `data/nhanes_lifestyle_biomarkers.csv`
- **Source:** [CDC/NCHS NHANES 2017-2018](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)

The datasets are not row-concatenated because they contain different participants, schemas, sampling designs, and outcomes.

## NHANES measurement coverage
| Field | Non-missing coverage |
|---|---:|
{coverage_rows}

Fasting glucose is collected in a subsample, so its lower coverage is expected. SLCE retains survey weights, strata, and PSU columns; in-app percentiles use positive examination weights, while full design-based variance estimation remains future work.

## Processing
- Official CDC XPT files are merged on `SEQN`.
- Physiologically impossible values are set missing using explicit bounds.
- Adult activity days use the maximum of vigorous/moderate recreation days to avoid double-counting unknown overlap.
- The processed extract can be rebuilt with `python -m src.nhanes_dataset --build`.

## Missingness and quality risks
- Laboratory eligibility, fasting subsampling, and nonresponse vary by measurement and participant characteristics.
- UCI missing model fields use fold-local medians during evaluation and training medians at inference.
- User-uploaded lab CSVs are validated for one-row structure, units, numeric type, and bounded ranges.
- No names or personal identifiers are required or retained by the import format.

## Responsible use
NHANES percentiles describe position in an age/sex reference sample and are not clinical thresholds. See the [CDC laboratory overview](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overviewlab.aspx?BeginYear=2017) for collection, quality-control, subsampling, and analytic guidance.
"""
    DATA_CARD_PATH.write_text(data_card, encoding="utf-8")


def main() -> int:
    metrics = {
        "baseline": generate_baseline_evaluation(),
        "model_benchmark": generate_model_benchmark(),
        "subgroup_evaluation": generate_subgroup_evaluation(),
        "missing_feature_ablation": generate_missing_feature_ablation(),
        "nhanes": generate_nhanes_coverage(),
        "monte_carlo_convergence": generate_monte_carlo_convergence(),
        "scenario_sensitivity": generate_scenario_sensitivity(),
        "worked_example": generate_worked_example(),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_research_cards(metrics)
    print(f"Generated figures in {FIGURE_DIR}")
    print(f"Saved metrics to {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
