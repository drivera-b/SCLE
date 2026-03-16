from __future__ import annotations

from typing import Any

import numpy as np

from .stochastic_model import (
    PersonalizationWeights,
    baseline_risk_to_health,
    build_weekly_schedule,
    clamp,
    compute_weekly_drift,
    effective_noise_sigma,
    risk_from_health,
)


def summarize_paths(
    health_paths: np.ndarray,
    risk_paths: np.ndarray,
    *,
    threshold: float = 0.6,
) -> dict[str, Any]:
    health_paths = np.asarray(health_paths, dtype=float)
    risk_paths = np.asarray(risk_paths, dtype=float)

    final_health = health_paths[:, -1]
    final_risk = risk_paths[:, -1]
    mean_risk_per_path = risk_paths.mean(axis=1)
    exceed = (risk_paths >= threshold).any(axis=1)

    return {
        "expected_mean_risk": float(np.mean(mean_risk_per_path)),
        "prob_exceed_threshold": float(np.mean(exceed)),
        "final_health_mean": float(np.mean(final_health)),
        "final_health_std": float(np.std(final_health)),
        "final_risk_mean": float(np.mean(final_risk)),
        "final_risk_std": float(np.std(final_risk)),
        "final_health": final_health,
        "final_risk": final_risk,
        "mean_risk_per_path": mean_risk_per_path,
        "risk_median": np.median(risk_paths, axis=0),
        "risk_p05": np.percentile(risk_paths, 5, axis=0),
        "risk_p95": np.percentile(risk_paths, 95, axis=0),
        "health_median": np.median(health_paths, axis=0),
        "health_p05": np.percentile(health_paths, 5, axis=0),
        "health_p95": np.percentile(health_paths, 95, axis=0),
    }


def run_monte_carlo(
    profile: dict[str, Any],
    baseline_output: dict[str, Any],
    *,
    plan: dict[str, Any] | None = None,
    weights: dict[str, float] | PersonalizationWeights | None = None,
    horizon_years: int = 1,
    n_paths: int = 1000,
    seed: int = 42,
    threshold: float = 0.6,
) -> dict[str, Any]:
    weeks = 52 * int(horizon_years)
    n_paths = int(n_paths)
    n_paths = max(1, n_paths)
    rng = np.random.default_rng(seed)
    w = PersonalizationWeights.from_any(weights)
    schedule = build_weekly_schedule(profile, weeks, plan)

    health = np.zeros((n_paths, weeks + 1), dtype=np.float32)
    risk = np.zeros((n_paths, weeks + 1), dtype=np.float32)
    baseline_risk = float(baseline_output["probability"])
    baseline_logit = float(baseline_output["logit"])
    health[:, 0] = baseline_risk_to_health(baseline_risk)
    risk[:, 0], _ = risk_from_health(health[:, 0], baseline_logit)

    for t in range(weeks):
        habits = schedule[t]
        drift = compute_weekly_drift(
            sleep_mean_hours=habits["sleep_mean_hours"],
            exercise_days_per_week=habits["exercise_days_per_week"],
            stress_score=habits["stress_score"],
            nutrition_score=habits["nutrition_score"],
            weights=w,
        )
        sigma = effective_noise_sigma(
            sleep_variability_hours=habits["sleep_variability_hours"],
            stress_score=habits["stress_score"],
        )
        noise = rng.normal(0.0, sigma, size=n_paths).astype(np.float32)
        mean_reversion = -0.02 * (health[:, t] - 70.0)
        health[:, t + 1] = clamp(health[:, t] + drift + mean_reversion + noise, 0.0, 100.0)
        risk[:, t + 1], _ = risk_from_health(health[:, t + 1], baseline_logit)

    summary = summarize_paths(health, risk, threshold=threshold)
    summary.update(
        {
            "weeks": weeks,
            "time_axis_weeks": np.arange(weeks + 1),
            "health_paths": health,
            "risk_paths": risk,
            "schedule": schedule,
            "threshold": float(threshold),
        }
    )
    return summary

