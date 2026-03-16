from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def clamp(value: float | np.ndarray, min_value: float, max_value: float):
    return np.clip(value, min_value, max_value)


def sigmoid(x: float | np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class PersonalizationWeights:
    w_sleep: float = 1.0
    w_stress: float = 1.0
    w_exercise: float = 1.0
    w_nutrition: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "w_sleep": float(self.w_sleep),
            "w_stress": float(self.w_stress),
            "w_exercise": float(self.w_exercise),
            "w_nutrition": float(self.w_nutrition),
        }

    @classmethod
    def from_any(cls, value: "PersonalizationWeights | dict[str, Any] | None") -> "PersonalizationWeights":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                w_sleep=float(value.get("w_sleep", 1.0)),
                w_stress=float(value.get("w_stress", 1.0)),
                w_exercise=float(value.get("w_exercise", 1.0)),
                w_nutrition=float(value.get("w_nutrition", 1.0)),
            )
        return cls()


DEFAULT_RISK_ALPHA = -2.2
DEFAULT_RISK_BETA = 3.0


def baseline_risk_to_health(baseline_risk: float) -> float:
    baseline_risk = float(clamp(baseline_risk, 0.0, 1.0))
    return float(clamp(100.0 - 60.0 * baseline_risk, 20.0, 95.0))


def risk_from_health(
    health_state: float | np.ndarray,
    baseline_logit: float,
    *,
    alpha: float = DEFAULT_RISK_ALPHA,
    beta: float = DEFAULT_RISK_BETA,
):
    normalized_health_load = (100.0 - np.asarray(health_state)) / 100.0
    logit = alpha + beta * normalized_health_load + float(baseline_logit)
    return sigmoid(logit), logit


def effective_noise_sigma(*, sleep_variability_hours: float, stress_score: float) -> float:
    sigma = 0.8 + 0.45 * float(sleep_variability_hours) + 0.12 * max(0.0, float(stress_score) - 4.0)
    return float(clamp(sigma, 0.5, 4.0))


def compute_weekly_drift(
    *,
    sleep_mean_hours: float,
    exercise_days_per_week: float,
    stress_score: float,
    nutrition_score: float,
    weights: PersonalizationWeights | dict[str, float] | None = None,
) -> float:
    w = PersonalizationWeights.from_any(weights)

    sleep_term = 0.60 * w.w_sleep * (float(sleep_mean_hours) - 7.5)
    exercise_term = 0.45 * w.w_exercise * (float(exercise_days_per_week) - 3.0)
    stress_term = -0.55 * w.w_stress * (float(stress_score) - 5.0)
    nutrition_term = 0.35 * w.w_nutrition * (float(nutrition_score) - 6.0)

    penalty = 0.0
    if sleep_mean_hours < 5.5:
        penalty -= 0.8
    if stress_score > 8:
        penalty -= 0.6
    if exercise_days_per_week == 0:
        penalty -= 0.4

    drift = sleep_term + exercise_term + stress_term + nutrition_term + penalty
    return float(clamp(drift, -3.0, 3.0))


def _step_toward(current: float, target: float, max_delta: float) -> float:
    current = float(current)
    target = float(target)
    max_delta = max(0.0, float(max_delta))
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


def build_weekly_schedule(
    baseline_profile: dict[str, Any],
    weeks: int,
    plan: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    current = {
        "sleep_mean_hours": float(baseline_profile["sleep_mean_hours"]),
        "exercise_days_per_week": float(baseline_profile["exercise_days_per_week"]),
        "stress_score": float(baseline_profile["stress_score"]),
        "nutrition_score": float(baseline_profile["nutrition_score"]),
        "sleep_variability_hours": float(baseline_profile["sleep_variability_hours"]),
    }
    if not plan:
        return [current.copy() for _ in range(weeks)]

    targets = {
        "sleep_mean_hours": float(plan.get("target_sleep_mean", current["sleep_mean_hours"])),
        "exercise_days_per_week": float(plan.get("target_exercise_days", current["exercise_days_per_week"])),
        "stress_score": float(plan.get("target_stress_score", current["stress_score"])),
        "nutrition_score": float(plan.get("target_nutrition_score", current["nutrition_score"])),
        "sleep_variability_hours": float(
            plan.get("target_sleep_variability_hours", current["sleep_variability_hours"])
        ),
    }
    ramp = {
        "sleep_mean_hours": float(plan.get("max_sleep_increase_per_week", 0.5)),
        "exercise_days_per_week": float(plan.get("max_exercise_days_increase_per_week", 1.0)),
        "stress_score": float(plan.get("max_stress_reduction_per_week", 1.0)),
        "nutrition_score": float(plan.get("nutrition_improvement_cap_per_week", 1.0)),
        "sleep_variability_hours": float(plan.get("max_sleep_variability_change_per_week", 0.25)),
    }

    schedule: list[dict[str, float]] = []
    for _ in range(weeks):
        # Sleep and exercise move upward toward targets; stress may move down; variability often down.
        current["sleep_mean_hours"] = _step_toward(
            current["sleep_mean_hours"], targets["sleep_mean_hours"], ramp["sleep_mean_hours"]
        )
        current["exercise_days_per_week"] = _step_toward(
            current["exercise_days_per_week"],
            targets["exercise_days_per_week"],
            ramp["exercise_days_per_week"],
        )
        current["stress_score"] = _step_toward(
            current["stress_score"], targets["stress_score"], ramp["stress_score"]
        )
        current["nutrition_score"] = _step_toward(
            current["nutrition_score"], targets["nutrition_score"], ramp["nutrition_score"]
        )
        current["sleep_variability_hours"] = _step_toward(
            current["sleep_variability_hours"],
            targets["sleep_variability_hours"],
            ramp["sleep_variability_hours"],
        )

        # Keep all values inside UI ranges.
        current["sleep_mean_hours"] = float(clamp(current["sleep_mean_hours"], 3.0, 12.0))
        current["exercise_days_per_week"] = float(clamp(current["exercise_days_per_week"], 0.0, 7.0))
        current["stress_score"] = float(clamp(current["stress_score"], 1.0, 10.0))
        current["nutrition_score"] = float(clamp(current["nutrition_score"], 1.0, 10.0))
        current["sleep_variability_hours"] = float(clamp(current["sleep_variability_hours"], 0.0, 3.0))
        schedule.append(current.copy())

    return schedule


def simulate_single_path(
    profile: dict[str, Any],
    baseline_logit: float,
    baseline_risk: float,
    *,
    horizon_years: int = 1,
    plan: dict[str, Any] | None = None,
    weights: PersonalizationWeights | dict[str, float] | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    weeks = 52 * int(horizon_years)
    rng = np.random.default_rng(seed)
    schedule = build_weekly_schedule(profile, weeks, plan)
    w = PersonalizationWeights.from_any(weights)

    health = np.zeros(weeks + 1, dtype=float)
    risk = np.zeros(weeks + 1, dtype=float)
    health[0] = baseline_risk_to_health(baseline_risk)
    risk[0], _ = risk_from_health(health[0], baseline_logit)

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
        mean_reversion = -0.02 * (health[t] - 70.0)
        noise = rng.normal(0.0, sigma)
        health[t + 1] = float(clamp(health[t] + drift + mean_reversion + noise, 0.0, 100.0))
        risk[t + 1], _ = risk_from_health(health[t + 1], baseline_logit)

    return {"health": health, "risk": risk}


def quick_health_projection_score(
    profile: dict[str, Any],
    weights: PersonalizationWeights | dict[str, float] | None = None,
) -> float:
    w = PersonalizationWeights.from_any(weights)
    drift = compute_weekly_drift(
        sleep_mean_hours=float(profile["sleep_mean_hours"]),
        exercise_days_per_week=float(profile["exercise_days_per_week"]),
        stress_score=float(profile["stress_score"]),
        nutrition_score=float(profile["nutrition_score"]),
        weights=w,
    )
    sigma = effective_noise_sigma(
        sleep_variability_hours=float(profile["sleep_variability_hours"]),
        stress_score=float(profile["stress_score"]),
    )
    # A deterministic proxy used by personalization to compare weeks without expensive simulation.
    return float(70.0 + 4.0 * drift - 1.2 * sigma)

