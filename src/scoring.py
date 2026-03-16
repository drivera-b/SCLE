from __future__ import annotations


def expected_risk_score(mean_risk: float) -> float:
    return float(mean_risk)


def time_cost_score(minutes_per_day: float) -> float:
    return float(minutes_per_day) / 180.0


def adherence_score(
    *,
    sleep_change: float,
    exercise_change: float,
    stress_change: float,
    nutrition_change: float,
) -> float:
    # Smaller required changes imply better adherence probability.
    difficulty = (
        abs(sleep_change) / 3.0
        + abs(exercise_change) / 7.0
        + abs(stress_change) / 9.0
        + abs(nutrition_change) / 9.0
    )
    score = max(0.0, 1.0 - 0.35 * difficulty)
    return float(score)


def composite_objective(
    *,
    expected_mean_risk: float,
    time_cost_minutes_per_day: float,
    adherence: float,
    lambda_time: float = 0.20,
    lambda_adherence: float = 0.15,
) -> float:
    return float(
        expected_risk_score(expected_mean_risk)
        + lambda_time * time_cost_score(time_cost_minutes_per_day)
        - lambda_adherence * adherence
    )
