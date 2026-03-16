from __future__ import annotations

from typing import Any

import numpy as np

from .stochastic_model import PersonalizationWeights, clamp, quick_health_projection_score


def default_weights() -> dict[str, float]:
    return PersonalizationWeights().to_dict()


def summarize_weekly_log(log: dict[str, Any]) -> dict[str, float]:
    sleep = np.asarray(log["sleep_hours"], dtype=float)
    stress = np.asarray(log["stress_scores"], dtype=float)
    exercise = np.asarray(log["exercise_minutes"], dtype=float)
    nutrition = np.asarray(log.get("nutrition_scores", [6.0] * 7), dtype=float)
    return {
        "sleep_mean_hours": float(np.mean(sleep)),
        "sleep_variability_hours": float(np.std(sleep)),
        "stress_score": float(np.mean(stress)),
        "exercise_days_per_week": float(np.sum(exercise >= 20)),
        "exercise_minutes_per_day_mean": float(np.mean(exercise)),
        "nutrition_score": float(np.mean(nutrition)),
    }


def _corr(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 3 or len(values_x) != len(values_y):
        return None
    if len(set(round(x, 5) for x in values_x)) <= 1:
        return None
    if len(set(round(y, 5) for y in values_y)) <= 1:
        return None
    return float(np.corrcoef(values_x, values_y)[0, 1])


def _bounded_weight(value: float) -> float:
    return float(clamp(value, 0.6, 1.8))


def update_personalization_weights(
    previous_weights: dict[str, float] | None,
    *,
    baseline_profile: dict[str, Any],
    log_history: list[dict[str, Any]],
) -> dict[str, Any]:
    prev = PersonalizationWeights.from_any(previous_weights).to_dict()
    history = [dict(item) for item in log_history]

    if not history:
        return {
            "previous_weights": prev,
            "new_weights": prev.copy(),
            "explanation": "No weekly logs were available, so the personalization weights stayed the same.",
            "history": history,
        }

    # Ensure summaries and deterministic outcomes exist for every history row.
    processed: list[dict[str, Any]] = []
    for item in history:
        summary = dict(item.get("summary", {}))
        if not summary:
            summary = summarize_weekly_log(item)
        profile_for_week = {
            **baseline_profile,
            "sleep_mean_hours": summary["sleep_mean_hours"],
            "sleep_variability_hours": summary["sleep_variability_hours"],
            "exercise_days_per_week": summary["exercise_days_per_week"],
            "stress_score": summary["stress_score"],
            "nutrition_score": summary.get("nutrition_score", float(baseline_profile["nutrition_score"])),
        }
        outcome_proxy = quick_health_projection_score(profile_for_week, prev)
        processed.append({**item, "summary": summary, "outcome_proxy": outcome_proxy})

    outcomes = [float(item["outcome_proxy"]) for item in processed]
    sleep_series = [float(item["summary"]["sleep_mean_hours"]) for item in processed]
    stress_series = [float(item["summary"]["stress_score"]) for item in processed]
    exercise_series = [float(item["summary"]["exercise_days_per_week"]) for item in processed]
    nutrition_series = [float(item["summary"]["nutrition_score"]) for item in processed]

    corrs = {
        "w_sleep": _corr(sleep_series, outcomes),
        "w_stress": _corr(stress_series, outcomes),
        "w_exercise": _corr(exercise_series, outcomes),
        "w_nutrition": _corr(nutrition_series, outcomes),
    }

    new_weights = prev.copy()
    explanations: list[str] = []
    rules = {
        "w_sleep": {"expected_sign": 1, "label": "sleep"},
        "w_stress": {"expected_sign": -1, "label": "stress"},
        "w_exercise": {"expected_sign": 1, "label": "exercise"},
        "w_nutrition": {"expected_sign": 1, "label": "nutrition"},
    }

    for key, rule in rules.items():
        corr = corrs[key]
        if corr is None:
            continue
        expected_sign = int(rule["expected_sign"])
        sign = 1 if corr > 0 else -1 if corr < 0 else 0
        magnitude = min(1.0, abs(corr))
        if sign == 0:
            continue
        if sign == expected_sign:
            delta = 0.12 * magnitude
            new_weights[key] = _bounded_weight(new_weights[key] + delta)
            explanations.append(
                f"{rule['label'].capitalize()} weight increased slightly (correlation {corr:+.2f} matched expected direction)."
            )
        else:
            delta = 0.06 * magnitude
            new_weights[key] = _bounded_weight(new_weights[key] - delta)
            explanations.append(
                f"{rule['label'].capitalize()} weight decreased slightly (correlation {corr:+.2f} was weaker/opposite)."
            )

    if not explanations:
        latest = processed[-1]["summary"]
        if latest["stress_score"] >= 7.0:
            new_weights["w_stress"] = _bounded_weight(new_weights["w_stress"] + 0.05)
            explanations.append("Stress weight increased because the recent week had high average stress.")
        if latest["sleep_mean_hours"] <= 6.0:
            new_weights["w_sleep"] = _bounded_weight(new_weights["w_sleep"] + 0.05)
            explanations.append("Sleep weight increased because the recent week had low average sleep.")
        if latest["exercise_days_per_week"] >= 4:
            new_weights["w_exercise"] = _bounded_weight(new_weights["w_exercise"] + 0.03)
            explanations.append("Exercise weight increased because exercise was consistently present.")

    explanation = (
        " ".join(explanations)
        if explanations
        else "Weights stayed nearly unchanged because the log history did not contain enough variation yet."
    )

    return {
        "previous_weights": prev,
        "new_weights": new_weights,
        "explanation": explanation,
        "history": processed,
    }

