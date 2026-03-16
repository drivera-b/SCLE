from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    values: dict[str, Any]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _to_float(value: Any, field_label: str, errors: list[str]) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{field_label} must be a number.")
        return None


def _to_int(value: Any, field_label: str, errors: list[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{field_label} must be an integer.")
        return None


def _validate_range(
    raw: Any,
    name: str,
    min_value: float,
    max_value: float,
    *,
    as_int: bool = False,
) -> tuple[float | int | None, list[str]]:
    errors: list[str] = []
    parser = _to_int if as_int else _to_float
    value = parser(raw, name, errors)
    if value is None:
        return None, errors
    if value < min_value or value > max_value:
        errors.append(f"{name} must be between {min_value:g} and {max_value:g}.")
        return None, errors
    return value, errors


def validate_dashboard_inputs(raw: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    values: dict[str, Any] = {}

    fields: list[tuple[str, float, float, bool]] = [
        ("age", 10, 100, True),
        ("resting_hr", 40, 120, True),
        ("sleep_mean_hours", 3, 12, False),
        ("sleep_variability_hours", 0, 3, False),
        ("exercise_days_per_week", 0, 7, True),
        ("stress_score", 1, 10, True),
        ("nutrition_score", 1, 10, True),
        ("time_budget_minutes_per_day", 0, 180, True),
        ("simulation_count", 500, 5000, True),
    ]
    for name, min_v, max_v, as_int in fields:
        value, field_errors = _validate_range(raw.get(name), name, min_v, max_v, as_int=as_int)
        errors.extend(field_errors)
        if value is not None:
            values[name] = value

    sex_raw = raw.get("sex", "Unknown")
    if sex_raw in ("Female", "Male", "Unknown", "Other"):
        values["sex"] = sex_raw
    elif isinstance(sex_raw, (int, float)) and int(sex_raw) in (0, 1):
        values["sex"] = "Male" if int(sex_raw) == 1 else "Female"
    else:
        errors.append("sex must be Female, Male, Other, or Unknown.")

    horizon_raw = raw.get("horizon_years", 1)
    if str(horizon_raw) in {"1", "5"}:
        values["horizon_years"] = int(horizon_raw)
    elif horizon_raw in (1, 5):
        values["horizon_years"] = int(horizon_raw)
    else:
        errors.append("horizon_years must be 1 or 5.")

    return ValidationResult(values=values, errors=errors)


def validate_optimizer_constraints(raw: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    values: dict[str, Any] = {}
    fields: list[tuple[str, float, float, bool]] = [
        ("max_minutes_per_day", 0, 180, True),
        ("max_exercise_days_per_week", 0, 7, True),
        ("max_sleep_increase_per_week", 0.0, 2.0, False),
        ("max_stress_reduction_per_week", 0.0, 3.0, False),
        ("nutrition_improvement_cap_per_week", 0.0, 3.0, False),
        ("optimization_paths", 100, 1000, True),
    ]
    for name, min_v, max_v, as_int in fields:
        value, field_errors = _validate_range(raw.get(name), name, min_v, max_v, as_int=as_int)
        errors.extend(field_errors)
        if value is not None:
            values[name] = value
    return ValidationResult(values=values, errors=errors)


def validate_weekly_log(
    sleep_hours: list[Any],
    stress_scores: list[Any],
    exercise_minutes: list[Any],
    nutrition_scores: list[Any] | None = None,
) -> ValidationResult:
    errors: list[str] = []
    values: dict[str, Any] = {}

    for name, series, min_v, max_v in [
        ("sleep_hours", sleep_hours, 0.0, 16.0),
        ("stress_scores", stress_scores, 1.0, 10.0),
        ("exercise_minutes", exercise_minutes, 0.0, 300.0),
    ]:
        if len(series) != 7:
            errors.append(f"{name} must contain exactly 7 entries (one for each day).")
            continue
        parsed: list[float] = []
        for idx, item in enumerate(series, start=1):
            value = _to_float(item, f"{name}[day {idx}]", errors)
            if value is None:
                continue
            if value < min_v or value > max_v:
                errors.append(f"{name}[day {idx}] must be between {min_v:g} and {max_v:g}.")
                continue
            parsed.append(value)
        if len(parsed) == 7:
            values[name] = parsed

    if nutrition_scores is not None:
        if len(nutrition_scores) != 7:
            errors.append("nutrition_scores must contain exactly 7 entries.")
        else:
            parsed_nutrition: list[float] = []
            for idx, item in enumerate(nutrition_scores, start=1):
                value = _to_float(item, f"nutrition_scores[day {idx}]", errors)
                if value is None:
                    continue
                if value < 1 or value > 10:
                    errors.append(f"nutrition_scores[day {idx}] must be between 1 and 10.")
                    continue
                parsed_nutrition.append(value)
            if len(parsed_nutrition) == 7:
                values["nutrition_scores"] = parsed_nutrition

    return ValidationResult(values=values, errors=errors)


def sex_to_numeric(sex: str | int | float) -> int:
    if isinstance(sex, (int, float)):
        return 1 if int(sex) == 1 else 0
    sex_norm = str(sex).strip().lower()
    if sex_norm in {"m", "male"}:
        return 1
    return 0

