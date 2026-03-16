from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .monte_carlo import run_monte_carlo
from .scoring import adherence_score, composite_objective


@dataclass
class OptimizationConstraints:
    max_minutes_per_day: int
    max_exercise_days_per_week: int
    max_sleep_increase_per_week: float
    max_stress_reduction_per_week: float
    nutrition_improvement_cap_per_week: float
    optimization_paths: int = 300


def estimate_time_cost_minutes_per_day(profile: dict[str, Any], plan: dict[str, Any]) -> float:
    sleep_gain_hours = max(0.0, float(plan["target_sleep_mean"]) - float(profile["sleep_mean_hours"]))
    exercise_gain_days = max(0.0, float(plan["target_exercise_days"]) - float(profile["exercise_days_per_week"]))
    stress_reduction = max(0.0, float(profile["stress_score"]) - float(plan["target_stress_score"]))
    nutrition_gain = max(0.0, float(plan["target_nutrition_score"]) - float(profile["nutrition_score"]))

    # Coarse expo-friendly estimate of daily effort.
    minutes = (
        12.0 * exercise_gain_days
        + 8.0 * stress_reduction
        + 6.0 * nutrition_gain
        + 10.0 * sleep_gain_hours
    )
    return float(minutes)


def _candidate_grid(current: float, deltas: list[float], lo: float, hi: float) -> list[float]:
    values = {round(min(hi, max(lo, current + d)), 2) for d in deltas}
    return sorted(values)


def generate_candidate_plans(
    profile: dict[str, Any],
    constraints: OptimizationConstraints,
) -> list[dict[str, Any]]:
    sleep_targets = _candidate_grid(
        float(profile["sleep_mean_hours"]),
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        3.0,
        10.5,
    )
    exercise_targets = _candidate_grid(
        float(profile["exercise_days_per_week"]),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        0.0,
        constraints.max_exercise_days_per_week,
    )
    stress_targets = _candidate_grid(
        float(profile["stress_score"]),
        [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        1.0,
        10.0,
    )
    nutrition_targets = _candidate_grid(
        float(profile["nutrition_score"]),
        [0.0, 1.0, 2.0, 3.0, 4.0],
        1.0,
        10.0,
    )
    variability_targets = _candidate_grid(
        float(profile["sleep_variability_hours"]),
        [0.0, -0.3, -0.6, -1.0],
        0.0,
        3.0,
    )

    candidates: list[dict[str, Any]] = []
    idx = 1
    for sleep_target in sleep_targets:
        for exercise_target in exercise_targets:
            for stress_target in stress_targets:
                for nutrition_target in nutrition_targets:
                    for variability_target in variability_targets:
                        plan = {
                            "id": f"plan_{idx}",
                            "name": f"Plan {idx}",
                            "target_sleep_mean": float(sleep_target),
                            "target_exercise_days": float(exercise_target),
                            "target_stress_score": float(stress_target),
                            "target_nutrition_score": float(nutrition_target),
                            "target_sleep_variability_hours": float(variability_target),
                            "max_sleep_increase_per_week": float(constraints.max_sleep_increase_per_week),
                            "max_stress_reduction_per_week": float(constraints.max_stress_reduction_per_week),
                            "nutrition_improvement_cap_per_week": float(
                                constraints.nutrition_improvement_cap_per_week
                            ),
                            "max_exercise_days_increase_per_week": 1.0,
                            "max_sleep_variability_change_per_week": 0.25,
                        }
                        time_cost = estimate_time_cost_minutes_per_day(profile, plan)
                        if time_cost > constraints.max_minutes_per_day + 1e-9:
                            continue
                        candidates.append(plan)
                        idx += 1

    # Remove duplicates that arise from clipping.
    deduped: dict[tuple[float, float, float, float, float], dict[str, Any]] = {}
    for plan in candidates:
        key = (
            plan["target_sleep_mean"],
            plan["target_exercise_days"],
            plan["target_stress_score"],
            plan["target_nutrition_score"],
            plan["target_sleep_variability_hours"],
        )
        deduped.setdefault(key, plan)
    return list(deduped.values())


def _change_magnitude(profile: dict[str, Any], plan: dict[str, Any]) -> float:
    return float(
        abs(plan["target_sleep_mean"] - float(profile["sleep_mean_hours"])) / 3.0
        + abs(plan["target_exercise_days"] - float(profile["exercise_days_per_week"])) / 7.0
        + abs(float(profile["stress_score"]) - plan["target_stress_score"]) / 9.0
        + abs(plan["target_nutrition_score"] - float(profile["nutrition_score"])) / 9.0
    )


def _pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            if (
                other["expected_mean_risk"] <= cand["expected_mean_risk"]
                and other["time_cost_minutes_per_day"] <= cand["time_cost_minutes_per_day"]
                and (
                    other["expected_mean_risk"] < cand["expected_mean_risk"]
                    or other["time_cost_minutes_per_day"] < cand["time_cost_minutes_per_day"]
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return sorted(front, key=lambda x: (x["expected_mean_risk"], x["time_cost_minutes_per_day"]))


def optimize_habit_plans(
    profile: dict[str, Any],
    baseline_output: dict[str, Any],
    constraints_dict: dict[str, Any],
    *,
    horizon_years: int,
    personalization_weights: dict[str, float] | None = None,
    seed: int = 123,
) -> dict[str, Any]:
    constraints = OptimizationConstraints(
        max_minutes_per_day=int(constraints_dict["max_minutes_per_day"]),
        max_exercise_days_per_week=int(constraints_dict["max_exercise_days_per_week"]),
        max_sleep_increase_per_week=float(constraints_dict["max_sleep_increase_per_week"]),
        max_stress_reduction_per_week=float(constraints_dict["max_stress_reduction_per_week"]),
        nutrition_improvement_cap_per_week=float(constraints_dict["nutrition_improvement_cap_per_week"]),
        optimization_paths=int(constraints_dict.get("optimization_paths", 300)),
    )

    candidates = generate_candidate_plans(profile, constraints)
    if not candidates:
        return {"top_plans": [], "all_candidates": [], "pareto": [], "candidate_count": 0}

    def heuristic_pre_score(plan: dict[str, Any]) -> float:
        time_cost = estimate_time_cost_minutes_per_day(profile, plan)
        # Cheap proxy before Monte Carlo to keep runtime bounded.
        proxy = (
            0.35 * max(0.0, float(profile["stress_score"]) - float(plan["target_stress_score"]))
            + 0.30 * max(0.0, float(plan["target_exercise_days"]) - float(profile["exercise_days_per_week"]))
            + 0.22 * max(0.0, float(plan["target_sleep_mean"]) - float(profile["sleep_mean_hours"]))
            + 0.15 * max(0.0, float(plan["target_nutrition_score"]) - float(profile["nutrition_score"]))
            - 0.12 * (time_cost / max(1.0, constraints.max_minutes_per_day))
            + 0.10 * _change_magnitude(profile, plan)
        )
        return proxy

    max_candidates = 90
    if len(candidates) > max_candidates:
        candidates = sorted(candidates, key=heuristic_pre_score, reverse=True)[:max_candidates]
    evaluated: list[dict[str, Any]] = []

    baseline_mc = run_monte_carlo(
        profile,
        baseline_output,
        plan=None,
        weights=personalization_weights,
        horizon_years=horizon_years,
        n_paths=constraints.optimization_paths,
        seed=seed,
        threshold=0.6,
    )
    baseline_expected_risk = float(baseline_mc["expected_mean_risk"])

    for plan in candidates:
        mc = run_monte_carlo(
            profile,
            baseline_output,
            plan=plan,
            weights=personalization_weights,
            horizon_years=horizon_years,
            n_paths=constraints.optimization_paths,
            # Use a stable seed across candidates for fair plan-to-plan comparison.
            seed=seed,
            threshold=0.6,
        )
        time_cost = estimate_time_cost_minutes_per_day(profile, plan)
        adherence = adherence_score(
            sleep_change=plan["target_sleep_mean"] - float(profile["sleep_mean_hours"]),
            exercise_change=plan["target_exercise_days"] - float(profile["exercise_days_per_week"]),
            stress_change=float(profile["stress_score"]) - plan["target_stress_score"],
            nutrition_change=plan["target_nutrition_score"] - float(profile["nutrition_score"]),
        )
        objective = composite_objective(
            expected_mean_risk=mc["expected_mean_risk"],
            time_cost_minutes_per_day=time_cost,
            adherence=adherence,
        )
        expected_risk_reduction = baseline_expected_risk - float(mc["expected_mean_risk"])
        change_mag = _change_magnitude(profile, plan)
        evaluated.append(
            {
                **plan,
                "expected_mean_risk": float(mc["expected_mean_risk"]),
                "expected_risk_reduction": float(expected_risk_reduction),
                "prob_exceed_threshold": float(mc["prob_exceed_threshold"]),
                "final_health_mean": float(mc["final_health_mean"]),
                "time_cost_minutes_per_day": float(time_cost),
                "adherence_score": float(adherence),
                "objective": float(objective),
                "change_magnitude": float(change_mag),
                "schedule_preview": mc["schedule"][:8],
            }
        )

    by_objective = sorted(
        evaluated,
        key=lambda x: (
            x["objective"],
            -x["expected_risk_reduction"],
            x["time_cost_minutes_per_day"],
            -x["adherence_score"],
        )
    )

    top3: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def _pick(plan: dict[str, Any] | None, label: str) -> None:
        if plan is None:
            return
        if plan["id"] in selected_ids:
            return
        plan["name"] = label
        top3.append(plan)
        selected_ids.add(plan["id"])

    # 1) Best objective plan.
    _pick(by_objective[0] if by_objective else None, "Balanced Momentum Strategy")

    # 2) Practical plan: lower time cost with decent risk reduction.
    practical_cap = 0.50 * constraints.max_minutes_per_day
    practical_pool = [p for p in evaluated if p["time_cost_minutes_per_day"] <= practical_cap]
    if practical_pool:
        practical = max(practical_pool, key=lambda p: (p["expected_risk_reduction"], -p["objective"]))
        _pick(practical, "Time-Smart Stability Strategy")

    # 3) Stretch plan: more aggressive change, near top risk reduction.
    stretch_floor = 0.70 * constraints.max_minutes_per_day
    stretch_pool = [
        p
        for p in evaluated
        if p["time_cost_minutes_per_day"] >= stretch_floor or p["change_magnitude"] >= 1.2
    ]
    if stretch_pool:
        stretch = max(stretch_pool, key=lambda p: (p["expected_risk_reduction"], -p["adherence_score"]))
        _pick(stretch, "Transformational Reset Strategy")

    for candidate in by_objective:
        if len(top3) >= 3:
            break
        _pick(candidate, f"Alternative Strategy {len(top3) + 1}")

    for rank, plan in enumerate(top3, start=1):
        plan["rank"] = rank
    pareto = _pareto_front(evaluated)

    return {
        "top_plans": top3,
        "all_candidates": by_objective,
        "pareto": pareto,
        "candidate_count": len(by_objective),
        "baseline_expected_risk": baseline_expected_risk,
    }
