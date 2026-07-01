import numpy as np

from src.stochastic_model import (
    baseline_risk_to_health,
    compute_weekly_drift,
    effective_noise_sigma,
    risk_from_health,
    simulate_single_path,
)


def test_baseline_risk_to_health_is_clamped_and_monotonic():
    low = baseline_risk_to_health(0.1)
    high = baseline_risk_to_health(0.9)
    assert 0 <= high <= 100
    assert 0 <= low <= 100
    assert low > high


def test_week_zero_risk_matches_baseline_probability():
    baseline_probability = 0.23
    baseline_logit = np.log(baseline_probability / (1.0 - baseline_probability))
    initial_health = baseline_risk_to_health(baseline_probability)
    risk, _ = risk_from_health(initial_health, baseline_logit)
    assert np.isclose(risk, baseline_probability)


def test_drift_and_sigma_behave_reasonably():
    good_drift = compute_weekly_drift(
        sleep_mean_hours=8.0,
        exercise_days_per_week=5,
        stress_score=3,
        nutrition_score=8,
    )
    poor_drift = compute_weekly_drift(
        sleep_mean_hours=5.0,
        exercise_days_per_week=0,
        stress_score=9,
        nutrition_score=3,
    )
    assert good_drift > poor_drift
    assert -0.35 <= poor_drift <= good_drift <= 0.35

    low_sigma = effective_noise_sigma(sleep_variability_hours=0.2, stress_score=3)
    high_sigma = effective_noise_sigma(sleep_variability_hours=2.5, stress_score=9)
    assert high_sigma > low_sigma


def test_lifestyle_scenario_changes_are_directional_but_bounded():
    baseline = compute_weekly_drift(
        sleep_mean_hours=6.5,
        exercise_days_per_week=2,
        stress_score=7,
        nutrition_score=5,
    )
    improved = compute_weekly_drift(
        sleep_mean_hours=7.5,
        exercise_days_per_week=4,
        stress_score=5,
        nutrition_score=7,
    )
    assert improved > baseline
    assert improved - baseline < 0.5


def test_simulate_single_path_returns_bounded_arrays():
    profile = {
        "sleep_mean_hours": 7.0,
        "sleep_variability_hours": 0.5,
        "exercise_days_per_week": 3,
        "stress_score": 5,
        "nutrition_score": 6,
    }
    result = simulate_single_path(profile, baseline_logit=0.0, baseline_risk=0.25, horizon_years=1, seed=1)
    health = result["health"]
    risk = result["risk"]
    assert len(health) == 53
    assert len(risk) == 53
    assert np.all(health >= 0) and np.all(health <= 100)
    assert np.all(risk >= 0) and np.all(risk <= 1)
