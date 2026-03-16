from src.validate import validate_dashboard_inputs, validate_weekly_log


def test_validate_dashboard_inputs_accepts_valid_profile():
    result = validate_dashboard_inputs(
        {
            "age": 18,
            "sex": "Female",
            "resting_hr": 70,
            "sleep_mean_hours": 7.5,
            "sleep_variability_hours": 0.8,
            "exercise_days_per_week": 3,
            "stress_score": 5,
            "nutrition_score": 7,
            "time_budget_minutes_per_day": 45,
            "horizon_years": 1,
            "simulation_count": 1000,
        }
    )
    assert result.ok
    assert result.values["age"] == 18
    assert result.values["horizon_years"] == 1


def test_validate_dashboard_inputs_rejects_out_of_range():
    result = validate_dashboard_inputs(
        {
            "age": 200,
            "sex": "Alien",
            "resting_hr": 20,
            "sleep_mean_hours": 15,
            "sleep_variability_hours": 5,
            "exercise_days_per_week": 9,
            "stress_score": 20,
            "nutrition_score": 0,
            "time_budget_minutes_per_day": 999,
            "horizon_years": 3,
            "simulation_count": 20,
        }
    )
    assert not result.ok
    assert any("age" in e for e in result.errors)
    assert any("sex" in e for e in result.errors)
    assert any("horizon_years" in e for e in result.errors)


def test_validate_weekly_log_requires_seven_days():
    result = validate_weekly_log(
        sleep_hours=[7] * 6,
        stress_scores=[5] * 7,
        exercise_minutes=[20] * 7,
    )
    assert not result.ok
    assert any("exactly 7 entries" in e for e in result.errors)

