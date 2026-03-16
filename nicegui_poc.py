from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
from nicegui import ui

from src.baseline_model import predict_baseline_risk
from src.dataset import load_heart_dataset
from src.monte_carlo import run_monte_carlo
from src.plots import fan_chart, risk_histogram
from src.validate import validate_dashboard_inputs


APP_TITLE = "SLCE"
APP_SUBTITLE = "Stochastic Lifestyle Control Engine"

DEMO_PROFILES: dict[str, dict[str, Any]] = {
    "Balanced Student": {
        "age": 18,
        "sex": "Female",
        "resting_hr": 68,
        "sleep_mean_hours": 7.6,
        "sleep_variability_hours": 0.7,
        "exercise_days_per_week": 4,
        "stress_score": 4,
        "nutrition_score": 7,
        "time_budget_minutes_per_day": 45,
        "horizon_years": 1,
        "simulation_count": 1200,
    },
    "High Stress Student": {
        "age": 18,
        "sex": "Female",
        "resting_hr": 84,
        "sleep_mean_hours": 5.8,
        "sleep_variability_hours": 1.7,
        "exercise_days_per_week": 1,
        "stress_score": 8,
        "nutrition_score": 4,
        "time_budget_minutes_per_day": 30,
        "horizon_years": 5,
        "simulation_count": 1500,
    },
    "Inconsistent Sleeper": {
        "age": 17,
        "sex": "Male",
        "resting_hr": 78,
        "sleep_mean_hours": 6.4,
        "sleep_variability_hours": 2.1,
        "exercise_days_per_week": 2,
        "stress_score": 6,
        "nutrition_score": 5,
        "time_budget_minutes_per_day": 35,
        "horizon_years": 1,
        "simulation_count": 1200,
    },
}


def _stability_score(profile: dict[str, Any]) -> float:
    sleep_consistency = max(0.0, 100.0 - (float(profile["sleep_variability_hours"]) / 3.0) * 100.0)
    stress_component = max(0.0, 100.0 - ((float(profile["stress_score"]) - 1.0) / 9.0) * 100.0)
    exercise_component = (float(profile["exercise_days_per_week"]) / 7.0) * 100.0
    return max(0.0, min(100.0, 0.42 * sleep_consistency + 0.33 * stress_component + 0.25 * exercise_component))


def _improvement_potential(profile: dict[str, Any]) -> float:
    sleep_gap = max(0.0, 7.8 - float(profile["sleep_mean_hours"])) + 0.6 * max(
        0.0, float(profile["sleep_variability_hours"]) - 0.8
    )
    stress_gap = max(0.0, float(profile["stress_score"]) - 4.0)
    exercise_gap = max(0.0, 5.0 - float(profile["exercise_days_per_week"]))
    nutrition_gap = max(0.0, 8.0 - float(profile["nutrition_score"]))
    estimate = 0.02 + 0.012 * (sleep_gap + stress_gap + exercise_gap + nutrition_gap)
    return max(0.0, min(0.35, estimate))


def _fig_to_data_url(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{data}"


def _dataset_source_label() -> str:
    try:
        _, info = load_heart_dataset(try_download=False, allow_demo_fallback=True)
        source = str(info.get("source", "unknown"))
        if source == "local_heart_csv":
            return "data/heart.csv"
        if source == "uci_download":
            return "UCI Heart Disease (cached)"
        if source == "demo_sample":
            return "demo_sample.csv (fallback)"
        return source
    except Exception:
        return "demo_sample.csv (fallback)"


ui.add_head_html(
    """
    <style>
      body { background: linear-gradient(180deg,#f3efe8 0%,#fffdf8 100%); }
      .shell { max-width: 1200px; margin: 0 auto; }
      .hero { background: rgba(255,255,255,0.80); border: 1px solid rgba(15,118,110,0.18); border-radius: 16px; padding: 16px; }
      .section-title { font-size: 1.1rem; font-weight: 700; margin-top: 8px; margin-bottom: 8px; }
      .explain-box { border-left: 4px solid #0f766e; background: #ecfdf5; border-radius: 8px; padding: 10px 12px; }
      .metric-card { background: rgba(255,255,255,0.9); border: 1px solid rgba(15,118,110,0.16); border-radius: 14px; padding: 12px; min-height: 96px; }
      .metric-title { font-size: 0.85rem; color: #475569; }
      .metric-value { font-size: 1.55rem; font-weight: 700; }
    </style>
    """
)

with ui.column().classes("shell q-pa-md gap-4"):
    with ui.card().classes("hero w-full"):
        ui.label(APP_TITLE).classes("text-h4 text-weight-bold")
        ui.label(APP_SUBTITLE).classes("text-subtitle1")
        ui.label("Educational tool only. Not medical advice or diagnosis.").classes("text-negative text-weight-medium")

    with ui.row().classes("items-end w-full gap-4"):
        profile_select = ui.select(
            options=list(DEMO_PROFILES.keys()),
            value="Balanced Student",
            label="Demo profile",
        ).classes("w-64")
        load_btn = ui.button("Load Profile", color="secondary")
        run_btn = ui.button("Run Simulation", color="primary")
        status_badge = ui.label("").classes("text-caption text-grey-8")

    ui.label("User Inputs").classes("section-title")
    with ui.row().classes("w-full gap-6"):
        with ui.column().classes("w-1/2 gap-3"):
            age = ui.number("Age", value=18, min=10, max=100, step=1)
            sex = ui.select(["Female", "Male", "Other", "Unknown"], value="Female", label="Sex")
            resting_hr = ui.number("Resting heart rate", value=68, min=40, max=120, step=1)
            sleep_mean = ui.number("Sleep mean (hours/night)", value=7.6, min=3.0, max=12.0, step=0.1)
            sleep_var = ui.number("Sleep variability (hours)", value=0.7, min=0.0, max=3.0, step=0.1)
        with ui.column().classes("w-1/2 gap-3"):
            exercise_days = ui.number("Exercise days/week", value=4, min=0, max=7, step=1)
            stress = ui.number("Stress score (1-10)", value=4, min=1, max=10, step=1)
            nutrition = ui.number("Nutrition score (1-10)", value=7, min=1, max=10, step=1)
            time_budget = ui.number("Time budget (minutes/day)", value=45, min=0, max=180, step=5)
            horizon = ui.select({1: "1 year", 5: "5 years"}, value=1, label="Horizon")
            sim_count = ui.number("Monte Carlo paths", value=1200, min=500, max=5000, step=100)

    ui.label("Simulation Results").classes("section-title")
    with ui.row().classes("w-full gap-3"):
        with ui.card().classes("metric-card col"):
            ui.label("Baseline Risk").classes("metric-title")
            baseline_val = ui.label("—").classes("metric-value")
        with ui.card().classes("metric-card col"):
            ui.label("Stability Score").classes("metric-title")
            stability_val = ui.label("—").classes("metric-value")
        with ui.card().classes("metric-card col"):
            ui.label("Improvement Potential").classes("metric-title")
            improve_val = ui.label("—").classes("metric-value")

    with ui.row().classes("w-full gap-4"):
        risk_fan_img = ui.image().classes("w-1/2 rounded-borders")
        risk_hist_img = ui.image().classes("w-1/2 rounded-borders")

    ui.label("Model Explanation").classes("section-title")
    explain_box = ui.html("<div class='explain-box'>Run a simulation to see plain-English interpretation.</div>").classes("w-full")
    model_summary = ui.markdown("")


def _current_profile() -> dict[str, Any]:
    return {
        "age": age.value,
        "sex": sex.value,
        "resting_hr": resting_hr.value,
        "sleep_mean_hours": sleep_mean.value,
        "sleep_variability_hours": sleep_var.value,
        "exercise_days_per_week": exercise_days.value,
        "stress_score": stress.value,
        "nutrition_score": nutrition.value,
        "time_budget_minutes_per_day": time_budget.value,
        "horizon_years": horizon.value,
        "simulation_count": sim_count.value,
    }


def _load_profile() -> None:
    profile = DEMO_PROFILES[profile_select.value]
    age.value = profile["age"]
    sex.value = profile["sex"]
    resting_hr.value = profile["resting_hr"]
    sleep_mean.value = profile["sleep_mean_hours"]
    sleep_var.value = profile["sleep_variability_hours"]
    exercise_days.value = profile["exercise_days_per_week"]
    stress.value = profile["stress_score"]
    nutrition.value = profile["nutrition_score"]
    time_budget.value = profile["time_budget_minutes_per_day"]
    horizon.value = profile["horizon_years"]
    sim_count.value = profile["simulation_count"]
    ui.notify(f"Loaded {profile_select.value}", type="positive")


def _run_simulation() -> None:
    raw = _current_profile()
    validation = validate_dashboard_inputs(raw)
    if not validation.ok:
        for err in validation.errors:
            ui.notify(err, type="negative")
        return

    try:
        profile = validation.values
        baseline = predict_baseline_risk(profile)
        simulation = run_monte_carlo(
            profile,
            baseline,
            horizon_years=int(profile["horizon_years"]),
            n_paths=int(profile["simulation_count"]),
            threshold=0.6,
            seed=42,
        )

        baseline_val.text = f"{baseline['probability']:.1%}"
        stability_val.text = f"{_stability_score(profile):.0f}/100"
        improve_val.text = f"{_improvement_potential(profile):.1%}"
        status_badge.text = f"Completed • {profile['simulation_count']} paths • {profile['horizon_years']} year(s)"

        fan = fan_chart(
            simulation["time_axis_weeks"],
            simulation["risk_median"],
            simulation["risk_p05"],
            simulation["risk_p95"],
            title="Uncertainty Fan Chart",
            y_label="Risk probability",
            line_color="#0f766e",
            fill_color="#99f6e4",
        )
        hist = risk_histogram(simulation["final_risk"])
        risk_fan_img.source = _fig_to_data_url(fan)
        risk_hist_img.source = _fig_to_data_url(hist)

        explanation = (
            f"Expected mean risk is {simulation['expected_mean_risk']:.1%}. "
            f"Chance of crossing the 0.60 threshold is {simulation['prob_exceed_threshold']:.1%}. "
            f"The biggest improvement lever is usually stress/sleep consistency for this profile."
        )
        explain_box.content = f"<div class='explain-box'>{explanation}</div>"
        model_summary.content = (
            f"**Model Summary**  \n"
            f"- Dataset used: `{_dataset_source_label()}`  \n"
            f"- Baseline model: `{'Trained model' if baseline.get('source') == 'trained_model' else 'Heuristic fallback'}`  \n"
            f"- Simulation horizon: `{profile['horizon_years']} year(s)`  \n"
            f"- Monte Carlo runs: `{profile['simulation_count']}`"
        )
    except Exception as exc:
        ui.notify(
            f"Simulation failed: {exc.__class__.__name__}. Please check inputs and try again.",
            type="negative",
        )


load_btn.on_click(_load_profile)
run_btn.on_click(_run_simulation)

ui.run(title=f"{APP_TITLE} NiceGUI POC")

