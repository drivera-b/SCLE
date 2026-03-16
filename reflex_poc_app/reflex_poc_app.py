from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import reflex as rx

from src.baseline_model import predict_baseline_risk
from src.dataset import load_heart_dataset
from src.monte_carlo import run_monte_carlo
from src.plots import fan_chart, risk_histogram
from src.validate import validate_dashboard_inputs


DEMO_PROFILES: dict[str, dict[str, Any]] = {
    "Balanced Student": {
        "age": "18",
        "sex": "Female",
        "resting_hr": "68",
        "sleep_mean_hours": "7.6",
        "sleep_variability_hours": "0.7",
        "exercise_days_per_week": "4",
        "stress_score": "4",
        "nutrition_score": "7",
        "time_budget_minutes_per_day": "45",
        "horizon_years": "1",
        "simulation_count": "1200",
    },
    "High Stress Student": {
        "age": "18",
        "sex": "Female",
        "resting_hr": "84",
        "sleep_mean_hours": "5.8",
        "sleep_variability_hours": "1.7",
        "exercise_days_per_week": "1",
        "stress_score": "8",
        "nutrition_score": "4",
        "time_budget_minutes_per_day": "30",
        "horizon_years": "5",
        "simulation_count": "1500",
    },
    "Inconsistent Sleeper": {
        "age": "17",
        "sex": "Male",
        "resting_hr": "78",
        "sleep_mean_hours": "6.4",
        "sleep_variability_hours": "2.1",
        "exercise_days_per_week": "2",
        "stress_score": "6",
        "nutrition_score": "5",
        "time_budget_minutes_per_day": "35",
        "horizon_years": "1",
        "simulation_count": "1200",
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
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    fig.clf()
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


class DashboardState(rx.State):
    profile_choice: str = "Balanced Student"
    age: str = "18"
    sex: str = "Female"
    resting_hr: str = "68"
    sleep_mean_hours: str = "7.6"
    sleep_variability_hours: str = "0.7"
    exercise_days_per_week: str = "4"
    stress_score: str = "4"
    nutrition_score: str = "7"
    time_budget_minutes_per_day: str = "45"
    horizon_years: str = "1"
    simulation_count: str = "1200"

    baseline_risk: str = "—"
    stability_score: str = "—"
    improvement_potential: str = "—"
    status_text: str = ""
    model_summary_md: str = ""
    explanation_text: str = "Run a simulation to see your projected risk, uncertainty, and guidance."
    insights: list[str] = []
    error_text: str = ""
    risk_fan_src: str = ""
    risk_hist_src: str = ""

    def load_selected_profile(self) -> None:
        profile = DEMO_PROFILES.get(self.profile_choice, DEMO_PROFILES["Balanced Student"])
        self.age = profile["age"]
        self.sex = profile["sex"]
        self.resting_hr = profile["resting_hr"]
        self.sleep_mean_hours = profile["sleep_mean_hours"]
        self.sleep_variability_hours = profile["sleep_variability_hours"]
        self.exercise_days_per_week = profile["exercise_days_per_week"]
        self.stress_score = profile["stress_score"]
        self.nutrition_score = profile["nutrition_score"]
        self.time_budget_minutes_per_day = profile["time_budget_minutes_per_day"]
        self.horizon_years = profile["horizon_years"]
        self.simulation_count = profile["simulation_count"]
        self.status_text = f"Loaded {self.profile_choice}"

    def run_simulation(self) -> None:
        self.error_text = ""
        raw = {
            "age": self.age,
            "sex": self.sex,
            "resting_hr": self.resting_hr,
            "sleep_mean_hours": self.sleep_mean_hours,
            "sleep_variability_hours": self.sleep_variability_hours,
            "exercise_days_per_week": self.exercise_days_per_week,
            "stress_score": self.stress_score,
            "nutrition_score": self.nutrition_score,
            "time_budget_minutes_per_day": self.time_budget_minutes_per_day,
            "horizon_years": self.horizon_years,
            "simulation_count": self.simulation_count,
        }
        validation = validate_dashboard_inputs(raw)
        if not validation.ok:
            self.error_text = " | ".join(validation.errors[:4])
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
            self.baseline_risk = f"{baseline['probability']:.1%}"
            self.stability_score = f"{_stability_score(profile):.0f}/100"
            self.improvement_potential = f"{_improvement_potential(profile):.1%}"
            self.status_text = f"Complete • {profile['simulation_count']} paths • {profile['horizon_years']} year(s)"

            dataset_used = _dataset_source_label()
            model_type = "Trained baseline model" if baseline.get("source") == "trained_model" else "Heuristic fallback model"
            self.model_summary_md = (
                f"**Model Summary**\n\n"
                f"- Dataset used: `{dataset_used}`\n"
                f"- Model type: `{model_type}`\n"
                f"- Simulation horizon: `{profile['horizon_years']} year(s)`\n"
                f"- Monte Carlo runs: `{profile['simulation_count']}`"
            )

            uncertainty_range = float(simulation["risk_p95"][-1] - simulation["risk_p05"][-1])
            self.explanation_text = (
                f"Projected mean risk is {simulation['expected_mean_risk']:.1%}. "
                f"Uncertainty range at the horizon is about {uncertainty_range:.1%}. "
                "Focus on consistent sleep and stress control for the biggest near-term impact."
            )
            self.insights = [
                f"Chance of crossing risk threshold 0.60: {simulation['prob_exceed_threshold']:.1%}.",
                f"Final mean health state: {simulation['final_health_mean']:.1f}/100.",
                "Use demo profiles to compare lifestyle strategies quickly.",
            ]

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
            self.risk_fan_src = _fig_to_data_url(fan)
            self.risk_hist_src = _fig_to_data_url(hist)
        except Exception as exc:
            self.error_text = f"Simulation failed: {exc.__class__.__name__}"


def _metric_card(title: str, value_var: rx.Var) -> rx.Component:
    return rx.box(
        rx.text(title, font_size="0.85rem", color="#64748b"),
        rx.heading(value_var, size="6"),
        padding="0.8rem",
        border="1px solid rgba(15,118,110,0.18)",
        border_radius="14px",
        background_color="rgba(255,255,255,0.9)",
        width="100%",
        min_height="92px",
    )


def _input_field(label: str, value: rx.Var, on_change) -> rx.Component:
    return rx.vstack(
        rx.text(label, font_size="0.85rem", color="#334155"),
        rx.input(value=value, on_change=on_change, width="100%"),
        spacing="1",
        width="100%",
    )


def index() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.box(
                rx.heading("SLCE", size="8"),
                rx.text("Stochastic Lifestyle Control Engine", color="#334155"),
                rx.text("Educational tool only. Not medical advice or diagnosis.", color="#b91c1c"),
                border="1px solid rgba(15,118,110,0.2)",
                border_radius="16px",
                background_color="rgba(255,255,255,0.85)",
                padding="1rem",
                width="100%",
            ),
            rx.hstack(
                rx.select(
                    list(DEMO_PROFILES.keys()),
                    value=DashboardState.profile_choice,
                    on_change=DashboardState.set_profile_choice,
                    placeholder="Demo profile",
                    width="280px",
                ),
                rx.button("Load Profile", on_click=DashboardState.load_selected_profile, color_scheme="gray"),
                rx.button("Run Simulation", on_click=DashboardState.run_simulation, color_scheme="teal"),
                rx.text(DashboardState.status_text, color="#475569", font_size="0.85rem"),
                width="100%",
                align="end",
                spacing="3",
            ),
            rx.cond(
                DashboardState.error_text != "",
                rx.box(
                    DashboardState.error_text,
                    border_left="4px solid #b91c1c",
                    background_color="#fef2f2",
                    padding="0.6rem 0.8rem",
                    border_radius="8px",
                    width="100%",
                ),
            ),
            rx.heading("User Inputs", size="5"),
            rx.hstack(
                rx.vstack(
                    _input_field("Age", DashboardState.age, DashboardState.set_age),
                    _input_field("Sex", DashboardState.sex, DashboardState.set_sex),
                    _input_field("Resting heart rate", DashboardState.resting_hr, DashboardState.set_resting_hr),
                    _input_field("Sleep mean (hours/night)", DashboardState.sleep_mean_hours, DashboardState.set_sleep_mean_hours),
                    _input_field(
                        "Sleep variability (hours)",
                        DashboardState.sleep_variability_hours,
                        DashboardState.set_sleep_variability_hours,
                    ),
                    width="50%",
                ),
                rx.vstack(
                    _input_field(
                        "Exercise days/week",
                        DashboardState.exercise_days_per_week,
                        DashboardState.set_exercise_days_per_week,
                    ),
                    _input_field("Stress score (1-10)", DashboardState.stress_score, DashboardState.set_stress_score),
                    _input_field(
                        "Nutrition score (1-10)",
                        DashboardState.nutrition_score,
                        DashboardState.set_nutrition_score,
                    ),
                    _input_field(
                        "Time budget (minutes/day)",
                        DashboardState.time_budget_minutes_per_day,
                        DashboardState.set_time_budget_minutes_per_day,
                    ),
                    _input_field("Horizon years (1 or 5)", DashboardState.horizon_years, DashboardState.set_horizon_years),
                    _input_field(
                        "Monte Carlo paths",
                        DashboardState.simulation_count,
                        DashboardState.set_simulation_count,
                    ),
                    width="50%",
                ),
                width="100%",
                align="start",
                spacing="4",
            ),
            rx.heading("Simulation Results", size="5"),
            rx.hstack(
                _metric_card("Baseline Risk", DashboardState.baseline_risk),
                _metric_card("Stability Score", DashboardState.stability_score),
                _metric_card("Improvement Potential", DashboardState.improvement_potential),
                width="100%",
                spacing="3",
            ),
            rx.hstack(
                rx.cond(
                    DashboardState.risk_fan_src != "",
                    rx.image(src=DashboardState.risk_fan_src, width="100%"),
                    rx.box("Run simulation to render uncertainty fan chart.", padding="1rem"),
                ),
                rx.cond(
                    DashboardState.risk_hist_src != "",
                    rx.image(src=DashboardState.risk_hist_src, width="100%"),
                    rx.box("Run simulation to render risk histogram.", padding="1rem"),
                ),
                width="100%",
                spacing="4",
            ),
            rx.heading("Key Insights", size="5"),
            rx.foreach(
                DashboardState.insights,
                lambda line: rx.text(f"• {line}"),
            ),
            rx.heading("Model Explanation", size="5"),
            rx.box(
                DashboardState.explanation_text,
                border_left="4px solid #0f766e",
                background_color="#ecfdf5",
                border_radius="8px",
                padding="0.7rem 0.9rem",
                width="100%",
            ),
            rx.markdown(DashboardState.model_summary_md),
            spacing="4",
            width="100%",
            max_width="1200px",
            margin_x="auto",
            padding="1rem",
        ),
        background="linear-gradient(180deg, #f3efe8 0%, #fffdf8 100%)",
        min_height="100vh",
        width="100%",
    )


app = rx.App()
app.add_page(index, title="SLCE Reflex POC")

