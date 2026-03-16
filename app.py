from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.baseline_model import load_baseline_artifacts, predict_baseline_risk
from src.dataset import dataset_status_message, load_heart_dataset
from src.monte_carlo import run_monte_carlo
from src.optimizer import optimize_habit_plans
from src.personalization import default_weights, update_personalization_weights
from src.plots import fan_chart, risk_histogram, tradeoff_scatter
from src.validate import validate_dashboard_inputs, validate_optimizer_constraints, validate_weekly_log


APP_TITLE = "SLCE"
DISCLAIMER_TEXT = "Educational tool only. Not medical advice or diagnosis."
EXPORTS_DIRNAME = "exports"
EXPERIMENTS_DIRNAME = "experiments"

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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --vita-bg-1: #f5f1ea;
          --vita-bg-2: #fffdf8;
          --vita-ink: #16212f;
          --vita-accent: #0f766e;
          --vita-accent-2: #b45309;
        }
        .stApp {
          background:
            radial-gradient(1000px 400px at 100% -10%, rgba(15,118,110,0.10), transparent 60%),
            radial-gradient(800px 300px at 0% 0%, rgba(180,83,9,0.09), transparent 60%),
            linear-gradient(180deg, var(--vita-bg-1), var(--vita-bg-2));
          color: var(--vita-ink);
          font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        }
        h1, h2, h3 {
          font-family: Georgia, "Times New Roman", serif;
          letter-spacing: 0.2px;
        }
        .vita-banner {
          border: 1px solid rgba(15,118,110,0.22);
          background: rgba(255,255,255,0.72);
          border-radius: 14px;
          padding: 0.7rem 0.9rem;
          margin-bottom: 0.75rem;
        }
        .vita-shell-note {
          border: 1px solid rgba(22,33,47,0.08);
          background: rgba(255,255,255,0.70);
          border-radius: 12px;
          padding: 0.65rem 0.8rem;
          margin: 0.2rem 0 0.8rem 0;
        }
        .vita-disclaimer {
          border-left: 4px solid #b91c1c;
          background: rgba(254,242,242,0.9);
          padding: 0.6rem 0.8rem;
          border-radius: 8px;
          margin-bottom: 0.8rem;
          font-size: 0.95rem;
        }
        .vita-step {
          margin-top: 0.8rem;
          margin-bottom: 0.3rem;
          padding: 0.7rem 0.85rem;
          border-radius: 12px;
          border: 1px solid rgba(15,118,110,0.16);
          background: rgba(255,255,255,0.72);
        }
        .vita-step-title {
          font-family: Georgia, "Times New Roman", serif;
          font-size: 1.02rem;
          font-weight: 700;
          color: #0f172a;
          margin: 0;
        }
        .vita-step-sub {
          color: #475569;
          font-size: 0.88rem;
          margin-top: 0.15rem;
        }
        .vita-card {
          border: 1px solid rgba(22,33,47,0.08);
          background: rgba(255,255,255,0.78);
          border-radius: 14px;
          padding: 0.9rem;
          margin: 0.35rem 0;
        }
        .vita-card h4 {
          margin: 0 0 0.3rem 0;
          font-size: 1.0rem;
        }
        .vita-muted {
          color: #64748b;
          font-size: 0.88rem;
        }
        .vita-pill {
          display: inline-block;
          padding: 0.2rem 0.5rem;
          border-radius: 999px;
          background: rgba(15,118,110,0.1);
          color: #0f766e;
          border: 1px solid rgba(15,118,110,0.18);
          font-size: 0.78rem;
          margin-right: 0.35rem;
        }
        .vita-callout {
          border-left: 4px solid #0f766e;
          background: rgba(240,253,250,0.9);
          border-radius: 8px;
          padding: 0.65rem 0.8rem;
          margin: 0.4rem 0 0.6rem 0;
        }
        .vita-warning {
          border-left-color: #b45309;
          background: rgba(255,251,235,0.9);
        }
        div[data-testid="stMetric"] {
          background: rgba(255,255,255,0.80);
          border: 1px solid rgba(15,118,110,0.14);
          border-radius: 14px;
          padding: 0.65rem 0.75rem;
        }
        .stButton > button {
          border-radius: 12px;
          border: 1px solid rgba(15,118,110,0.2);
          font-weight: 600;
          padding-top: 0.5rem;
          padding-bottom: 0.5rem;
        }
        .stButton > button:hover {
          border-color: rgba(15,118,110,0.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_session_state() -> None:
    if "personalization_weights" not in st.session_state:
        st.session_state.personalization_weights = default_weights()
    if "log_history" not in st.session_state:
        st.session_state.log_history = []
    if "dashboard_result" not in st.session_state:
        st.session_state.dashboard_result = None
    if "optimization_result" not in st.session_state:
        st.session_state.optimization_result = None
    if "busy_dashboard" not in st.session_state:
        st.session_state.busy_dashboard = False
    if "busy_optimizer" not in st.session_state:
        st.session_state.busy_optimizer = False
    if "busy_weekly_log" not in st.session_state:
        st.session_state.busy_weekly_log = False
    if "active_demo_profile" not in st.session_state:
        st.session_state.active_demo_profile = "Balanced Student"
    if "ui_mode" not in st.session_state:
        st.session_state.ui_mode = "App Mode"
    if "demo_day_mode" not in st.session_state:
        st.session_state.demo_day_mode = False
    if "demo_day_initialized" not in st.session_state:
        st.session_state.demo_day_initialized = False
    if "last_export_path" not in st.session_state:
        st.session_state.last_export_path = None
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Dashboard"
    if "explain_mode" not in st.session_state:
        st.session_state.explain_mode = True
    if "compare_result" not in st.session_state:
        st.session_state.compare_result = None
    _load_profile_into_state_if_missing(DEMO_PROFILES["Balanced Student"])


def _profile_keys() -> list[str]:
    return [
        "age",
        "sex",
        "resting_hr",
        "sleep_mean_hours",
        "sleep_variability_hours",
        "exercise_days_per_week",
        "stress_score",
        "nutrition_score",
        "time_budget_minutes_per_day",
        "horizon_years",
        "simulation_count",
    ]


def _load_profile_into_state_if_missing(profile: dict[str, Any]) -> None:
    for key in _profile_keys():
        if f"profile_{key}" not in st.session_state:
            st.session_state[f"profile_{key}"] = profile[key]
    if "profile_horizon_label" not in st.session_state:
        st.session_state.profile_horizon_label = "1 year" if int(profile["horizon_years"]) == 1 else "5 years"


def _apply_demo_profile(name: str) -> None:
    profile = DEMO_PROFILES[name]
    for key in _profile_keys():
        st.session_state[f"profile_{key}"] = profile[key]
    st.session_state.profile_horizon_label = "1 year" if int(profile["horizon_years"]) == 1 else "5 years"
    st.session_state.active_demo_profile = name


def _current_profile_from_state() -> dict[str, Any]:
    return {key: st.session_state.get(f"profile_{key}") for key in _profile_keys()}


def _render_header() -> None:
    st.markdown(
        f"<div class='vita-banner'><h2 style='margin:0'>{APP_TITLE}</h2><div>Stochastic Lifestyle Control Engine</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='vita-disclaimer'><strong>Safety:</strong> {DISCLAIMER_TEXT}</div>", unsafe_allow_html=True)


def _is_research_mode() -> bool:
    return st.session_state.get("ui_mode", "App Mode") == "Research Mode"


def _section_step(step_no: int, title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<div class='vita-step-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='vita-step'><div class='vita-step-title'>Step {step_no}: {title}</div>{subtitle_html}</div>",
        unsafe_allow_html=True,
    )


def _consumer_callout(text: str, *, warning: bool = False) -> None:
    klass = "vita-callout vita-warning" if warning else "vita-callout"
    st.markdown(f"<div class='{klass}'>{text}</div>", unsafe_allow_html=True)


def _render_global_toggles() -> None:
    c1, c2, c3 = st.columns([2.0, 1.2, 1.2])
    with c1:
        st.radio(
            "View",
            options=["App Mode", "Research Mode"],
            key="ui_mode",
            horizontal=True,
            help="App Mode is judge-friendly and guided. Research Mode reveals technical assumptions and parameters.",
        )
    with c2:
        st.toggle(
            "Demo Day Mode",
            key="demo_day_mode",
            help="Loads a default profile and auto-runs a dashboard simulation once for a fast expo start.",
        )
    with c3:
        st.toggle(
            "Explain Mode",
            key="explain_mode",
            help="Shows short plain-English explanations next to charts and optimization outputs.",
        )

    mode_text = (
        "App Mode: guided workflow, plain-English explanations, action-oriented recommendations."
        if not _is_research_mode()
        else "Research Mode: expanders include equations, parameters, assumptions, and model details."
    )
    st.markdown(f"<div class='vita-shell-note'>{mode_text}</div>", unsafe_allow_html=True)


def _dashboard_profile_signature(profile: dict[str, Any]) -> tuple[Any, ...]:
    keys = [
        "age",
        "sex",
        "resting_hr",
        "sleep_mean_hours",
        "sleep_variability_hours",
        "exercise_days_per_week",
        "stress_score",
        "nutrition_score",
        "horizon_years",
    ]
    return tuple(profile.get(k) for k in keys)


def _run_dashboard_simulation(profile: dict[str, Any], *, seed: int = 42) -> dict[str, Any]:
    profile = _clamp_profile_values(profile)
    baseline = predict_baseline_risk(profile)
    simulation = run_monte_carlo(
        profile,
        baseline,
        horizon_years=int(profile["horizon_years"]),
        n_paths=int(profile["simulation_count"]),
        weights=st.session_state.personalization_weights,
        seed=seed,
        threshold=0.6,
    )
    result = {
        "profile": profile,
        "baseline": baseline,
        "simulation": simulation,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.current_profile = profile
    st.session_state.dashboard_result = result
    _append_experiment_log(profile, baseline, simulation)
    return result


def _maybe_run_demo_day_bootstrap() -> None:
    if st.session_state.demo_day_mode and not st.session_state.demo_day_initialized:
        _apply_demo_profile("High Stress Student")
        st.session_state.nav_page = "Dashboard"
        validation = validate_dashboard_inputs(_current_profile_from_state())
        if validation.ok:
            try:
                with st.spinner("Demo Day Mode: loading profile and running simulation..."):
                    _run_dashboard_simulation(validation.values, seed=7)
                st.session_state.demo_day_initialized = True
                st.session_state.demo_day_message = "Demo Day Mode loaded the High Stress Student profile and pre-ran a simulation."
            except Exception as exc:
                st.session_state.demo_day_message = (
                    "Demo Day Mode could not auto-run the simulation "
                    f"({exc.__class__.__name__}). You can still run it manually."
                )
                st.session_state.demo_day_initialized = True
    elif not st.session_state.demo_day_mode and st.session_state.demo_day_initialized:
        st.session_state.demo_day_initialized = False


def _render_demo_day_notice() -> None:
    message = st.session_state.get("demo_day_message")
    if message and st.session_state.get("demo_day_mode"):
        st.success(message)


@st.cache_data(show_spinner=False)
def _cached_dataset_preview() -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    try:
        df, info = load_heart_dataset(try_download=False, allow_demo_fallback=True)
        return df.head(5), info
    except Exception:
        return None, None


def _render_project_status() -> None:
    model_obj, metadata = load_baseline_artifacts()
    preview_df, preview_info = _cached_dataset_preview()
    with st.expander("Model & Dataset Status", expanded=False):
        st.write(dataset_status_message())
        st.write(f"Baseline model artifact loaded: {'Yes' if model_obj is not None else 'No (app will auto-fallback)'}")
        if metadata:
            metrics = metadata.get("metrics", {})
            st.write(
                {
                    "trained_at_utc": metadata.get("trained_at_utc"),
                    "accuracy": metrics.get("accuracy"),
                    "roc_auc": metrics.get("roc_auc"),
                    "dataset_source": metadata.get("dataset_source", {}).get("source"),
                }
            )
        if preview_info:
            st.caption(f"Preview source: {preview_info.get('source')}")
        if preview_df is not None:
            st.dataframe(preview_df, use_container_width=True, hide_index=True)


def _render_demo_profile_picker() -> None:
    c1, c2 = st.columns([3, 1])
    selected = c1.selectbox(
        "Demo profile",
        options=list(DEMO_PROFILES.keys()),
        index=list(DEMO_PROFILES.keys()).index(st.session_state.active_demo_profile),
        help="Prebuilt profiles for reliable STEM Expo demos.",
    )
    if c2.button("Load Profile", use_container_width=True, type="secondary"):
        _apply_demo_profile(selected)
        st.rerun()


def _render_shared_inputs() -> dict[str, Any]:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            step=1,
            key="profile_age",
            help="Age is used by the baseline ML model and risk initialization.",
        )
        sex = st.selectbox(
            "Sex",
            options=["Female", "Male", "Other", "Unknown"],
            key="profile_sex",
            help="Optional for personalization; the baseline model only uses a simplified binary mapping when needed.",
        )
        resting_hr = st.number_input(
            "Resting heart rate (bpm)",
            min_value=40,
            max_value=120,
            step=1,
            key="profile_resting_hr",
            help="Resting HR affects the stochastic uncertainty layer and a proxy baseline feature estimate.",
        )
        sleep_mean = st.slider(
            "Sleep mean (hours/night)",
            min_value=3.0,
            max_value=12.0,
            step=0.1,
            key="profile_sleep_mean_hours",
            help="Average nightly sleep duration.",
        )
        sleep_var = st.slider(
            "Sleep variability (hours)",
            min_value=0.0,
            max_value=3.0,
            step=0.1,
            key="profile_sleep_variability_hours",
            help="Higher variability increases simulation uncertainty.",
        )
    with col2:
        exercise_days = st.slider(
            "Exercise days per week",
            min_value=0,
            max_value=7,
            step=1,
            key="profile_exercise_days_per_week",
            help="Number of active days per week. Used in drift and optimization.",
        )
        stress = st.slider(
            "Stress score (1-10)",
            min_value=1,
            max_value=10,
            step=1,
            key="profile_stress_score",
            help="Self-rated stress level. Higher values increase downward drift and uncertainty.",
        )
        nutrition = st.slider(
            "Nutrition score (1-10)",
            min_value=1,
            max_value=10,
            step=1,
            key="profile_nutrition_score",
            help="Higher values improve weekly drift in the health-state model.",
        )
        time_budget = st.slider(
            "Time budget (minutes/day)",
            min_value=0,
            max_value=180,
            step=5,
            key="profile_time_budget_minutes_per_day",
            help="Daily time available for lifestyle changes, used by the optimizer.",
        )
        horizon_label = st.selectbox(
            "Horizon",
            options=["1 year", "5 years"],
            key="profile_horizon_label",
            help="Simulation horizon in weekly timesteps.",
        )
        # Keep canonical integer in session for validators and other pages.
        st.session_state.profile_horizon_years = 1 if horizon_label == "1 year" else 5
        sim_count = st.slider(
            "Monte Carlo paths (N)",
            min_value=500,
            max_value=5000,
            step=100,
            key="profile_simulation_count",
            help="More paths improve stability but increase runtime.",
        )

    return {
        "age": age,
        "sex": sex,
        "resting_hr": resting_hr,
        "sleep_mean_hours": sleep_mean,
        "sleep_variability_hours": sleep_var,
        "exercise_days_per_week": exercise_days,
        "stress_score": stress,
        "nutrition_score": nutrition,
        "time_budget_minutes_per_day": time_budget,
        "horizon_years": st.session_state.profile_horizon_years,
        "simulation_count": sim_count,
    }


def _clamp_profile_values(profile: dict[str, Any]) -> dict[str, Any]:
    clamped = dict(profile)
    clamped["age"] = int(max(10, min(100, int(clamped["age"]))))
    clamped["resting_hr"] = int(max(40, min(120, int(clamped["resting_hr"]))))
    clamped["sleep_mean_hours"] = float(max(3.0, min(12.0, float(clamped["sleep_mean_hours"]))))
    clamped["sleep_variability_hours"] = float(max(0.0, min(3.0, float(clamped["sleep_variability_hours"]))))
    clamped["exercise_days_per_week"] = int(max(0, min(7, int(clamped["exercise_days_per_week"]))))
    clamped["stress_score"] = int(max(1, min(10, int(clamped["stress_score"]))))
    clamped["nutrition_score"] = int(max(1, min(10, int(clamped["nutrition_score"]))))
    clamped["time_budget_minutes_per_day"] = int(max(0, min(180, int(clamped["time_budget_minutes_per_day"]))))
    clamped["simulation_count"] = int(max(500, min(5000, int(clamped["simulation_count"]))))
    clamped["horizon_years"] = 5 if int(clamped["horizon_years"]) == 5 else 1
    return clamped


def _clamp_optimizer_constraints(values: dict[str, Any]) -> dict[str, Any]:
    clamped = dict(values)
    clamped["max_minutes_per_day"] = int(max(0, min(180, int(clamped["max_minutes_per_day"]))))
    clamped["max_exercise_days_per_week"] = int(max(0, min(7, int(clamped["max_exercise_days_per_week"]))))
    clamped["max_sleep_increase_per_week"] = float(max(0.0, min(2.0, float(clamped["max_sleep_increase_per_week"]))))
    clamped["max_stress_reduction_per_week"] = float(max(0.0, min(3.0, float(clamped["max_stress_reduction_per_week"]))))
    clamped["nutrition_improvement_cap_per_week"] = float(
        max(0.0, min(3.0, float(clamped["nutrition_improvement_cap_per_week"])))
    )
    clamped["optimization_paths"] = int(max(100, min(1000, int(clamped["optimization_paths"]))))
    return clamped


def _system_stability_score(profile: dict[str, Any]) -> float:
    sleep_consistency = max(0.0, 100.0 - (float(profile["sleep_variability_hours"]) / 3.0) * 100.0)
    stress_component = max(0.0, 100.0 - ((float(profile["stress_score"]) - 1.0) / 9.0) * 100.0)
    exercise_component = (float(profile["exercise_days_per_week"]) / 7.0) * 100.0
    score = 0.42 * sleep_consistency + 0.33 * stress_component + 0.25 * exercise_component
    return max(0.0, min(100.0, float(score)))


def _estimated_improvement_potential(
    profile: dict[str, Any],
    dashboard_result: dict[str, Any] | None,
    optimization_result: dict[str, Any] | None,
) -> float:
    _, measured_or_estimated = _estimated_top_plan_improvement(dashboard_result, optimization_result)
    if measured_or_estimated is not None:
        return max(0.0, min(0.35, float(measured_or_estimated)))
    lever_scores = _levers_ranked(profile)
    return max(0.0, min(0.35, 0.02 + 0.015 * (lever_scores[0][1] + lever_scores[1][1])))


def _slce_score(profile: dict[str, Any], baseline_risk: float, improvement_potential: float) -> float:
    stability = _system_stability_score(profile)
    score = 0.50 * (1.0 - float(baseline_risk)) * 100.0 + 0.35 * stability + 0.15 * (float(improvement_potential) * 100.0)
    return max(0.0, min(100.0, score))


def _lifestyle_risk_contributions(profile: dict[str, Any]) -> dict[str, float]:
    sleep_pressure = max(0.0, min(100.0, 22.0 * max(0.0, 7.8 - float(profile["sleep_mean_hours"])) + 18.0 * float(profile["sleep_variability_hours"])))
    stress_pressure = max(0.0, min(100.0, ((float(profile["stress_score"]) - 1.0) / 9.0) * 100.0))
    exercise_pressure = max(0.0, min(100.0, ((7.0 - float(profile["exercise_days_per_week"])) / 7.0) * 100.0))
    nutrition_pressure = max(0.0, min(100.0, ((10.0 - float(profile["nutrition_score"])) / 9.0) * 100.0))
    return {
        "Sleep": sleep_pressure,
        "Stress": stress_pressure,
        "Exercise": exercise_pressure,
        "Nutrition": nutrition_pressure,
    }


def _radar_chart(profile: dict[str, Any]):
    values_map = _lifestyle_risk_contributions(profile)
    labels = list(values_map.keys())
    values = [values_map[label] for label in labels]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.4, 4.8), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfbfd")
    ax.plot(angles, values, color="#0f766e", linewidth=2.2)
    ax.fill(angles, values, color="#0f766e", alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("Lifestyle Factor Contribution Radar", fontsize=11, pad=18)
    return fig


def _key_insights(result: dict[str, Any], optimization_result: dict[str, Any] | None = None) -> list[str]:
    sim = result["simulation"]
    profile = result["profile"]
    baseline = result["baseline"]
    improvement = _estimated_improvement_potential(profile, result, optimization_result)
    slce = _slce_score(profile, baseline["probability"], improvement)
    risk_band = float(sim["risk_p95"][-1] - sim["risk_p05"][-1])
    levers = _levers_ranked(profile)

    insights: list[str] = []
    insights.append(f"SLCE Score is {slce:.0f}/100. Higher is better and combines baseline risk, stability, and improvement potential.")
    insights.append(f"Projected mean risk over the horizon is {sim['expected_mean_risk']:.1%}, with {sim['prob_exceed_threshold']:.1%} chance of crossing the 0.60 threshold.")
    if risk_band >= 0.22:
        insights.append("Uncertainty is wide. Consistency in sleep and stress routines can shrink the risk spread.")
    elif risk_band >= 0.12:
        insights.append("Uncertainty is moderate. Small habit improvements should move both median risk and consistency.")
    else:
        insights.append("Uncertainty is relatively tight. Current routine is fairly predictable.")
    insights.append(f"Top drivers right now are {levers[0][0].lower()} and {levers[1][0].lower()}.")
    if optimization_result and optimization_result.get("result", {}).get("top_plans"):
        top = optimization_result["result"]["top_plans"][0]
        insights.append(
            f"Best available strategy ({top['name']}) estimates {top['expected_risk_reduction']:.2%} risk reduction at about {top['time_cost_minutes_per_day']:.1f} min/day."
        )
    return insights


def _simulate_profile_preview(profile: dict[str, Any], *, seed: int = 101) -> dict[str, Any]:
    profile = _clamp_profile_values(profile)
    baseline = predict_baseline_risk(profile)
    simulation = run_monte_carlo(
        profile,
        baseline,
        horizon_years=int(profile["horizon_years"]),
        n_paths=int(profile["simulation_count"]),
        weights=st.session_state.personalization_weights,
        seed=seed,
        threshold=0.6,
    )
    return {"profile": profile, "baseline": baseline, "simulation": simulation}


def _comparison_trend_plot(compare_payload: dict[str, Any]):
    a = compare_payload["scenario_a"]
    b = compare_payload["scenario_b"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfbfd")
    ax.plot(a["simulation"]["time_axis_weeks"], a["simulation"]["risk_median"], label=compare_payload["name_a"], color="#0f766e", linewidth=2.0)
    ax.plot(b["simulation"]["time_axis_weeks"], b["simulation"]["risk_median"], label=compare_payload["name_b"], color="#b45309", linewidth=2.0)
    ax.grid(True, alpha=0.2)
    ax.set_title("Scenario Comparison: Median Risk Trajectories", fontsize=11)
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Risk probability")
    ax.legend(frameon=False, fontsize=8)
    return fig


def _render_monte_carlo_animation(simulation: dict[str, Any]) -> None:
    if not st.session_state.get("explain_mode", True):
        return
    with st.expander("Monte Carlo Animation (Optional)", expanded=False):
        st.caption("Gradually reveals sample simulation paths so users can see uncertainty unfolding over time.")
        if st.button("Play Path Animation", key="play_mc_animation"):
            risk_paths = simulation["risk_paths"]
            max_paths = min(35, risk_paths.shape[0])
            sample = risk_paths[:max_paths]
            weeks = simulation["time_axis_weeks"]
            frame_step = max(1, int(len(weeks) / 18))
            canvas = st.empty()
            for frame_end in range(frame_step, len(weeks) + 1, frame_step):
                fig, ax = plt.subplots(figsize=(7.4, 4.0))
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#fbfbfd")
                for path in sample:
                    ax.plot(weeks[:frame_end], path[:frame_end], color="#0f766e", alpha=0.10, linewidth=0.8)
                ax.set_title(f"Path Reveal ({frame_end}/{len(weeks) - 1} weeks)", fontsize=10)
                ax.set_xlabel("Weeks")
                ax.set_ylabel("Risk probability")
                ax.grid(True, alpha=0.15)
                canvas.pyplot(fig, use_container_width=True)
                plt.close(fig)
                time.sleep(0.08)


def _render_compare_scenarios(current_profile: dict[str, Any] | None = None) -> None:
    st.subheader("Compare Scenarios")
    if st.session_state.get("explain_mode", True):
        st.caption("Run two profiles side by side to compare baseline risk, projected risk, uncertainty, and SLCE Score.")

    options = ["Current Inputs"] + list(DEMO_PROFILES.keys())
    c1, c2, c3 = st.columns([1.6, 1.6, 1.0])
    default_a = 0
    default_b = options.index("High Stress Student") if "High Stress Student" in options else min(1, len(options) - 1)
    selected_a = c1.selectbox("Scenario A", options=options, index=default_a, key="compare_a")
    selected_b = c2.selectbox("Scenario B", options=options, index=default_b, key="compare_b")
    compare_paths = c3.slider("Compare paths", 200, 2000, 700, 100, key="compare_paths")

    if st.button("Run Comparison", key="run_scenario_compare", use_container_width=True):
        try:
            if selected_a == selected_b:
                st.warning("Choose two different scenarios to compare.")
            else:
                profile_a = current_profile if selected_a == "Current Inputs" and current_profile else (
                    _clamp_profile_values(_current_profile_from_state()) if selected_a == "Current Inputs" else dict(DEMO_PROFILES[selected_a])
                )
                profile_b = current_profile if selected_b == "Current Inputs" and current_profile else (
                    _clamp_profile_values(_current_profile_from_state()) if selected_b == "Current Inputs" else dict(DEMO_PROFILES[selected_b])
                )
                profile_a["simulation_count"] = int(compare_paths)
                profile_b["simulation_count"] = int(compare_paths)

                with st.spinner("Comparing scenarios..."):
                    scenario_a = _simulate_profile_preview(profile_a, seed=171)
                    scenario_b = _simulate_profile_preview(profile_b, seed=172)
                st.session_state.compare_result = {
                    "name_a": selected_a,
                    "name_b": selected_b,
                    "scenario_a": scenario_a,
                    "scenario_b": scenario_b,
                }
        except Exception as exc:
            st.error(f"Comparison failed. Please try again. Error type: {exc.__class__.__name__}")

    compare = st.session_state.get("compare_result")
    if not compare:
        return

    a = compare["scenario_a"]
    b = compare["scenario_b"]
    ar = float(a["simulation"]["expected_mean_risk"])
    br = float(b["simulation"]["expected_mean_risk"])
    a_unc = float(a["simulation"]["risk_p95"][-1] - a["simulation"]["risk_p05"][-1])
    b_unc = float(b["simulation"]["risk_p95"][-1] - b["simulation"]["risk_p05"][-1])
    a_impr = _estimated_improvement_potential(a["profile"], a, None)
    b_impr = _estimated_improvement_potential(b["profile"], b, None)
    a_slce = _slce_score(a["profile"], float(a["baseline"]["probability"]), a_impr)
    b_slce = _slce_score(b["profile"], float(b["baseline"]["probability"]), b_impr)

    diff = pd.DataFrame(
        [
            {
                "Metric": "Baseline risk",
                compare["name_a"]: f"{float(a['baseline']['probability']):.1%}",
                compare["name_b"]: f"{float(b['baseline']['probability']):.1%}",
                "Difference (B - A)": f"{(float(b['baseline']['probability']) - float(a['baseline']['probability'])):+.1%}",
            },
            {
                "Metric": "Projected mean risk",
                compare["name_a"]: f"{ar:.1%}",
                compare["name_b"]: f"{br:.1%}",
                "Difference (B - A)": f"{(br - ar):+.1%}",
            },
            {
                "Metric": "Uncertainty range",
                compare["name_a"]: f"{a_unc:.1%}",
                compare["name_b"]: f"{b_unc:.1%}",
                "Difference (B - A)": f"{(b_unc - a_unc):+.1%}",
            },
            {
                "Metric": "SLCE Score",
                compare["name_a"]: f"{a_slce:.0f}/100",
                compare["name_b"]: f"{b_slce:.0f}/100",
                "Difference (B - A)": f"{(b_slce - a_slce):+.1f}",
            },
        ]
    )
    st.dataframe(diff, use_container_width=True, hide_index=True)
    fig = _comparison_trend_plot(compare)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def _experiments_root() -> Path:
    path = Path(__file__).resolve().parent / EXPERIMENTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_experiment_log(profile: dict[str, Any], baseline: dict[str, Any], simulation: dict[str, Any]) -> None:
    try:
        experiments_csv = _experiments_root() / "simulation_runs.csv"
        uncertainty_range = float(simulation["risk_p95"][-1] - simulation["risk_p05"][-1])
        sleep_consistency = max(0.0, 10.0 - (float(profile["sleep_variability_hours"]) / 3.0) * 10.0)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sleep_hours": float(profile["sleep_mean_hours"]),
            "sleep_consistency": float(round(sleep_consistency, 3)),
            "stress_level": float(profile["stress_score"]),
            "exercise_days": float(profile["exercise_days_per_week"]),
            "time_budget": float(profile["time_budget_minutes_per_day"]),
            "baseline_risk": float(baseline["probability"]),
            "projected_risk": float(simulation["expected_mean_risk"]),
            "uncertainty_range": float(round(uncertainty_range, 6)),
        }
        df = pd.DataFrame([row])
        if experiments_csv.exists():
            df.to_csv(experiments_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(experiments_csv, index=False)
    except Exception:
        # Logging must never crash the app during demos.
        return


def _levers_ranked(profile: dict[str, Any]) -> list[tuple[str, float, str]]:
    sleep_gap = max(0.0, 7.8 - float(profile["sleep_mean_hours"])) + 0.6 * max(
        0.0, float(profile["sleep_variability_hours"]) - 0.8
    )
    stress_gap = max(0.0, float(profile["stress_score"]) - 4.0)
    exercise_gap = max(0.0, 5.0 - float(profile["exercise_days_per_week"]))
    nutrition_gap = max(0.0, 8.0 - float(profile["nutrition_score"]))
    scores = [
        (
            "Stress management",
            1.35 * stress_gap,
            "High stress raises downward drift and uncertainty in the weekly health-state model.",
        ),
        (
            "Exercise consistency",
            1.10 * exercise_gap,
            "More active days improve weekly health drift and often reduce risk spread over time.",
        ),
        (
            "Sleep quality/consistency",
            1.20 * sleep_gap,
            "Low average sleep and high variability reduce recovery and increase simulation noise.",
        ),
        (
            "Nutrition quality",
            0.95 * nutrition_gap,
            "Nutrition improves drift more gradually, but it compounds over long horizons.",
        ),
    ]
    return sorted(scores, key=lambda x: x[1], reverse=True)


def _dashboard_interpretation(result: dict[str, Any]) -> str:
    profile = result["profile"]
    sim = result["simulation"]
    levers = _levers_ranked(profile)
    primary = levers[0][0]
    secondary = levers[1][0]
    risk_band = sim["risk_p95"][-1] - sim["risk_p05"][-1]
    uncertainty_note = (
        "uncertainty is relatively tight"
        if risk_band < 0.12
        else "uncertainty is moderate"
        if risk_band < 0.22
        else "uncertainty is wide"
    )
    return (
        f"Your current scenario shows an expected average risk of {sim['expected_mean_risk']:.1%} over the selected horizon. "
        f"The biggest improvement lever appears to be {primary.lower()}, followed by {secondary.lower()}. "
        f"For this profile, {uncertainty_note}, so consistency (especially sleep and stress) matters almost as much as the average values."
    )


def _estimated_top_plan_improvement(
    dashboard_result: dict[str, Any] | None,
    optimization_result: dict[str, Any] | None,
) -> tuple[str, float | None]:
    if dashboard_result and optimization_result:
        top_plans = optimization_result.get("result", {}).get("top_plans", [])
        if top_plans:
            current_risk = float(dashboard_result["simulation"]["expected_mean_risk"])
            top_risk = float(top_plans[0]["expected_mean_risk"])
            delta = max(0.0, current_risk - top_risk)
            return "Based on the current top optimized plan", delta

    if dashboard_result:
        profile = dashboard_result["profile"]
        lever_scores = _levers_ranked(profile)
        # Convert the top two gaps to a coarse estimate for expo messaging.
        estimate = min(0.18, 0.02 + 0.015 * (lever_scores[0][1] + lever_scores[1][1]))
        return "Estimated from current profile levers (run Optimize Plan for a measured value)", estimate

    if optimization_result:
        top_plans = optimization_result.get("result", {}).get("top_plans", [])
        if top_plans:
            top = top_plans[0]
            estimate = min(0.18, 0.03 + 0.01 * max(0.0, 8.0 - float(top.get("target_stress_score", 8.0))))
            return "Estimated from top plan targets (run Dashboard for profile-specific baseline comparison)", estimate

    return "Run a simulation first to estimate improvement", None


def _matching_optimization_for_dashboard(dashboard_result: dict[str, Any] | None) -> dict[str, Any] | None:
    opt = st.session_state.get("optimization_result")
    if not dashboard_result or not opt:
        return None
    try:
        sig1 = _dashboard_profile_signature(dashboard_result["profile"])
        sig2 = _dashboard_profile_signature(opt["profile"])
        if sig1 == sig2:
            return opt
    except Exception:
        return None
    return None


def _render_key_takeaways(
    *,
    dashboard_result: dict[str, Any] | None = None,
    optimization_result: dict[str, Any] | None = None,
) -> None:
    profile = (dashboard_result or {}).get("profile") or (optimization_result or {}).get("profile")
    if not profile:
        return
    levers = _levers_ranked(profile)
    primary = levers[0]
    secondary = levers[1]
    source_label, improvement = _estimated_top_plan_improvement(dashboard_result, optimization_result)

    st.subheader("Key Takeaways")
    st.caption("Most impactful levers and likely upside")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.caption("Primary Lever")
            st.markdown(f"### {primary[0]}")
            st.write(primary[2])
    with c2:
        with st.container(border=True):
            st.caption("Secondary Lever")
            st.markdown(f"### {secondary[0]}")
            st.write(secondary[2])
    with c3:
        with st.container(border=True):
            st.caption("Estimated Improvement")
            if improvement is None:
                st.markdown("### N/A")
                st.write(source_label)
            else:
                st.markdown(f"### {improvement:.1%}")
                st.write(source_label)


def _dashboard_recommendations_text(result: dict[str, Any]) -> str:
    profile = result["profile"]
    levers = _levers_ranked(profile)
    actions: list[str] = []
    top_names = [lever[0] for lever in levers[:2]]
    if "Sleep quality/consistency" in top_names:
        actions.append("stabilize bedtime/wake time to reduce sleep variability")
    if "Stress management" in top_names:
        actions.append("add a daily stress-reset routine (10-15 minutes)")
    if "Exercise consistency" in top_names:
        actions.append("increase active days gradually (even short sessions count)")
    if "Nutrition quality" in top_names:
        actions.append("raise nutrition score one point at a time with simple swaps")
    if not actions:
        actions.append("maintain current habits and focus on consistency")
    return "Best next actions: " + "; ".join(actions[:3]) + "."


def _render_metrics_grid(
    result: dict[str, Any],
    *,
    optimization_result: dict[str, Any] | None = None,
) -> None:
    profile = result["profile"]
    baseline = result["baseline"]
    stability = _system_stability_score(profile)
    improvement = _estimated_improvement_potential(profile, result, optimization_result)
    slce = _slce_score(profile, baseline["probability"], improvement)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Risk", f"{baseline['probability']:.1%}")
    c2.metric("Stability Score", f"{stability:.0f}/100")
    c3.metric("Improvement Potential", f"{improvement:.1%}")
    c4.metric("SLCE Score", f"{slce:.0f}/100")


def _simulation_figures(simulation: dict[str, Any], *, include_health: bool = False) -> dict[str, Any]:
    weeks = simulation["time_axis_weeks"]
    figs = {
        "risk_fan": fan_chart(
            weeks,
            simulation["risk_median"],
            simulation["risk_p05"],
            simulation["risk_p95"],
            title="Risk Over Time (Fan Chart)",
            y_label="Risk probability",
            line_color="#0f766e",
            fill_color="#99f6e4",
        ),
        "risk_hist": risk_histogram(simulation["final_risk"]),
    }
    if include_health:
        figs["health_fan"] = fan_chart(
            weeks,
            simulation["health_median"],
            simulation["health_p05"],
            simulation["health_p95"],
            title="Latent Health State Over Time (Fan Chart)",
            y_label="Health state H_t",
            line_color="#b45309",
            fill_color="#fde68a",
        )
    return figs


def _render_simulation_plots(simulation: dict[str, Any]) -> None:
    figs = _simulation_figures(simulation, include_health=_is_research_mode())

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(figs["risk_fan"], use_container_width=True)
        if st.session_state.get("explain_mode", True):
            st.caption(
                "Uncertainty Fan Chart: the middle line is the median simulation. The shaded band shows the likely spread of outcomes."
            )
    with chart_col2:
        st.pyplot(figs["risk_hist"], use_container_width=True)
        if st.session_state.get("explain_mode", True):
            st.caption(
                "Risk Distribution Histogram: each bar shows how many simulated futures ended in that risk range."
            )

    if _is_research_mode():
        with st.expander("Optional Technical Plot: Latent Health State Fan Chart", expanded=False):
            if "health_fan" in figs:
                st.pyplot(figs["health_fan"], use_container_width=True)

    for fig in figs.values():
        plt.close(fig)


def _export_root() -> Path:
    path = Path(__file__).resolve().parent / EXPORTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model_summary_payload(result: dict[str, Any]) -> dict[str, Any]:
    _, preview_info = _cached_dataset_preview()
    dataset_source = "demo_sample.csv (fallback)"
    if preview_info:
        source = str(preview_info.get("source", "unknown"))
        if source == "local_heart_csv":
            dataset_source = "data/heart.csv"
        elif source == "uci_download":
            dataset_source = "UCI Heart Disease (cached)"
        elif source == "demo_sample":
            dataset_source = "demo_sample.csv (fallback)"
        else:
            dataset_source = source

    baseline = result.get("baseline", {})
    model_type = "Logistic Regression" if baseline.get("source") == "trained_model" else "Heuristic fallback model"
    metadata = baseline.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("model_type") == "logistic_regression":
        model_type = "Logistic Regression"

    return {
        "Dataset used": dataset_source,
        "Model type": model_type,
        "Simulation horizon": f"{int(result['profile']['horizon_years'])} year(s)",
        "Monte Carlo runs": int(result["profile"]["simulation_count"]),
    }


def _blank_tradeoff_figure(message: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    fig.patch.set_facecolor("#ffffff")
    return fig


def _export_key_charts_png(
    dashboard_result: dict[str, Any],
    optimization_result: dict[str, Any] | None,
) -> Path:
    export_dir = _export_root() / f"key_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    figs = _simulation_figures(dashboard_result["simulation"], include_health=False)
    figs["risk_fan"].savefig(export_dir / "uncertainty_fan_chart.png", dpi=180, bbox_inches="tight")
    figs["risk_hist"].savefig(export_dir / "risk_distribution_histogram.png", dpi=180, bbox_inches="tight")
    for fig in figs.values():
        plt.close(fig)

    if optimization_result and optimization_result.get("result", {}).get("all_candidates"):
        top_ids = [p["id"] for p in optimization_result["result"].get("top_plans", [])]
        tradeoff = tradeoff_scatter(
            optimization_result["result"]["all_candidates"],
            top_ids=top_ids,
            y_field="expected_risk_reduction",
        )
    else:
        tradeoff = _blank_tradeoff_figure("Run Optimize Plan to generate tradeoff data")
    tradeoff.savefig(export_dir / "optimization_tradeoff_plot.png", dpi=180, bbox_inches="tight")
    plt.close(tradeoff)
    return export_dir


def _export_dashboard_results(result: dict[str, Any]) -> Path:
    export_dir = _export_root() / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    figs = _simulation_figures(result["simulation"], include_health=True)
    figs["risk_fan"].savefig(export_dir / "risk_fan_chart.png", dpi=180, bbox_inches="tight")
    figs["health_fan"].savefig(export_dir / "health_fan_chart.png", dpi=180, bbox_inches="tight")
    figs["risk_hist"].savefig(export_dir / "risk_histogram.png", dpi=180, bbox_inches="tight")
    for fig in figs.values():
        plt.close(fig)

    summary = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "profile": result["profile"],
        "baseline": {
            "probability": float(result["baseline"]["probability"]),
            "logit": float(result["baseline"]["logit"]),
            "source": result["baseline"].get("source"),
        },
        "simulation_summary": {
            "expected_mean_risk": float(result["simulation"]["expected_mean_risk"]),
            "prob_exceed_threshold": float(result["simulation"]["prob_exceed_threshold"]),
            "final_health_mean": float(result["simulation"]["final_health_mean"]),
            "final_health_std": float(result["simulation"]["final_health_std"]),
            "final_risk_mean": float(result["simulation"]["final_risk_mean"]),
            "final_risk_std": float(result["simulation"]["final_risk_std"]),
            "weeks": int(result["simulation"]["weeks"]),
            "threshold": float(result["simulation"]["threshold"]),
        },
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return export_dir


def _export_optimizer_results(stored: dict[str, Any]) -> Path:
    export_dir = _export_root() / f"optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)
    result = stored["result"]
    top_plans = result.get("top_plans", [])
    pd.DataFrame(top_plans).to_csv(export_dir / "top_plans.csv", index=False)
    pd.DataFrame(result.get("all_candidates", [])).to_csv(export_dir / "all_candidates.csv", index=False)
    fig = tradeoff_scatter(
        result.get("all_candidates", []),
        top_ids=[p["id"] for p in top_plans],
        y_field="expected_risk_reduction",
    )
    fig.savefig(export_dir / "tradeoff_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "profile": stored.get("profile"),
        "constraints": stored.get("constraints"),
        "candidate_count": int(result.get("candidate_count", 0)),
        "top_plans": [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "expected_mean_risk": float(p.get("expected_mean_risk", 0.0)),
                "expected_risk_reduction": float(p.get("expected_risk_reduction", 0.0)),
                "prob_exceed_threshold": float(p.get("prob_exceed_threshold", 0.0)),
                "time_cost_minutes_per_day": float(p.get("time_cost_minutes_per_day", 0.0)),
                "adherence_score": float(p.get("adherence_score", 0.0)),
                "objective": float(p.get("objective", 0.0)),
            }
            for p in top_plans
        ],
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return export_dir


def _export_weekly_log_snapshot() -> Path:
    export_dir = _export_root() / f"weekly_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_dir.mkdir(parents=True, exist_ok=True)
    if "weekly_log_editor_df" in st.session_state:
        st.session_state.weekly_log_editor_df.to_csv(export_dir / "weekly_log_current.csv", index=False)
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "personalization_weights": st.session_state.get("personalization_weights"),
        "last_personalization_update": st.session_state.get("last_personalization_update", {}),
        "log_history_count": len(st.session_state.get("log_history", [])),
    }
    # `last_personalization_update` includes only JSON-safe values in this app flow.
    (export_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return export_dir


def _render_research_dashboard_details(result: dict[str, Any]) -> None:
    if not _is_research_mode():
        return
    with st.expander("Research Details: Dashboard Model Assumptions", expanded=False):
        st.markdown("**Stochastic update (weekly):**")
        st.latex(r"H_{t+1}=\mathrm{clamp}(H_t + \mathrm{drift} + \mathrm{mean\ reversion} + \epsilon_t, 0, 100)")
        st.markdown(r"Noise term: $\epsilon_t \sim \mathcal{N}(0,\sigma)$, where $\sigma$ increases with stress and sleep variability.")
        st.markdown("**Risk mapping:**")
        st.latex(r"p_t = \sigma\left(\alpha + \beta \cdot \frac{100-H_t}{100} + \text{baseline\_logit}\right)")
        st.write(
            {
                "horizon_years": result["profile"]["horizon_years"],
                "simulation_count": result["profile"]["simulation_count"],
                "personalization_weights": st.session_state.personalization_weights,
                "baseline_source": result["baseline"].get("source"),
            }
        )


def _render_research_optimizer_details(stored: dict[str, Any]) -> None:
    if not _is_research_mode():
        return
    with st.expander("Research Details: Optimization Objective & Constraints", expanded=False):
        st.markdown("Candidates are generated on a bounded grid and filtered by daily time budget.")
        st.latex(
            r"\text{objective} = \mathbb{E}[\text{mean risk}] + \lambda_t \cdot \text{time cost} - \lambda_a \cdot \text{adherence}"
        )
        st.write(
            {
                "constraints": stored.get("constraints"),
                "candidate_count": stored.get("result", {}).get("candidate_count"),
                "optimization_paths": stored.get("constraints", {}).get("optimization_paths"),
            }
        )


def _render_research_weekly_details() -> None:
    if not _is_research_mode():
        return
    with st.expander("Research Details: Personalization Update Rule", expanded=False):
        st.markdown(
            "Weights update incrementally based on correlations between weekly habit summaries and a deterministic health-outcome proxy."
        )
        st.write(
            {
                "weight_bounds": [0.6, 1.8],
                "current_weights": st.session_state.personalization_weights,
                "log_history_count": len(st.session_state.log_history),
            }
        )


def _plan_why_text(plan: dict[str, Any], profile: dict[str, Any]) -> str:
    changes: list[str] = []
    if plan["target_stress_score"] < profile["stress_score"]:
        changes.append(
            f"reduces stress from {profile['stress_score']} to {plan['target_stress_score']}"
        )
    if plan["target_exercise_days"] > profile["exercise_days_per_week"]:
        changes.append(
            f"increases exercise days from {profile['exercise_days_per_week']} to {plan['target_exercise_days']}"
        )
    if plan["target_sleep_mean"] > profile["sleep_mean_hours"]:
        changes.append(
            f"raises average sleep from {profile['sleep_mean_hours']}h to {plan['target_sleep_mean']}h"
        )
    if plan["target_nutrition_score"] > profile["nutrition_score"]:
        changes.append(
            f"improves nutrition score from {profile['nutrition_score']} to {plan['target_nutrition_score']}"
        )
    if not changes:
        return "Why this plan: It preserves current habits while fitting strict constraints and maximizing adherence."
    return "Why this plan: It " + ", ".join(changes[:3]) + ", while staying within the daily time budget."


def _dashboard_page() -> None:
    st.header("Dashboard")
    if not _is_research_mode():
        _consumer_callout(
            "Use this page to estimate risk under uncertainty. Start with a demo profile, run the simulation, then read the interpretation before changing habits."
        )
    _render_demo_day_notice()
    st.subheader("User Inputs")
    _render_demo_profile_picker()
    try:
        _render_project_status()
    except Exception:
        st.warning("Model status preview is temporarily unavailable, but the simulation can still run.")

    with st.form("dashboard_form"):
        raw_inputs = _render_shared_inputs()
        run_clicked = st.form_submit_button(
            "Run Simulation",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.busy_dashboard,
            help="Runs the baseline model and then Monte Carlo simulation over the selected horizon.",
        )

    if run_clicked:
        validation = validate_dashboard_inputs(raw_inputs)
        if not validation.ok:
            for msg in validation.errors:
                st.error(msg)
        else:
            st.session_state.busy_dashboard = True
            try:
                profile = validation.values
                with st.spinner("Running baseline model + Monte Carlo simulation..."):
                    _run_dashboard_simulation(profile, seed=42)
            except Exception as exc:
                st.error(
                    "Simulation failed, but the app is still running. "
                    f"Please adjust inputs and try again. Error type: {exc.__class__.__name__}"
                )
            finally:
                st.session_state.busy_dashboard = False

    result = st.session_state.dashboard_result
    if result:
        st.subheader("Simulation Results")
        top_c1, top_c2 = st.columns([3, 1])
        matched_opt = _matching_optimization_for_dashboard(result)
        with top_c1:
            _render_metrics_grid(result, optimization_result=matched_opt)
            st.caption("Stability Score combines sleep consistency, stress level, and exercise frequency on a 0-100 scale.")
        with top_c2:
            with st.container(border=True):
                st.caption("Run Info")
                st.write(f"Horizon: {result['profile']['horizon_years']} year(s)")
                st.write(f"Paths: {result['profile']['simulation_count']}")
                if st.button("Export Key Charts (PNG)", use_container_width=True, key="export_key_charts"):
                    try:
                        export_path = _export_key_charts_png(result, matched_opt)
                        st.session_state.last_export_path = str(export_path)
                        st.success(f"Saved to {export_path}")
                        if matched_opt is None:
                            st.info("Tradeoff plot exported with placeholder text. Run Optimize Plan first for full tradeoff data.")
                    except Exception as exc:
                        st.error(f"Export failed: {exc.__class__.__name__}")
                if st.button("Export Full Results", use_container_width=True, key="export_dashboard_results"):
                    try:
                        export_path = _export_dashboard_results(result)
                        st.session_state.last_export_path = str(export_path)
                        st.success(f"Saved to {export_path}")
                    except Exception as exc:
                        st.error(f"Export failed: {exc.__class__.__name__}")

        st.markdown("#### Model Summary")
        st.dataframe(
            pd.DataFrame([_model_summary_payload(result)]),
            use_container_width=True,
            hide_index=True,
        )

        baseline = result["baseline"]
        if baseline.get("note"):
            st.info(baseline["note"])
        if _is_research_mode():
            st.caption(
                f"Layer 1 source: {baseline.get('source')} | "
                f"Layer 2 weights: {json.dumps(st.session_state.personalization_weights)}"
            )
        _render_simulation_plots(result["simulation"])
        _render_monte_carlo_animation(result["simulation"])

        insight_col1, insight_col2 = st.columns([1.55, 1.0])
        with insight_col1:
            st.subheader("Key Insights")
            for line in _key_insights(result, matched_opt):
                st.write(f"- {line}")
        with insight_col2:
            radar = _radar_chart(result["profile"])
            st.pyplot(radar, use_container_width=True)
            plt.close(radar)
            if st.session_state.get("explain_mode", True):
                st.caption("Radar chart shows estimated relative contribution of each lifestyle factor to current risk pressure.")

        st.subheader("Model Explanation")
        st.markdown("**What This Means**")
        st.markdown(f"<div class='vita-callout'>{_dashboard_interpretation(result)}</div>", unsafe_allow_html=True)
        if st.session_state.get("explain_mode", True):
            st.caption(
                "Monte Carlo simulation runs many possible futures. Uncertainty bands show the range of likely outcomes instead of a single prediction."
            )
        _consumer_callout(_dashboard_recommendations_text(result))
        levers = _levers_ranked(result["profile"])
        with st.container(border=True):
            st.markdown("**What changed risk the most (for this profile)**")
            st.write(f"1. {levers[0][0]}")
            st.write(f"2. {levers[1][0]}")
            st.write(f"3. {levers[2][0]}")

        _render_key_takeaways(
            dashboard_result=result,
            optimization_result=matched_opt,
        )
        _render_research_dashboard_details(result)

    current_validation = validate_dashboard_inputs(_current_profile_from_state())
    _render_compare_scenarios(current_validation.values if current_validation.ok else None)


def _optimizer_page() -> None:
    st.header("Optimize Plan")
    if not _is_research_mode():
        _consumer_callout(
            "This page searches for realistic habit plans that lower risk while respecting time and weekly ramp limits."
        )

    if "current_profile" not in st.session_state:
        st.warning("No validated profile from Dashboard yet. Using the current inputs after validation.")

    st.subheader("Constraint Inputs")
    with st.expander("Profile Used for Optimization", expanded=True):
        _render_demo_profile_picker()

    with st.form("optimizer_form"):
        with st.expander("Profile Used for Optimization", expanded=True):
            raw_profile = _render_shared_inputs()

        c1, c2, c3 = st.columns(3)
        with c1:
            max_minutes = st.slider(
                "Max minutes/day",
                0,
                180,
                int(st.session_state.get("profile_time_budget_minutes_per_day", 45)),
                5,
                help="Typical school-week budgets are often 20-60 minutes/day.",
            )
            max_ex_days = st.slider(
                "Max exercise days/week",
                0,
                7,
                6,
                1,
                help="Common realistic target range is 3-6 days/week.",
            )
        with c2:
            max_sleep_inc = st.slider(
                "Max sleep increase per week (hrs)",
                0.0,
                2.0,
                0.5,
                0.1,
                help="A gradual range like 0.3-0.7 hr/week is more realistic than sudden jumps.",
            )
            max_stress_red = st.slider(
                "Max stress reduction per week",
                0.0,
                3.0,
                1.0,
                0.1,
                help="Use 0.5-1.5 for realistic week-to-week stress improvement.",
            )
        with c3:
            nutrition_cap = st.slider(
                "Nutrition improvement cap / week",
                0.0,
                3.0,
                1.0,
                0.1,
                help="1 point/week is a good gradual target for sustainable changes.",
            )
            optimization_paths = st.slider(
                "Optimization Monte Carlo paths",
                100,
                1000,
                300,
                50,
                help="300-500 is a good balance for live demos.",
            )

        optimize_clicked = st.form_submit_button(
            "Find Best Plan",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.busy_optimizer,
            help="Runs a bounded search with Monte Carlo evaluation for each candidate plan.",
        )

    if optimize_clicked:
        profile_validation = validate_dashboard_inputs(raw_profile)
        constraints_validation = validate_optimizer_constraints(
            {
                "max_minutes_per_day": max_minutes,
                "max_exercise_days_per_week": max_ex_days,
                "max_sleep_increase_per_week": max_sleep_inc,
                "max_stress_reduction_per_week": max_stress_red,
                "nutrition_improvement_cap_per_week": nutrition_cap,
                "optimization_paths": optimization_paths,
            }
        )

        all_errors = profile_validation.errors + constraints_validation.errors
        if all_errors:
            for msg in all_errors:
                st.error(msg)
        else:
            st.session_state.busy_optimizer = True
            try:
                profile = _clamp_profile_values(profile_validation.values)
                constraints = _clamp_optimizer_constraints(constraints_validation.values)
                st.session_state.current_profile = profile
                with st.spinner("Evaluating candidate plans with Monte Carlo..."):
                    baseline = predict_baseline_risk(profile)
                    opt_result = optimize_habit_plans(
                        profile,
                        baseline,
                        constraints,
                        horizon_years=int(profile["horizon_years"]),
                        personalization_weights=st.session_state.personalization_weights,
                    )
                st.session_state.optimization_result = {
                    "profile": profile,
                    "baseline": baseline,
                    "constraints": constraints,
                    "result": opt_result,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            except Exception as exc:
                st.error(
                    "Optimization failed, but the app is still stable. "
                    f"Try a shorter horizon or fewer paths. Error type: {exc.__class__.__name__}"
                )
            finally:
                st.session_state.busy_optimizer = False

    stored = st.session_state.optimization_result
    if stored:
        result = stored["result"]
        st.subheader("Top 3 Plans")
        st.caption(
            f"Evaluated {result['candidate_count']} candidates | "
            f"Horizon: {stored['profile']['horizon_years']} year(s) | "
            f"Timestamp: {stored['timestamp']}"
        )
        if "baseline_expected_risk" in result:
            st.caption(f"Baseline expected mean risk (no new plan): {result['baseline_expected_risk']:.2%}")

        if not result["top_plans"]:
            st.warning("No feasible plans met the constraints. Increase the time budget or relax limits.")
            return

        if not _is_research_mode():
            _consumer_callout(
                "Read the plan cards first. The best plan is not always the most aggressive one; it balances expected risk reduction with time and adherence."
            )
        reductions = [float(plan["expected_risk_reduction"]) for plan in result["top_plans"]]
        if reductions and (max(reductions) - min(reductions) < 0.002):
            st.info(
                "Top plans are close in projected risk reduction under current constraints. "
                "Increase time budget or adjust ramp limits for more separation."
            )

        for plan in result["top_plans"]:
            with st.container(border=True):
                top_cols = st.columns([1.3, 1, 1, 1])
                top_cols[0].markdown(f"### {plan['name']}")
                top_cols[0].caption(f"Rank {plan.get('rank', '')} | {plan['id']}")
                top_cols[1].metric("Risk Reduction", f"{plan['expected_risk_reduction']:.3%}")
                top_cols[2].metric("Time Cost", f"{plan['time_cost_minutes_per_day']:.1f} min/day")
                top_cols[3].metric("Adherence", f"{plan['adherence_score']:.2f}")

                tag_line = " ".join(
                    [
                        f"<span class='vita-pill'>Sleep {plan['target_sleep_mean']}h</span>",
                        f"<span class='vita-pill'>Exercise {plan['target_exercise_days']}d/wk</span>",
                        f"<span class='vita-pill'>Stress {plan['target_stress_score']}</span>",
                        f"<span class='vita-pill'>Nutrition {plan['target_nutrition_score']}</span>",
                    ]
                )
                st.markdown(tag_line, unsafe_allow_html=True)
                st.write(_plan_why_text(plan, stored["profile"]))

                with st.expander("Ramp schedule preview (first 8 weeks)", expanded=(plan.get("rank") == 1)):
                    st.dataframe(pd.DataFrame(plan["schedule_preview"]), use_container_width=True, hide_index=True)

        plot_row1, plot_row2 = st.columns([3, 1])
        with plot_row1:
            fig = tradeoff_scatter(
                result["all_candidates"],
                top_ids=[plan["id"] for plan in result["top_plans"]],
                y_field="expected_risk_reduction",
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            if st.session_state.get("explain_mode", True):
                st.caption(
                    "Optimizer tradeoff: each point is a candidate plan. Higher points reduce more risk; leftward points require less daily time."
                )
        with plot_row2:
            with st.container(border=True):
                st.caption("Export")
                st.write("Save plan results and plots for your logbook.")
                if st.button("Export Results", use_container_width=True, key="export_optimizer_results"):
                    try:
                        export_path = _export_optimizer_results(stored)
                        st.session_state.last_export_path = str(export_path)
                        st.success(f"Saved to {export_path}")
                    except Exception as exc:
                        st.error(f"Export failed: {exc.__class__.__name__}")

        _render_key_takeaways(
            dashboard_result=st.session_state.get("dashboard_result"),
            optimization_result=stored,
        )
        if st.session_state.get("explain_mode", True):
            st.info("Optimizer selection rule: choose plans that lower expected risk while balancing time cost and adherence.")
        _render_research_optimizer_details(stored)


def _default_weekly_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Sleep Hours": [7.0, 7.5, 6.8, 7.2, 6.5, 8.1, 8.0],
            "Stress (1-10)": [5, 6, 7, 5, 6, 4, 3],
            "Exercise Minutes": [20, 0, 35, 15, 0, 45, 30],
            "Nutrition (1-10)": [6, 6, 5, 7, 6, 7, 7],
        }
    )


def _weekly_log_page() -> None:
    st.header("Weekly Log")
    if not _is_research_mode():
        _consumer_callout(
            "Enter one week of habits to personalize how strongly sleep, stress, exercise, and nutrition affect your simulated outcomes."
        )
    st.write("Enter 7 days of habits. The personalization layer updates sensitivity weights based on your logged patterns.")

    if "weekly_log_editor_df" not in st.session_state:
        st.session_state.weekly_log_editor_df = _default_weekly_log_df()

    st.subheader("7-Day Input Form")
    edited = st.data_editor(
        st.session_state.weekly_log_editor_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="weekly_log_editor",
        column_config={
            "Day": st.column_config.TextColumn(disabled=True, help="Day labels for the 7-day log."),
            "Sleep Hours": st.column_config.NumberColumn(
                min_value=0.0, max_value=16.0, step=0.1, help="Typical target for teens is often ~7-9 hours/night."
            ),
            "Stress (1-10)": st.column_config.NumberColumn(
                min_value=1, max_value=10, step=1, help="Self-rating where 1 is very low stress and 10 is very high."
            ),
            "Exercise Minutes": st.column_config.NumberColumn(
                min_value=0, max_value=300, step=1, help="Include workouts, sports, or brisk activity minutes."
            ),
            "Nutrition (1-10)": st.column_config.NumberColumn(
                min_value=1, max_value=10, step=1, help="Simple quality score for the day (balanced meals, hydration, etc.)."
            ),
        },
    )
    st.session_state.weekly_log_editor_df = edited

    submit = st.button(
        "Update Personalization",
        type="primary",
        disabled=st.session_state.busy_weekly_log,
        help="Updates personalization weights from weekly logs and shows the reason.",
    )

    if submit:
        st.session_state.busy_weekly_log = True
        try:
            sleep = edited["Sleep Hours"].tolist()
            stress = edited["Stress (1-10)"].tolist()
            exercise = edited["Exercise Minutes"].tolist()
            nutrition = edited["Nutrition (1-10)"].tolist()

            validation = validate_weekly_log(sleep, stress, exercise, nutrition)
            if not validation.ok:
                for msg in validation.errors:
                    st.error(msg)
            else:
                entry = {
                    "sleep_hours": [float(x) for x in validation.values["sleep_hours"]],
                    "stress_scores": [float(x) for x in validation.values["stress_scores"]],
                    "exercise_minutes": [float(x) for x in validation.values["exercise_minutes"]],
                    "nutrition_scores": [float(x) for x in validation.values["nutrition_scores"]],
                    "logged_at": datetime.now().isoformat(timespec="seconds"),
                }
                st.session_state.log_history.append(entry)

                baseline_profile = st.session_state.get("current_profile", validate_dashboard_inputs(_current_profile_from_state()).values or DEMO_PROFILES["Balanced"])
                with st.spinner("Updating personalization weights..."):
                    update = update_personalization_weights(
                        st.session_state.personalization_weights,
                        baseline_profile=baseline_profile,
                        log_history=st.session_state.log_history,
                    )
                st.session_state.personalization_weights = update["new_weights"]
                st.session_state.log_history = update["history"]
                st.session_state.last_personalization_update = update
        except Exception as exc:
            st.error(
                "Weekly personalization update failed. Please check the inputs and try again. "
                f"Error type: {exc.__class__.__name__}"
            )
        finally:
            st.session_state.busy_weekly_log = False

    st.subheader("Before vs After Weights")
    st.subheader("Current Sensitivity Weights")
    st.json(st.session_state.personalization_weights)

    if "last_personalization_update" in st.session_state:
        update = st.session_state.last_personalization_update
        comp = pd.DataFrame(
            {
                "Weight": list(update["new_weights"].keys()),
                "Previous": [round(update["previous_weights"][k], 3) for k in update["new_weights"].keys()],
                "New": [round(update["new_weights"][k], 3) for k in update["new_weights"].keys()],
            }
        )
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.info(f"Your model adapted because: {update['explanation']}")

    export_col1, export_col2 = st.columns([1, 3])
    if export_col1.button("Export Results", key="export_weekly_results", use_container_width=True):
        try:
            export_path = _export_weekly_log_snapshot()
            st.session_state.last_export_path = str(export_path)
            st.success(f"Saved to {export_path}")
        except Exception as exc:
            st.error(f"Export failed: {exc.__class__.__name__}")
    if st.session_state.get("last_export_path"):
        export_col2.caption(f"Last export: {st.session_state.last_export_path}")

    if st.session_state.log_history:
        st.subheader("Log History (Summaries)")
        rows = []
        for item in st.session_state.log_history:
            summary = item.get("summary")
            if not summary:
                continue
            rows.append(
                {
                    "Logged At": item.get("logged_at", ""),
                    "Sleep Mean": round(summary["sleep_mean_hours"], 2),
                    "Sleep Variability": round(summary["sleep_variability_hours"], 2),
                    "Stress Mean": round(summary["stress_score"], 2),
                    "Exercise Days": round(summary["exercise_days_per_week"], 1),
                    "Nutrition Mean": round(summary["nutrition_score"], 2),
                    "Outcome Proxy": round(float(item.get("outcome_proxy", 0.0)), 2),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    _render_research_weekly_details()


def main() -> None:
    st.set_page_config(page_title="SLCE - STEM Expo", page_icon="🧪", layout="wide")
    _inject_styles()
    _ensure_session_state()
    _render_header()
    _render_global_toggles()
    _maybe_run_demo_day_bootstrap()

    page = st.sidebar.radio(
        "Navigation",
        options=["Dashboard", "Optimize Plan", "Weekly Log"],
        key="nav_page",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("SLCE STEM Expo Demo")
    st.sidebar.write(f"Mode: {st.session_state.ui_mode}")
    st.sidebar.write(f"Demo Day Mode: {'On' if st.session_state.demo_day_mode else 'Off'}")
    st.sidebar.write(f"Explain Mode: {'On' if st.session_state.explain_mode else 'Off'}")
    if st.session_state.get("last_export_path"):
        st.sidebar.caption(f"Last export:\n{st.session_state.last_export_path}")
    st.sidebar.info(DISCLAIMER_TEXT)

    if page == "Dashboard":
        _dashboard_page()
    elif page == "Optimize Plan":
        _optimizer_page()
    else:
        _weekly_log_page()


if __name__ == "__main__":
    main()
