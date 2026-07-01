from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .plots import fan_chart, risk_histogram, tradeoff_scatter


def export_research_pdf(
    destination: Path,
    dashboard_result: dict[str, Any],
    optimization_result: dict[str, Any] | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = dashboard_result["profile"]
    baseline = dashboard_result["baseline"]
    simulation = dashboard_result["simulation"]
    evidence = dashboard_result.get("input_evidence", {})
    lab_impact = dashboard_result.get("lab_impact")

    with PdfPages(destination) as pdf:
        title = plt.figure(figsize=(8.5, 11))
        title.patch.set_facecolor("white")
        title.text(0.08, 0.93, "SLCE Research Summary", fontsize=24, weight="bold", color="#16212f")
        title.text(0.08, 0.895, "Stochastic Lifestyle Control Engine", fontsize=13, color="#0f766e")
        lines = [
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"Model: {baseline.get('source', 'unknown')} logistic baseline + stochastic simulation",
            f"Horizon: {profile['horizon_years']} year(s) | Monte Carlo paths: {profile['simulation_count']}",
            f"Baseline probability: {float(baseline['probability']):.2%}",
            f"Expected mean simulated risk: {float(simulation['expected_mean_risk']):.2%}",
            f"Final-risk 5th-95th percentile: {np.percentile(simulation['final_risk'], 5):.2%} to {np.percentile(simulation['final_risk'], 95):.2%}",
            f"Input Evidence Score: {float(evidence.get('score', 0)):.0f}/100 ({evidence.get('level', 'Unknown')})",
        ]
        if lab_impact:
            lines.extend(
                [
                    "",
                    "Measured-lab impact analysis",
                    f"With measured labs: {float(lab_impact['measured_probability']):.2%}",
                    f"With proxy/imputed labs: {float(lab_impact['proxy_probability']):.2%}",
                    f"Difference: {float(lab_impact['absolute_difference']):+.2%}",
                ]
            )
        y = 0.82
        for line in lines:
            title.text(0.08, y, line, fontsize=11, color="#334155")
            y -= 0.038
        title.text(
            0.08,
            0.12,
            "Educational tool only. Not medical advice or diagnosis.\n"
            "Scenario changes are outputs under stated assumptions, not causal treatment effects.",
            fontsize=10,
            color="#991b1b",
        )
        pdf.savefig(title, bbox_inches="tight")
        plt.close(title)

        risk_fan = fan_chart(
            simulation["time_axis_weeks"],
            simulation["risk_median"],
            simulation["risk_p05"],
            simulation["risk_p95"],
            title="Risk Over Time Under Uncertainty",
            y_label="Risk probability",
            line_color="#0f766e",
            fill_color="#99d8c9",
        )
        pdf.savefig(risk_fan, bbox_inches="tight")
        plt.close(risk_fan)

        histogram = risk_histogram(simulation["final_risk"])
        pdf.savefig(histogram, bbox_inches="tight")
        plt.close(histogram)

        stored_result = (optimization_result or {}).get("result", {})
        candidates = stored_result.get("all_candidates", [])
        if candidates:
            tradeoff = tradeoff_scatter(
                candidates,
                top_ids=[plan["id"] for plan in stored_result.get("top_plans", [])],
                y_field="expected_risk_reduction",
            )
            pdf.savefig(tradeoff, bbox_inches="tight")
            plt.close(tradeoff)
    return destination
