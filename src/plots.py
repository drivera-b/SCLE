from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _base_figure(figsize=(7.5, 4.2)):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fbfbfd")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(True, alpha=0.18)
    return fig, ax


def fan_chart(
    x: np.ndarray,
    median: np.ndarray,
    p05: np.ndarray,
    p95: np.ndarray,
    *,
    title: str,
    y_label: str,
    line_color: str,
    fill_color: str,
):
    fig, ax = _base_figure()
    ax.fill_between(x, p05, p95, color=fill_color, alpha=0.28, label="5th-95th percentile")
    ax.plot(x, median, color=line_color, linewidth=2.2, label="Median")
    ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.set_xlabel("Weeks")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False, fontsize=9)
    return fig


def risk_histogram(final_risk: np.ndarray):
    fig, ax = _base_figure()
    ax.hist(final_risk, bins=20, color="#d95f02", alpha=0.82, edgecolor="white")
    ax.set_title("Final Risk Distribution", fontsize=12, fontweight="semibold")
    ax.set_xlabel("Risk probability at horizon end")
    ax.set_ylabel("Count")
    return fig


def tradeoff_scatter(
    candidates: list[dict[str, Any]],
    top_ids: list[str] | None = None,
    *,
    y_field: str = "expected_mean_risk",
):
    top_ids = set(top_ids or [])
    fig, ax = _base_figure()
    x = [c["time_cost_minutes_per_day"] for c in candidates]
    y = [c[y_field] for c in candidates]
    ax.scatter(x, y, s=34, color="#8da0cb", alpha=0.7, label="Candidates")
    if top_ids:
        top = [c for c in candidates if c.get("id") in top_ids]
        if top:
            ax.scatter(
                [c["time_cost_minutes_per_day"] for c in top],
                [c[y_field] for c in top],
                s=70,
                color="#1b9e77",
                edgecolor="black",
                linewidth=0.4,
                label="Top 3",
                zorder=3,
            )
            for c in top:
                ax.annotate(
                    c.get("name", c.get("id", "")),
                    (c["time_cost_minutes_per_day"], c["expected_mean_risk"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
    if y_field == "expected_risk_reduction":
        ax.set_title("Risk Reduction vs Time Cost Tradeoff", fontsize=12, fontweight="semibold")
        y_label = "Expected risk reduction"
    else:
        ax.set_title("Risk vs Time Cost Tradeoff", fontsize=12, fontweight="semibold")
        y_label = "Expected mean risk"
    ax.set_xlabel("Estimated minutes/day")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False, fontsize=9)
    return fig
