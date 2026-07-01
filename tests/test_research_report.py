from pathlib import Path

import numpy as np

from src.research_report import export_research_pdf


def test_research_pdf_exports_without_optional_optimizer(tmp_path: Path):
    weeks = np.arange(3)
    result = {
        "profile": {"horizon_years": 1, "simulation_count": 500},
        "baseline": {"probability": 0.2, "source": "test"},
        "input_evidence": {"score": 75, "level": "Moderate"},
        "simulation": {
            "time_axis_weeks": weeks,
            "risk_median": np.array([0.2, 0.21, 0.22]),
            "risk_p05": np.array([0.18, 0.18, 0.19]),
            "risk_p95": np.array([0.22, 0.24, 0.26]),
            "final_risk": np.array([0.19, 0.22, 0.25]),
            "expected_mean_risk": 0.21,
        },
    }
    destination = export_research_pdf(tmp_path / "summary.pdf", result)
    assert destination.exists()
    assert destination.read_bytes().startswith(b"%PDF")
