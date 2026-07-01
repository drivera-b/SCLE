from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd


LAB_SPECS: dict[str, dict[str, Any]] = {
    "systolic_bp": {
        "label": "Systolic blood pressure",
        "unit": "mm Hg",
        "range": (70.0, 250.0),
        "aliases": {"systolic_bp", "systolic_blood_pressure", "sbp"},
        "unit_aliases": {"mmhg", "mm_hg"},
    },
    "total_cholesterol": {
        "label": "Total cholesterol",
        "unit": "mg/dL",
        "range": (80.0, 500.0),
        "aliases": {"total_cholesterol", "cholesterol_total", "cholesterol"},
        "unit_aliases": {"mgdl", "mg_dl", "mg/dl"},
    },
    "fasting_glucose": {
        "label": "Fasting glucose",
        "unit": "mg/dL",
        "range": (40.0, 400.0),
        "aliases": {"fasting_glucose", "glucose_fasting", "glucose"},
        "unit_aliases": {"mgdl", "mg_dl", "mg/dl"},
    },
    "hba1c": {
        "label": "HbA1c",
        "unit": "%",
        "range": (3.0, 20.0),
        "aliases": {"hba1c", "a1c", "hemoglobin_a1c"},
        "unit_aliases": {"percent", "pct", "%"},
    },
    "bmi": {
        "label": "BMI",
        "unit": "kg/m^2",
        "range": (10.0, 70.0),
        "aliases": {"bmi", "body_mass_index"},
        "unit_aliases": {"kgm2", "kg_m2", "kg/m2", "kg/m^2"},
    },
}


@dataclass(frozen=True)
class LabImportResult:
    values: dict[str, float]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.values)


def _normalize(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() or char in {"%", "/", "^"} else "_" for char in text).strip("_")


def lab_csv_template() -> bytes:
    columns: list[str] = []
    values: list[str] = []
    examples = {
        "systolic_bp": "128",
        "total_cholesterol": "198",
        "fasting_glucose": "96",
        "hba1c": "5.5",
        "bmi": "25.4",
    }
    for field, spec in LAB_SPECS.items():
        columns.extend([field, f"{field}_unit"])
        values.extend([examples[field], str(spec["unit"])])
    return (",".join(columns) + "\n" + ",".join(values) + "\n").encode("utf-8")


def parse_lab_csv(payload: bytes | bytearray) -> LabImportResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        frame = pd.read_csv(BytesIO(bytes(payload)))
    except Exception:
        return LabImportResult({}, ["The lab file could not be read as CSV."], [])

    if frame.empty:
        return LabImportResult({}, ["The lab CSV is empty."], [])
    if len(frame) != 1:
        return LabImportResult({}, ["Use exactly one row per lab import."], [])

    normalized_columns = {_normalize(column): column for column in frame.columns}
    row = frame.iloc[0]
    values: dict[str, float] = {}
    missing_units: list[str] = []

    for field, spec in LAB_SPECS.items():
        matching = next((alias for alias in spec["aliases"] if _normalize(alias) in normalized_columns), None)
        if matching is None:
            continue
        source_column = normalized_columns[_normalize(matching)]
        raw_value = row[source_column]
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{spec['label']} must be numeric.")
            continue
        if not np.isfinite(value):
            errors.append(f"{spec['label']} must be a finite number.")
            continue
        low, high = spec["range"]
        if not low <= value <= high:
            errors.append(f"{spec['label']} must be between {low:g} and {high:g} {spec['unit']}.")
            continue

        unit_key = _normalize(f"{field}_unit")
        if unit_key in normalized_columns:
            unit = _normalize(row[normalized_columns[unit_key]])
            accepted = {_normalize(item) for item in spec["unit_aliases"]}
            if unit not in accepted:
                errors.append(f"{spec['label']} must use {spec['unit']}; received '{row[normalized_columns[unit_key]]}'.")
                continue
        else:
            missing_units.append(str(spec["label"]))
        values[field] = value

    if not values and not errors:
        errors.append("No supported lab columns were found. Download the SLCE template and keep its column names.")
    if missing_units:
        warnings.append(
            "Units were not supplied for " + ", ".join(missing_units) + "; SLCE assumed the template units."
        )
    return LabImportResult(values, errors, warnings)
