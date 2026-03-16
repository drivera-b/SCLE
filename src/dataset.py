from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd


UCI_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

TRAINING_COLUMNS = [c for c in UCI_COLUMNS if c != "num"] + ["target"]

UCI_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    path = project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def heart_csv_path() -> Path:
    return data_dir() / "heart.csv"


def demo_sample_path() -> Path:
    return data_dir() / "demo_sample.csv"


def _clean_uci_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = UCI_COLUMNS
    df = df.replace("?", pd.NA)
    for col in UCI_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["num"])
    df["target"] = (df["num"] > 0).astype(int)
    df = df[[c for c in UCI_COLUMNS if c != "num"] + ["target"]]
    return df


def save_dataset_info(info: dict[str, Any]) -> None:
    path = data_dir() / "dataset_info.json"
    try:
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    except OSError:
        pass


def download_uci_heart_dataset(destination: Path | None = None, timeout: int = 10) -> tuple[bool, str]:
    destination = destination or heart_csv_path()
    last_error = "Unknown error"
    for url in UCI_URLS:
        try:
            with urlopen(url, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            rows = [line.strip().split(",") for line in raw.splitlines() if line.strip()]
            df = pd.DataFrame(rows)
            df = _clean_uci_dataframe(df)
            destination.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(destination, index=False)
            save_dataset_info({"source": "UCI Heart Disease (Cleveland)", "url": url, "rows": int(len(df))})
            return True, f"Downloaded UCI dataset from {url}"
        except (URLError, TimeoutError, OSError, ValueError, pd.errors.ParserError) as exc:
            last_error = str(exc)
            continue
    return False, last_error


def load_csv_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in TRAINING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    for col in TRAINING_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "target" in df.columns:
        df["target"] = df["target"].fillna(0).astype(int)
    return df


def load_heart_dataset(
    *,
    try_download: bool = True,
    allow_demo_fallback: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    heart_path = heart_csv_path()
    if heart_path.exists():
        df = load_csv_dataset(heart_path)
        return df, {"source": "local_heart_csv", "path": str(heart_path)}

    if try_download:
        ok, message = download_uci_heart_dataset(heart_path)
        if ok and heart_path.exists():
            df = load_csv_dataset(heart_path)
            return df, {"source": "uci_download", "path": str(heart_path), "message": message}
        download_error = message
    else:
        download_error = "download disabled"

    if allow_demo_fallback and demo_sample_path().exists():
        df = load_csv_dataset(demo_sample_path())
        return df, {
            "source": "demo_sample",
            "path": str(demo_sample_path()),
            "message": f"Using demo_sample.csv because UCI data was unavailable ({download_error}).",
        }

    raise FileNotFoundError(
        "Could not load heart dataset. Place a UCI-format CSV at data/heart.csv or keep data/demo_sample.csv."
    )


def dataset_status_message() -> str:
    heart_path = heart_csv_path()
    if heart_path.exists():
        return f"Using cached dataset at {heart_path.name}."
    if demo_sample_path().exists():
        return "Real dataset not cached yet. App can run in demo mode using data/demo_sample.csv."
    return "No dataset found. The app will attempt download, or place data/heart.csv manually."

