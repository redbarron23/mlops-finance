"""Detect data drift using Evidently."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def compute_drift_score(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    numerical_features: list[str],
) -> float:
    """
    Compute drift score between reference (training) and production (recent) data.

    Uses Evidently's DataDriftPreset to compute statistical drift.
    Returns drift score (0-1, higher = more drift).
    """
    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_df[numerical_features],
        current_data=production_df[numerical_features],
    )

    drift_dict = report.as_dict()
    metrics = drift_dict.get("metrics", [])

    if not metrics:
        return 0.0

    drift_metric = metrics[0]
    n_drifted = drift_metric.get("result", {}).get("number_of_drifted_features", 0)
    total_features = len(numerical_features)

    drift_score = n_drifted / max(total_features, 1)
    return float(drift_score)


def load_reference_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load reference (training) data for drift comparison."""
    all_dfs = []
    for ticker in ["SPY", "AAPL", "MSFT"]:
        path = os.path.join(data_dir, f"{ticker.lower()}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df[df["date"] <= "2021-12-31"]
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError("No reference data found")

    return pd.concat(all_dfs, ignore_index=True)


def load_production_data(predictions_dir: str = "data/predictions") -> pd.DataFrame:
    """Load recent production data (logged predictions)."""
    path = Path(predictions_dir) / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No production data found at {path}")
    return pd.read_parquet(path)


def compute_drift_from_logs(
    data_dir: str = "data/raw",
    predictions_dir: str = "data/predictions",
) -> float:
    """
    Compute drift between training distribution and recent predictions.

    Returns drift score (0-1).
    """
    try:
        reference = load_reference_data(data_dir)
        production = load_production_data(predictions_dir)

        numerical_features = [
            "sma_20",
            "sma_50",
            "rsi_14",
            "volatility_20",
            "return_lag_1",
            "return_lag_2",
            "return_lag_3",
            "return_lag_4",
            "return_lag_5",
        ]

        score = compute_drift_score(reference, production, numerical_features)
        return score

    except Exception as e:
        print(f"Warning: drift computation failed: {e}")
        return 0.0
