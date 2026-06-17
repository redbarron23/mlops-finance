"""Detect data drift using statistical methods."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def compute_drift_score(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    numerical_features: list[str],
) -> float:
    """
    Compute drift score between reference (training) and production (recent) data.

    Uses Kolmogorov-Smirnov test to detect distribution changes.
    Returns drift score (0-1, higher = more drift).
    """
    n_drifted = 0

    for feature in numerical_features:
        if feature not in reference_df.columns or feature not in production_df.columns:
            continue

        ref_data = reference_df[feature].dropna()
        prod_data = production_df[feature].dropna()

        if len(ref_data) == 0 or len(prod_data) == 0:
            continue

        # Kolmogorov-Smirnov test for distribution change
        ks_stat, p_value = stats.ks_2samp(ref_data, prod_data)

        # Consider drift if p-value < 0.05 (statistically significant)
        if p_value < 0.05:
            n_drifted += 1

    total_features = len(numerical_features)
    drift_score = n_drifted / max(total_features, 1)
    return float(drift_score)


def load_reference_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load reference (training) data for drift comparison."""
    all_dfs = []
    for ticker in ["GOOG", "NVDA", "ORCL", "MSFT"]:
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
