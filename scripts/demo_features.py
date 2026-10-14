"""Deterministic local feature/drift demo; no services or external data needed."""

import numpy as np
import pandas as pd

from src.data.features import compute_features, get_feature_columns
from src.monitoring.drift import compute_drift_score


def main():
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 500)))
    frame = pd.DataFrame({
        "date": pd.bdate_range("2023-01-01", periods=len(prices)),
        "close": prices,
    })
    features = compute_features(frame)
    columns = get_feature_columns()
    reference = features[columns]
    shifted = reference.copy()
    for column in columns:
        shifted[column] += 5 * reference[column].std()
    unchanged_score = compute_drift_score(reference, reference, columns)
    shifted_score = compute_drift_score(reference, shifted, columns)
    assert not reference.isna().any().any()
    assert unchanged_score == 0.0 and shifted_score == 1.0
    print("MLOps finance | local synthetic demo")
    print(f"Generated price rows: {len(frame)}")
    print(f"Feature rows: {len(features)} | model inputs: {len(columns)}")
    print(f"Unchanged feature distribution: drift={unchanged_score:.2f}")
    print(f"Deliberately shifted distribution: drift={shifted_score:.2f}")
    print("Feature and drift checks passed. No market data or services used.")


if __name__ == "__main__":
    main()
