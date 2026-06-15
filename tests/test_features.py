"""Test feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.data.features import compute_features, compute_rsi


def test_compute_rsi():
    """Test RSI calculation."""
    series = pd.Series([100, 101, 102, 101, 100, 101, 102, 103])
    rsi = compute_rsi(series, period=3)
    assert len(rsi) == len(series)
    assert rsi.iloc[-1] >= 0 and rsi.iloc[-1] <= 100


def test_compute_features():
    """Test feature computation on sample data."""
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=100),
        "close": np.linspace(100, 110, 100) + np.random.normal(0, 0.5, 100),
        "high": np.linspace(101, 111, 100),
        "low": np.linspace(99, 109, 100),
        "volume": np.ones(100) * 1000000,
    })

    features = compute_features(df)

    # Check required columns exist
    assert "sma_20" in features.columns
    assert "sma_50" in features.columns
    assert "rsi_14" in features.columns
    assert "volatility_20" in features.columns
    assert "next_day_return" in features.columns

    # Check no NaN in final df
    assert not features.isna().any().any()

    # Features should have reduced length (due to NaN drop)
    assert len(features) < len(df)
