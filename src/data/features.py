"""Feature engineering: technical indicators + target label."""
import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute technical indicators and target label.

    Features:
      - SMA_20, SMA_50: simple moving averages
      - RSI_14: relative strength index
      - volatility_20: rolling std of returns
      - returns_lag_1 through returns_lag_5: lagged daily returns

    Target:
      - next_day_return: next day's return (positive/negative for classification or magnitude for regression)

    Args:
      df: OHLCV DataFrame with 'close' column
      lookback: window for computing moving averages and volatility

    Returns:
      DataFrame with features and target; rows where target is NaN are dropped.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"]

    # Moving averages
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()

    # Returns
    df["return"] = close.pct_change()

    # RSI
    df["rsi_14"] = compute_rsi(close, period=14)

    # Volatility
    df["volatility_20"] = df["return"].rolling(20).std()

    # Lagged returns
    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df["return"].shift(lag)

    # Target: next day return (shift by -1 to look forward)
    df["next_day_return"] = df["return"].shift(-1)

    # Drop rows with NaN
    df = df.dropna()

    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute relative strength index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_feature_columns() -> list[str]:
    """Return list of feature column names (for model input)."""
    return [
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


def get_target_column() -> str:
    """Return target column name."""
    return "next_day_return"
