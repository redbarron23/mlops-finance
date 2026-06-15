"""Generate synthetic OHLCV data for demo/testing."""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_synthetic_data(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    initial_price: float = 100.0,
    data_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with realistic drift.

    Simulates market regimes: stable (2018-2021), volatile (2022), recovery (2023-2024).
    """
    os.makedirs(data_dir, exist_ok=True)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.bdate_range(start, end)

    np.random.seed(hash(ticker) % 2**32)

    prices = [initial_price]
    volatility_base = 0.01

    for i, date in enumerate(dates[1:], 1):
        # Increase volatility in 2022 (bear market)
        year = date.year
        if year == 2022:
            vol = volatility_base * 2.5
        elif year == 2023:
            vol = volatility_base * 1.5
        else:
            vol = volatility_base

        # Drift varies by regime
        if year <= 2021:
            drift = 0.0003  # Mild uptrend
        elif year == 2022:
            drift = -0.0005  # Downtrend
        else:
            drift = 0.0005  # Recovery uptrend

        # Random walk with drift
        ret = np.random.normal(drift, vol)
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 10))  # Don't let price go to zero

    prices = np.array(prices)
    highs = prices * (1 + np.abs(np.random.normal(0, volatility_base / 2, len(prices))))
    lows = prices * (1 - np.abs(np.random.normal(0, volatility_base / 2, len(prices))))
    opens = prices * (1 + np.random.normal(0, volatility_base / 3, len(prices)))
    volumes = np.random.uniform(1e6, 5e6, len(prices))

    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": np.maximum(highs, prices),
        "low": np.minimum(lows, prices),
        "close": prices,
        "volume": volumes,
        "ticker": ticker,
    })

    path = os.path.join(data_dir, f"{ticker.lower()}.parquet")
    df.to_parquet(path, index=False)
    print(f"Generated synthetic data for {ticker}: {len(df)} rows → {path}")

    return df


def generate_all_synthetic_data(
    tickers: list[str] = ["GOOG", "NVDA", "ORCL", "MSFT"],
) -> dict[str, pd.DataFrame]:
    """Generate synthetic data for all tickers."""
    # Realistic initial prices for demo
    initial_prices = {"GOOG": 100, "NVDA": 30, "ORCL": 20, "MSFT": 90}

    result = {}
    for ticker in tickers:
        initial_price = initial_prices.get(ticker, 100)
        df = generate_synthetic_data(ticker, initial_price=initial_price)
        result[ticker] = df

    return result


if __name__ == "__main__":
    generate_all_synthetic_data()
