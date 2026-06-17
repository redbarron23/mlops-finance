"""Simulate prediction traffic by replaying historical data through the API."""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetch import load_stock_data
from src.data.features import compute_features

API_URL = "http://localhost:8000"


async def simulate_traffic(
    tickers: list[str] = ["GOOG", "NVDA", "ORCL", "MSFT"],
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    delay_ms: int = 100,
):
    """
    Replay historical data through /predict endpoint.

    Simulates real-time traffic by making requests for each historical date.

    Args:
      tickers: list of ticker symbols
      start_date: start date for replay (inclusive)
      end_date: end date for replay (inclusive)
      delay_ms: delay between requests (ms)
    """
    print(f"Starting traffic simulation: {start_date} to {end_date}")
    print(f"API: {API_URL}")

    all_dfs = []
    for ticker in tickers:
        df = load_stock_data(ticker)
        df = compute_features(df)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    feature_cols = [
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

    async with httpx.AsyncClient() as client:
        for i, row in combined.iterrows():
            payload = {
                col: float(row[col]) for col in feature_cols
            }
            payload["ticker"] = row["ticker"]
            payload["timestamp"] = str(row["date"])

            try:
                resp = await client.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=5,
                )
                if resp.status_code == 200:
                    pred = resp.json()["prediction"]
                    if (i + 1) % 100 == 0:
                        print(
                            f"  [{i+1}/{len(combined)}] "
                            f"{row['date']} {row['ticker']}: pred={pred:.4f}"
                        )
                else:
                    print(f"  Error: {resp.status_code} - {resp.text}")
            except Exception as e:
                print(f"  Request failed: {e}")

            await asyncio.sleep(delay_ms / 1000.0)

    print(f"\n✓ Simulated {len(combined)} predictions")


if __name__ == "__main__":
    asyncio.run(
        simulate_traffic(
            start_date="2022-01-01",
            end_date="2024-06-30",
            delay_ms=50,
        )
    )
