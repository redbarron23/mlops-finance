"""Fetch OHLCV data from yfinance and save as parquet."""
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

DATA_DIR = "data/raw"
TICKERS = ["GOOG", "NVDA", "ORCL", "MSFT"]
START_DATE = "2018-01-01"
END_DATE = None  # defaults to today


def fetch_stock_data(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str | None = END_DATE,
    data_dir: str = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for given tickers and save to parquet.
    Returns dict of {ticker: DataFrame}.
    """
    os.makedirs(data_dir, exist_ok=True)
    result = {}

    for ticker in tickers:
        print(f"Fetching {ticker} from {start} to {end or 'today'}...")
        df = yf.download(ticker, start=start, end=end, progress=False)

        # Handle MultiIndex columns (when downloading multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = df.columns.str.lower()
        df["ticker"] = ticker
        df["date"] = df.index
        df = df.reset_index(drop=True)

        path = os.path.join(data_dir, f"{ticker.lower()}.parquet")
        df.to_parquet(path, index=False)
        print(f"  Saved to {path} ({len(df)} rows)")
        result[ticker] = df

    return result


def load_stock_data(ticker: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load previously fetched OHLCV data for a ticker."""
    path = os.path.join(data_dir, f"{ticker.lower()}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data found for {ticker} at {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    fetch_stock_data()
