# ================================================================
# utils/data_loader.py — Data Download and Caching
# ================================================================

import os
import pandas as pd
import yfinance as yf


def download_stock_data(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: str = "2026-03-01",
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Parameters:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        save_path: Optional path to save the downloaded CSV

    Returns:
        DataFrame with OHLCV data indexed by Date
    """
    print(f"📥 Downloading {ticker} data from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df is None or df.empty:  # type: ignore[union-attr]
        raise ValueError(f"No data found for ticker: {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"💾 Saved to {save_path}")

    print(f"✅ Downloaded {len(df)} trading days")
    return df  # type: ignore[return-value]


def load_cached_data(csv_path: str) -> pd.DataFrame:
    """
    Load previously downloaded stock data from a CSV file.

    Parameters:
        csv_path: Path to the saved CSV file

    Returns:
        DataFrame with OHLCV data indexed by Date
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cached data not found at: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"📂 Loaded {len(df)} rows from {csv_path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw stock data: handle missing values, sort, remove anomalies.

    Parameters:
        df: Raw DataFrame with OHLCV data

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Forward fill then backward fill missing values
    df_clean = df_clean.ffill()
    df_clean = df_clean.bfill()

    # Sort by date (chronological order)
    df_clean = df_clean.sort_index()

    # Flag anomalies (>20% daily change)
    daily_returns = df_clean["Close"].pct_change()
    anomalies = daily_returns[abs(daily_returns) > 0.20]
    if len(anomalies) > 0:
        print(f"⚠️  Found {len(anomalies)} days with >20% price change")

    return df_clean
