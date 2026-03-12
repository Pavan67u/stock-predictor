# ================================================================
# utils/feature_engineer.py — Technical Indicator Computation
# ================================================================

import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and add them as new columns.

    Parameters:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional technical indicator columns
    """
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    # ---- 1. Simple Moving Averages (SMA) ----
    data["SMA_20"] = close.rolling(window=20).mean()
    data["SMA_50"] = close.rolling(window=50).mean()
    data["SMA_200"] = close.rolling(window=200).mean()

    # ---- 2. Exponential Moving Averages (EMA) ----
    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # ---- 3. Relative Strength Index (RSI) ----
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # ---- 4. MACD ----
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]

    # ---- 5. Bollinger Bands ----
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    data["BB_Upper"] = sma_20 + (std_20 * 2)
    data["BB_Lower"] = sma_20 - (std_20 * 2)
    data["BB_Width"] = data["BB_Upper"] - data["BB_Lower"]

    # ---- 6. Price Momentum ----
    data["Momentum_10"] = close - close.shift(10)
    data["Momentum_30"] = close - close.shift(30)

    # ---- 7. Volatility (20-day rolling std of returns) ----
    data["Daily_Return"] = close.pct_change()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()

    # ---- 8. Additional useful features ----
    data["Price_Range"] = high - low
    data["Price_Change"] = close - close.shift(1)
    data["Volume_SMA_20"] = data["Volume"].rolling(20).mean()

    return data
