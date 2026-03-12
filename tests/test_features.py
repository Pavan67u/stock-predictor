# ================================================================
# tests/test_features.py — Tests for Feature Engineering
# ================================================================

import pytest  # type: ignore[import-untyped]
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.feature_engineer import compute_technical_indicators


@pytest.fixture
def sample_ohlcv():
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 250  # ~1 year of trading days
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 150 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "Open": close - np.random.rand(n),
        "High": close + np.abs(np.random.randn(n)),
        "Low": close - np.abs(np.random.randn(n)),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 50_000_000, n),
    }, index=dates)


class TestComputeTechnicalIndicators:
    """Tests for the compute_technical_indicators function."""

    def test_returns_dataframe(self, sample_ohlcv):
        """Should return a DataFrame."""
        result = compute_technical_indicators(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_adds_sma_columns(self, sample_ohlcv):
        """Should add SMA_20, SMA_50, SMA_200."""
        result = compute_technical_indicators(sample_ohlcv)
        for col in ["SMA_20", "SMA_50", "SMA_200"]:
            assert col in result.columns, f"Missing: {col}"

    def test_adds_ema_columns(self, sample_ohlcv):
        """Should add EMA_12 and EMA_26."""
        result = compute_technical_indicators(sample_ohlcv)
        assert "EMA_12" in result.columns
        assert "EMA_26" in result.columns

    def test_adds_rsi(self, sample_ohlcv):
        """Should add RSI column."""
        result = compute_technical_indicators(sample_ohlcv)
        assert "RSI" in result.columns

    def test_rsi_range(self, sample_ohlcv):
        """RSI should be between 0 and 100."""
        result = compute_technical_indicators(sample_ohlcv)
        rsi = result["RSI"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_adds_macd(self, sample_ohlcv):
        """Should add MACD, MACD_Signal, and MACD_Histogram."""
        result = compute_technical_indicators(sample_ohlcv)
        for col in ["MACD", "MACD_Signal", "MACD_Histogram"]:
            assert col in result.columns, f"Missing: {col}"

    def test_adds_bollinger_bands(self, sample_ohlcv):
        """Should add BB_Upper, BB_Lower, BB_Width."""
        result = compute_technical_indicators(sample_ohlcv)
        for col in ["BB_Upper", "BB_Lower", "BB_Width"]:
            assert col in result.columns, f"Missing: {col}"

    def test_bollinger_band_order(self, sample_ohlcv):
        """Upper band should always be above lower band."""
        result = compute_technical_indicators(sample_ohlcv).dropna()
        assert (result["BB_Upper"] >= result["BB_Lower"]).all()

    def test_adds_momentum(self, sample_ohlcv):
        """Should add Momentum_10 and Momentum_30."""
        result = compute_technical_indicators(sample_ohlcv)
        assert "Momentum_10" in result.columns
        assert "Momentum_30" in result.columns

    def test_adds_volatility(self, sample_ohlcv):
        """Should add Volatility_20 column."""
        result = compute_technical_indicators(sample_ohlcv)
        assert "Volatility_20" in result.columns

    def test_does_not_modify_original(self, sample_ohlcv):
        """Should not modify the input DataFrame."""
        original_cols = list(sample_ohlcv.columns)
        compute_technical_indicators(sample_ohlcv)
        assert list(sample_ohlcv.columns) == original_cols
