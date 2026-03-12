# ================================================================
# tests/test_data_loader.py — Tests for Data Loading Utilities
# ================================================================

import pytest  # type: ignore[import-untyped]
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import download_stock_data, load_cached_data, clean_data


class TestDownloadStockData:
    """Tests for the download_stock_data function."""

    def test_download_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = download_stock_data("AAPL", start_date="2024-01-01", end_date="2024-02-01")
        assert isinstance(df, pd.DataFrame)

    def test_download_has_required_columns(self):
        """Should contain OHLCV columns."""
        df = download_stock_data("AAPL", start_date="2024-01-01", end_date="2024-02-01")
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_download_not_empty(self):
        """Should return non-empty data for a valid ticker."""
        df = download_stock_data("AAPL", start_date="2024-01-01", end_date="2024-02-01")
        assert len(df) > 0

    def test_download_saves_csv(self, tmp_path):
        """Should save CSV when save_path is provided."""
        save_path = str(tmp_path / "test_data.csv")
        df = download_stock_data("AAPL", start_date="2024-01-01", end_date="2024-02-01", save_path=save_path)
        assert os.path.exists(save_path)


class TestCleanData:
    """Tests for the clean_data function."""

    def test_no_missing_values(self):
        """Cleaned data should have no NaN values."""
        df = pd.DataFrame({
            "Open": [1, np.nan, 3],
            "High": [2, 3, np.nan],
            "Low": [0.5, 1, 2],
            "Close": [1.5, np.nan, 2.5],
            "Volume": [100, 200, 300],
        }, index=pd.date_range("2024-01-01", periods=3))
        cleaned = clean_data(df)
        assert cleaned.isnull().sum().sum() == 0

    def test_chronological_order(self):
        """Cleaned data should be sorted by date."""
        df = pd.DataFrame({
            "Open": [3, 1, 2],
            "High": [4, 2, 3],
            "Low": [2, 0.5, 1],
            "Close": [3.5, 1.5, 2.5],
            "Volume": [300, 100, 200],
        }, index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]))
        cleaned = clean_data(df)
        assert cleaned.index.is_monotonic_increasing


class TestLoadCachedData:
    """Tests for the load_cached_data function."""

    def test_load_nonexistent_raises_error(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_cached_data("/nonexistent/path.csv")

    def test_load_existing_csv(self, tmp_path):
        """Should correctly load a saved CSV."""
        csv_path = str(tmp_path / "cached.csv")
        df = pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        df.to_csv(csv_path)
        loaded = load_cached_data(csv_path)
        assert len(loaded) == 3
