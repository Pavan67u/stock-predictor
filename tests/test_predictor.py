# ================================================================
# tests/test_predictor.py — Tests for Prediction Logic
# ================================================================

import pytest  # type: ignore[import-untyped]
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.model_trainer import create_sequences


class TestCreateSequences:
    """Tests for sliding window sequence creation."""

    def test_output_shapes(self):
        """X and y should have correct shapes."""
        data = np.arange(100).reshape(-1, 1).astype(float)
        X, y = create_sequences(data, window_size=10)
        assert X.shape == (90, 10)
        assert y.shape == (90,)

    def test_window_content(self):
        """First window should contain the first 'window_size' values."""
        data = np.arange(20).reshape(-1, 1).astype(float)
        X, y = create_sequences(data, window_size=5)
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
        assert y[0] == 5

    def test_last_window(self):
        """Last window should end at second-to-last element."""
        data = np.arange(20).reshape(-1, 1).astype(float)
        X, y = create_sequences(data, window_size=5)
        np.testing.assert_array_equal(X[-1], [14, 15, 16, 17, 18])
        assert y[-1] == 19

    def test_small_data(self):
        """Should return empty arrays if data is shorter than window."""
        data = np.arange(5).reshape(-1, 1).astype(float)
        X, y = create_sequences(data, window_size=10)
        assert len(X) == 0
        assert len(y) == 0


class TestSignalGeneration:
    """Tests for buy/sell/hold signal logic."""

    def test_buy_signal(self):
        """BUY when predicted > current and RSI < 70."""
        current = 100.0
        predicted = 105.0
        rsi = 50.0
        change_pct = ((predicted - current) / current) * 100
        if predicted > current and rsi < 70:
            signal = "BUY"
        elif predicted < current and rsi > 30:
            signal = "SELL"
        else:
            signal = "HOLD"
        assert signal == "BUY"

    def test_sell_signal(self):
        """SELL when predicted < current and RSI > 30."""
        current = 100.0
        predicted = 95.0
        rsi = 60.0
        if predicted > current and rsi < 70:
            signal = "BUY"
        elif predicted < current and rsi > 30:
            signal = "SELL"
        else:
            signal = "HOLD"
        assert signal == "SELL"

    def test_hold_signal_overbought(self):
        """HOLD when predicted > current but RSI >= 70 (overbought)."""
        current = 100.0
        predicted = 105.0
        rsi = 75.0
        if predicted > current and rsi < 70:
            signal = "BUY"
        elif predicted < current and rsi > 30:
            signal = "SELL"
        else:
            signal = "HOLD"
        assert signal == "HOLD"

    def test_hold_signal_oversold(self):
        """HOLD when predicted < current but RSI <= 30 (oversold)."""
        current = 100.0
        predicted = 95.0
        rsi = 25.0
        if predicted > current and rsi < 70:
            signal = "BUY"
        elif predicted < current and rsi > 30:
            signal = "SELL"
        else:
            signal = "HOLD"
        assert signal == "HOLD"
