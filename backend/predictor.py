# ================================================================
# backend/predictor.py — Prediction Logic
# ================================================================

import numpy as np
import joblib
from tensorflow.keras.models import load_model  # type: ignore[import-untyped]


class StockPredictor:
    """
    Complete stock prediction system using trained LSTM model.

    Features:
        - Next day price prediction
        - N-day recursive trend prediction
        - Buy/Sell/Hold signal generation
    """

    def __init__(self, model_path: str, scaler_path: str, window_size: int = 60):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.window_size = window_size

    def predict_next_day(self, recent_prices: np.ndarray) -> float:
        """
        Predict tomorrow's closing price.

        Parameters:
            recent_prices: Array of the last 'window_size' closing prices

        Returns:
            Predicted price (float)
        """
        scaled = self.scaler.transform(recent_prices.reshape(-1, 1))
        X = scaled[-self.window_size:].reshape(1, self.window_size, 1)
        scaled_pred = self.model.predict(X, verbose=0)
        return float(self.scaler.inverse_transform(scaled_pred)[0, 0])

    def predict_n_days(self, recent_prices: np.ndarray, n_days: int = 7) -> list:
        """
        Predict next N days using recursive forecasting.

        Parameters:
            recent_prices: Array of the last 'window_size' closing prices
            n_days: Number of days to forecast

        Returns:
            List of predicted prices
        """
        predictions = []
        current_window = self.scaler.transform(recent_prices.reshape(-1, 1)).flatten()
        current_window = list(current_window[-self.window_size:])

        for _ in range(n_days):
            X = np.array(current_window[-self.window_size:]).reshape(1, self.window_size, 1)
            scaled_pred = self.model.predict(X, verbose=0)[0, 0]

            price_pred = self.scaler.inverse_transform([[scaled_pred]])[0, 0]
            predictions.append(float(price_pred))

            current_window.append(scaled_pred)

        return predictions

    def generate_signal(self, current_price: float, predicted_price: float, rsi_value: float) -> dict:
        """
        Generate a Buy/Sell/Hold signal.

        Parameters:
            current_price: Today's closing price
            predicted_price: Model's predicted price for tomorrow
            rsi_value: Current RSI value (0-100)

        Returns:
            dict with signal, confidence, and reasoning
        """
        price_change_pct = ((predicted_price - current_price) / current_price) * 100

        if predicted_price > current_price and rsi_value < 70:
            signal = "BUY"
            reasoning = (
                f"Price predicted to increase by {price_change_pct:+.2f}% "
                f"and RSI ({rsi_value:.1f}) is not overbought"
            )
        elif predicted_price < current_price and rsi_value > 30:
            signal = "SELL"
            reasoning = (
                f"Price predicted to decrease by {price_change_pct:+.2f}% "
                f"and RSI ({rsi_value:.1f}) is not oversold"
            )
        else:
            signal = "HOLD"
            reasoning = f"Mixed signals: price change {price_change_pct:+.2f}%, RSI {rsi_value:.1f}"

        return {
            "signal": signal,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "price_change_pct": price_change_pct,
            "rsi": rsi_value,
            "reasoning": reasoning,
        }
