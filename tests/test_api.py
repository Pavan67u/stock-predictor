# ================================================================
# tests/test_api.py — Tests for FastAPI Backend
# ================================================================

import pytest  # type: ignore[import-untyped]
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestAPISchemas:
    """Test Pydantic request/response schemas."""

    def test_prediction_request_defaults(self):
        from backend.models import PredictionRequest
        req = PredictionRequest()
        assert req.ticker == "AAPL"
        assert req.days == 1

    def test_prediction_request_custom(self):
        from backend.models import PredictionRequest
        req = PredictionRequest(ticker="GOOGL", days=7)
        assert req.ticker == "GOOGL"
        assert req.days == 7

    def test_prediction_response_fields(self):
        from backend.models import PredictionResponse
        resp = PredictionResponse(
            ticker="AAPL",
            current_price=150.0,
            predicted_price=155.0,
            price_change_pct=3.33,
            signal="BUY",
            rsi=55.0,
            timestamp="2026-03-09T12:00:00",
        )
        assert resp.ticker == "AAPL"
        assert resp.signal == "BUY"

    def test_stock_data_response(self):
        from backend.models import StockDataResponse
        resp = StockDataResponse(
            ticker="AAPL",
            data=[{"date": "2024-01-01", "close": 150.0}],
            total_records=1,
        )
        assert resp.total_records == 1

    def test_health_response(self):
        from backend.models import HealthResponse
        resp = HealthResponse(status="healthy", message="OK")
        assert resp.status == "healthy"
