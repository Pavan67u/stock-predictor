# ================================================================
# backend/models.py — Pydantic Request/Response Schemas
# ================================================================

from pydantic import BaseModel
from typing import Optional, List


class PredictionRequest(BaseModel):
    """Request schema for stock price prediction."""
    ticker: str = "AAPL"
    days: Optional[int] = 1  # Number of days to predict


class PredictionResponse(BaseModel):
    """Response schema for single-day prediction."""
    ticker: str
    current_price: float
    predicted_price: float
    price_change_pct: float
    signal: str
    rsi: float
    timestamp: str


class TrendResponse(BaseModel):
    """Response schema for multi-day trend prediction."""
    ticker: str
    current_price: float
    forecast: List[dict]
    timestamp: str


class StockDataResponse(BaseModel):
    """Response schema for historical stock data."""
    ticker: str
    data: List[dict]
    total_records: int


class IndicatorResponse(BaseModel):
    """Response schema for technical indicators."""
    ticker: str
    indicators: dict
    timestamp: str


class HealthResponse(BaseModel):
    """Response schema for API health check."""
    status: str
    message: str
