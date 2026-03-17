# ================================================================
# backend/app.py — Stock Market Prediction API
# ================================================================
# Run: uvicorn app:app --reload --port 8000
# Docs: http://localhost:8000/docs
# ================================================================

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sqlite3
from pathlib import Path
import os
import hmac
import json
import base64
import hashlib
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model  # type: ignore[import-untyped]
from datetime import datetime, timedelta

# ---- Initialize App ----
app = FastAPI(
    title="Stock Market Prediction API",
    description="ML-powered stock price predictions using LSTM, Random Forest & SVR",
    version="2.0.0",
)

# CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load All Models and Scaler ----
lstm_model = load_model("../models/lstm_model.h5")
rf_model = joblib.load("../models/random_forest.pkl")
svr_model = joblib.load("../models/svr_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
WINDOW_SIZE = 60

# Model registry for easy dispatch
AVAILABLE_MODELS = {
    "lstm": {
        "name": "Long Short-Term Memory (LSTM)",
        "short_name": "LSTM",
        "description": "3-Layer LSTM Neural Network (128→64→32 neurons)",
        "type": "deep_learning",
    },
    "random_forest": {
        "name": "Random Forest Regressor",
        "short_name": "Random Forest",
        "description": "Ensemble of 100 Decision Trees (max_depth=20)",
        "type": "ensemble",
    },
    "svr": {
        "name": "Support Vector Regression (SVR)",
        "short_name": "SVR",
        "description": "RBF Kernel SVR (C=100, epsilon-tube regression)",
        "type": "kernel",
    },
}


# ---- Pydantic Models (Request/Response Schemas) ----
class PredictionRequest(BaseModel):
    ticker: str = "AAPL"
    days: Optional[int] = 1
    algorithm: Optional[str] = "lstm"


class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    price_change_pct: float
    signal: str
    rsi: float
    algorithm: str
    algorithm_name: str
    timestamp: str


class TrendResponse(BaseModel):
    ticker: str
    current_price: float
    forecast: List[dict]
    algorithm: str
    algorithm_name: str
    timestamp: str


class StockDataResponse(BaseModel):
    ticker: str
    data: List[dict]
    total_records: int


class RiskPlanRequest(BaseModel):
    ticker: str = "AAPL"
    account_size: float = 10000.0
    risk_percent: float = 1.0
    stop_loss_atr_multiplier: float = 1.5


class WatchlistRequest(BaseModel):
    user_id: str = "demo"
    ticker: str


class AlertRequest(BaseModel):
    user_id: str = "demo"
    ticker: str
    price: float
    direction: str = "Above"


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


DATA_DIR = Path("../data")
DB_PATH = DATA_DIR / "app_state.db"


def init_state_db() -> None:
    """Initialize persistence tables for watchlist and alerts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, ticker)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            direction TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


init_state_db()


# ---- Helper Functions ----
def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data is None or data.empty:  # type: ignore[union-attr]
            raise ValueError(f"No data found for ticker: {ticker}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data  # type: ignore[return-value]
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


def calculate_rsi(prices, period=14):
    """Calculate RSI from price series."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).iloc[-1]


def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """Average True Range for volatility-aware stop planning."""
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else 0.0


def safe_float(value, default: float = 0.0) -> float:
    """Return finite float for API JSON responses."""
    try:
        val = float(value)
        if np.isfinite(val):
            return val
    except Exception:
        pass
    return default


TOKEN_SECRET = os.getenv("APP_TOKEN_SECRET", "stock-predictor-dev-secret")


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """PBKDF2 password hash as salt$hash (hex)."""
    if salt is None:
        salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    ).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest = stored_hash.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), f"{salt}${digest}")


def create_token(user_id: str, hours: int = 24) -> str:
    payload = {
        "sub": user_id,
        "exp": int((datetime.utcnow() + timedelta(hours=hours)).timestamp()),
    }
    payload_raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_raw).decode("utf-8").rstrip("=")
    sig = hmac.new(TOKEN_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def parse_token(token: str) -> Optional[dict]:
    try:
        payload_b64, sig = token.split(".", 1)
        expected_sig = hmac.new(
            TOKEN_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(datetime.utcnow().timestamp()):
            return None
        return payload
    except Exception:
        return None


def get_authenticated_user(
    authorization: Optional[str], fallback_user_id: str = "demo"
) -> str:
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        payload = parse_token(token)
        if payload and payload.get("sub"):
            return str(payload["sub"])
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return fallback_user_id


def detect_support_resistance(close: pd.Series, lookback: int = 60) -> Dict[str, float]:
    """Simple zone detection from recent rolling extrema."""
    window = close.tail(lookback)
    support = safe_float(window.rolling(10).min().tail(20).median(), safe_float(window.min()))
    resistance = safe_float(window.rolling(10).max().tail(20).median(), safe_float(window.max()))
    return {"support": round(support, 2), "resistance": round(resistance, 2)}


def compute_confluence_score(
    model_signal: str,
    rsi_value: float,
    macd_value: float,
    volume_ratio: float,
    trend_bias: str,
) -> Dict[str, float | str]:
    """Weighted confluence score (0-100) for setup quality."""
    score = 0.0
    if model_signal == "BUY":
        score += 25
    elif model_signal == "HOLD":
        score += 12

    if 45 <= rsi_value <= 65:
        score += 20
    elif 30 <= rsi_value <= 70:
        score += 12

    if macd_value > 0:
        score += 20

    if volume_ratio >= 1.1:
        score += 15
    elif volume_ratio >= 0.9:
        score += 8

    if trend_bias == "bullish":
        score += 20
    elif trend_bias == "sideways":
        score += 10

    score = max(0.0, min(100.0, score))
    return {"score": round(score, 1), "grade": "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"}


def predict_price(prices, algorithm="lstm"):
    """Predict next day price using the selected algorithm."""
    algo = algorithm.lower()
    if algo not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm: {algorithm}. Choose from: {list(AVAILABLE_MODELS.keys())}",
        )

    close_values = prices.values.reshape(-1, 1)  # type: ignore[union-attr]
    scaled = scaler.transform(close_values)
    window = scaled[-WINDOW_SIZE:]

    if algo == "lstm":
        X = window.reshape(1, WINDOW_SIZE, 1)
        pred_scaled = lstm_model.predict(X, verbose=0)
        return float(scaler.inverse_transform(pred_scaled)[0, 0])
    elif algo == "random_forest":
        X = window.flatten().reshape(1, -1)
        pred_scaled = rf_model.predict(X)
        return float(scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])
    else:  # svr
        X = window.flatten().reshape(1, -1)
        pred_scaled = svr_model.predict(X)
        return float(scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])


# ---- API Endpoints ----


@app.get("/")
async def root():
    return {"status": "healthy", "message": "Stock Prediction API is running"}


@app.get("/api/stock/{ticker}", response_model=StockDataResponse)
async def get_stock(ticker: str, period: str = "1y"):
    """Get historical stock data for a given ticker."""
    data = get_stock_data(ticker, period)
    records = data.reset_index().to_dict(orient="records")
    return StockDataResponse(
        ticker=ticker.upper(), data=records, total_records=len(records)
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict next-day stock price using the selected algorithm."""
    algo = (request.algorithm or "lstm").lower()
    if algo not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm: {algo}. Choose from: {list(AVAILABLE_MODELS.keys())}",
        )

    data = get_stock_data(request.ticker, period="6mo")
    close_prices = data["Close"]
    current_price = float(close_prices.iloc[-1])
    predicted = float(predict_price(close_prices, algorithm=algo))
    rsi = float(calculate_rsi(close_prices))
    change_pct = ((predicted - current_price) / current_price) * 100

    if predicted > current_price and rsi < 70:
        signal = "BUY"
    elif predicted < current_price and rsi > 30:
        signal = "SELL"
    else:
        signal = "HOLD"

    return PredictionResponse(
        ticker=request.ticker.upper(),
        current_price=round(current_price, 2),
        predicted_price=round(predicted, 2),
        price_change_pct=round(change_pct, 2),
        signal=signal,
        rsi=round(rsi, 1),
        algorithm=algo,
        algorithm_name=AVAILABLE_MODELS[algo]["name"],
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/predict/trend", response_model=TrendResponse)
async def predict_trend(request: PredictionRequest):
    """Predict multi-day stock price trend using the selected algorithm."""
    algo = (request.algorithm or "lstm").lower()
    if algo not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm: {algo}. Choose from: {list(AVAILABLE_MODELS.keys())}",
        )

    n_days = min(request.days or 7, 30)
    data = get_stock_data(request.ticker, period="6mo")
    close_prices = data["Close"]
    current_price = float(close_prices.iloc[-1])

    close_values = close_prices.values.reshape(-1, 1)  # type: ignore[union-attr]
    scaled = scaler.transform(close_values).flatten()
    window = list(scaled[-WINDOW_SIZE:])
    forecasts = []

    for day in range(1, n_days + 1):
        arr = np.array(window[-WINDOW_SIZE:])
        pred: float = 0.0

        if algo == "lstm":
            X = arr.reshape(1, WINDOW_SIZE, 1)
            pred = float(lstm_model.predict(X, verbose=0)[0, 0])
        elif algo == "random_forest":
            X = arr.reshape(1, -1)
            pred = float(rf_model.predict(X)[0])
        else:  # svr
            X = arr.reshape(1, -1)
            pred = float(svr_model.predict(X)[0])

        price = float(scaler.inverse_transform([[pred]])[0, 0])
        change = ((price - current_price) / current_price) * 100
        forecasts.append(
            {"day": day, "predicted_price": round(price, 2), "change_pct": round(change, 2)}
        )
        window.append(pred)

    return TrendResponse(
        ticker=request.ticker.upper(),
        current_price=round(current_price, 2),
        forecast=forecasts,
        algorithm=algo,
        algorithm_name=AVAILABLE_MODELS[algo]["name"],
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/predict/ensemble")
async def predict_ensemble(request: PredictionRequest):
    """Ensemble prediction combining all 3 models with weighted averaging."""
    data = get_stock_data(request.ticker, period="6mo")
    close_prices = data["Close"]
    current_price = float(close_prices.iloc[-1])

    weights = {"lstm": 0.5, "random_forest": 0.3, "svr": 0.2}
    predictions = {}
    for algo in ["lstm", "random_forest", "svr"]:
        predictions[algo] = float(predict_price(close_prices, algorithm=algo))

    ensemble_price = sum(predictions[a] * weights[a] for a in weights)
    rsi = float(calculate_rsi(close_prices))
    change_pct = ((ensemble_price - current_price) / current_price) * 100

    spread = max(predictions.values()) - min(predictions.values())
    confidence = max(0, min(100, 100 - (spread / current_price * 100) * 10))

    if ensemble_price > current_price and rsi < 70:
        signal = "BUY"
    elif ensemble_price < current_price and rsi > 30:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "ticker": request.ticker.upper(),
        "current_price": round(current_price, 2),
        "ensemble_price": round(ensemble_price, 2),
        "individual_predictions": {k: round(v, 2) for k, v in predictions.items()},
        "weights": weights,
        "confidence_score": round(confidence, 1),
        "price_change_pct": round(change_pct, 2),
        "signal": signal,
        "rsi": round(rsi, 1),
        "spread": round(spread, 2),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/backtest/{ticker}")
async def backtest(ticker: str, algorithm: str = "lstm", days: int = 30):
    """Backtest: simulate what would have happened following model signals."""
    days = min(days, 90)
    data = get_stock_data(ticker, period="1y")
    close = data["Close"]

    if len(close) < WINDOW_SIZE + days + 5:
        raise HTTPException(status_code=400, detail="Not enough historical data for backtesting.")

    results = []
    initial_capital = 10000.0
    capital = initial_capital
    shares = 0.0
    trade_log = []

    for i in range(days, 0, -1):
        idx = len(close) - i
        if idx < WINDOW_SIZE + 1:
            continue

        hist_prices = close.iloc[: idx]
        actual_today = float(close.iloc[idx])
        actual_tomorrow = float(close.iloc[idx + 1]) if idx + 1 < len(close) else actual_today

        try:
            predicted = float(predict_price(hist_prices, algorithm=algorithm))
        except Exception:
            continue

        rsi = float(calculate_rsi(hist_prices))
        change_pct = ((predicted - actual_today) / actual_today) * 100

        if predicted > actual_today and rsi < 70:
            signal = "BUY"
        elif predicted < actual_today and rsi > 30:
            signal = "SELL"
        else:
            signal = "HOLD"

        if signal == "BUY" and capital > 0:
            shares = capital / actual_today
            capital = 0
            trade_log.append({"action": "BUY", "price": actual_today})
        elif signal == "SELL" and shares > 0:
            capital = shares * actual_today
            shares = 0
            trade_log.append({"action": "SELL", "price": actual_today})

        portfolio_value = capital + shares * actual_today

        results.append({
            "day": days - i + 1,
            "date": str(close.index[idx].date()),
            "actual_price": round(actual_today, 2),
            "predicted_price": round(predicted, 2),
            "signal": signal,
            "portfolio_value": round(portfolio_value, 2),
        })

    final_value = capital + shares * float(close.iloc[-1]) if results else initial_capital
    buy_hold_value = initial_capital * float(close.iloc[-1]) / float(close.iloc[-days]) if days < len(close) else initial_capital

    return {
        "ticker": ticker.upper(),
        "algorithm": algorithm,
        "period_days": days,
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "model_return_pct": round((final_value - initial_capital) / initial_capital * 100, 2),
        "buy_hold_value": round(buy_hold_value, 2),
        "buy_hold_return_pct": round((buy_hold_value - initial_capital) / initial_capital * 100, 2),
        "total_trades": len(trade_log),
        "daily_results": results,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/compare")
async def compare_stocks(tickers: str, period: str = "6mo"):
    """Compare multiple stocks. Pass tickers comma-separated, e.g. ?tickers=AAPL,MSFT,GOOGL"""
    ticker_list = [t.strip().upper() for t in tickers.split(",")][:5]
    result = {}
    for t in ticker_list:
        try:
            data = get_stock_data(t, period)
            close = data["Close"]
            first_price = float(close.iloc[0])
            normalized = ((close / first_price) * 100).tolist()
            dates = [str(d.date()) for d in close.index]
            current = float(close.iloc[-1])
            change = ((current - first_price) / first_price) * 100
            result[t] = {
                "dates": dates,
                "normalized": [round(v, 2) for v in normalized],
                "current_price": round(current, 2),
                "period_change_pct": round(change, 2),
            }
        except Exception:
            result[t] = {"error": f"Could not fetch data for {t}"}
    return {"comparison": result, "period": period, "timestamp": datetime.now().isoformat()}


@app.get("/api/sector-heatmap")
async def sector_heatmap():
    """Get sector performance for market overview heatmap."""
    sectors = {
        "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV"],
        "Finance": ["JPM", "BAC", "GS", "V"],
        "Energy": ["XOM", "CVX", "COP", "SLB"],
        "Consumer": ["AMZN", "TSLA", "NKE", "MCD"],
        "Communication": ["META", "NFLX", "DIS", "CMCSA"],
    }
    result = {}
    for sector, tickers in sectors.items():
        sector_changes = []
        stocks = []
        for t in tickers:
            try:
                data = yf.download(t, period="5d", progress=False)
                if data is not None and not data.empty:  # type: ignore[union-attr]
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    c = data["Close"]
                    chg = ((float(c.iloc[-1]) - float(c.iloc[0])) / float(c.iloc[0])) * 100
                    sector_changes.append(chg)
                    stocks.append({"ticker": t, "change_pct": round(chg, 2), "price": round(float(c.iloc[-1]), 2)})
            except Exception:
                continue
        avg_change = sum(sector_changes) / len(sector_changes) if sector_changes else 0
        result[sector] = {"avg_change_pct": round(avg_change, 2), "stocks": stocks}
    return {"sectors": result, "timestamp": datetime.now().isoformat()}


@app.get("/api/news-sentiment/{ticker}")
async def news_sentiment(ticker: str):
    """Get simulated news sentiment for a ticker (uses price momentum as proxy)."""
    data = get_stock_data(ticker, period="1mo")
    close = data["Close"]
    returns = close.pct_change().dropna()

    pos_days = int((returns > 0).sum())
    neg_days = int((returns < 0).sum())
    total = pos_days + neg_days
    bullish_pct = (pos_days / total * 100) if total > 0 else 50
    bearish_pct = (neg_days / total * 100) if total > 0 else 50

    avg_return = float(returns.mean()) * 100
    volatility = float(returns.std()) * 100

    if bullish_pct > 60:
        overall = "Bullish"
    elif bearish_pct > 60:
        overall = "Bearish"
    else:
        overall = "Neutral"

    recent_5d = float(close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0

    headlines = []
    if recent_5d > 2:
        headlines.append({"title": f"{ticker} surges with strong weekly gains", "sentiment": "positive"})
        headlines.append({"title": f"Analysts upgrade {ticker} outlook amid rally", "sentiment": "positive"})
    elif recent_5d < -2:
        headlines.append({"title": f"{ticker} faces selling pressure this week", "sentiment": "negative"})
        headlines.append({"title": f"Market uncertainty weighs on {ticker}", "sentiment": "negative"})
    else:
        headlines.append({"title": f"{ticker} trades sideways in mixed market", "sentiment": "neutral"})
        headlines.append({"title": f"Steady volume for {ticker} amid consolidation", "sentiment": "neutral"})

    headlines.append({"title": f"Technical analysis: {ticker} RSI at {float(calculate_rsi(close)):.0f}", "sentiment": "neutral"})

    return {
        "ticker": ticker.upper(),
        "overall_sentiment": overall,
        "bullish_pct": round(bullish_pct, 1),
        "bearish_pct": round(bearish_pct, 1),
        "avg_daily_return": round(avg_return, 3),
        "volatility": round(volatility, 3),
        "recent_5d_change": round(recent_5d, 2),
        "headlines": headlines,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/feature-importance/{ticker}")
async def feature_importance(ticker: str):
    """Estimate feature importance for predictions."""
    data = get_stock_data(ticker, period="1y")
    close = data["Close"]

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    rsi_vals = 100 - (100 / (1 + (close.diff().where(close.diff() > 0, 0).rolling(14).mean()
                                   / (-close.diff().where(close.diff() < 0, 0)).rolling(14).mean())))
    volume = data["Volume"].astype(float) if "Volume" in data.columns else pd.Series([0.0] * len(close))
    momentum = close.pct_change(5)

    valid = close.iloc[50:].copy()
    features_at = {
        "Price Momentum (5-Day)": float(abs(momentum.iloc[-50:].corr(valid.pct_change().iloc[-50:]))),
        "SMA-20 Trend": float(abs((close - sma_20).iloc[-50:].corr(valid.pct_change().iloc[-50:]))),
        "SMA-50 Trend": float(abs((close - sma_50).iloc[-50:].corr(valid.pct_change().iloc[-50:]))),
        "RSI Signal": float(abs(rsi_vals.iloc[-50:].corr(valid.pct_change().iloc[-50:]))),
        "Volume Pattern": float(abs(volume.iloc[-50:].corr(valid.pct_change().iloc[-50:]))),
        "Daily Volatility": float(abs(close.pct_change().rolling(10).std().iloc[-50:].corr(valid.pct_change().iloc[-50:]))),
    }

    total = sum(v for v in features_at.values() if not pd.isna(v))
    importance = {}
    for k, v in features_at.items():
        val = v if not pd.isna(v) else 0.0
        importance[k] = round((val / total * 100) if total > 0 else 16.67, 1)

    return {
        "ticker": ticker.upper(),
        "features": importance,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get model evaluation metrics."""
    import json

    try:
        with open("../models/metrics.json", "r") as f:
            metrics = json.load(f)
        return {"model": "LSTM (3-layer)", "window_size": WINDOW_SIZE, "metrics": metrics}
    except FileNotFoundError:
        return {
            "model": "LSTM (3-layer)",
            "window_size": WINDOW_SIZE,
            "metrics": "Metrics file not found. Run the training notebook first.",
        }


@app.get("/api/indicators/{ticker}")
async def get_indicators(ticker: str):
    """Get technical indicators for a given ticker."""
    data = get_stock_data(ticker, period="1y")
    close = data["Close"]

    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_50 = float(close.rolling(50).mean().iloc[-1])
    rsi = float(calculate_rsi(close))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = float((ema_12 - ema_26).iloc[-1])

    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = float((bb_sma + 2 * bb_std).iloc[-1])
    bb_lower = float((bb_sma - 2 * bb_std).iloc[-1])

    return {
        "ticker": ticker.upper(),
        "indicators": {
            "SMA_20": round(sma_20, 2),
            "SMA_50": round(sma_50, 2),
            "RSI": round(rsi, 1),
            "MACD": round(macd, 4),
            "Bollinger_Upper": round(bb_upper, 2),
            "Bollinger_Lower": round(bb_lower, 2),
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/models")
async def list_models():
    """List all available prediction algorithms."""
    return {
        "available_models": AVAILABLE_MODELS,
        "default": "lstm",
        "total": len(AVAILABLE_MODELS),
    }


@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    username = request.username.strip().lower()
    if len(username) < 3 or len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Username/password too short")

    conn = get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(request.password), datetime.now().isoformat()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="Username already exists")
    conn.close()

    token = create_token(username)
    return {"ok": True, "username": username, "token": token}


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    username = request.username.strip().lower()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row or not verify_password(request.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(username)
    return {"ok": True, "username": username, "token": token}


@app.get("/api/auth/me")
async def auth_me(authorization: Optional[str] = Header(default=None)):
    user_id = get_authenticated_user(authorization, fallback_user_id="")
    return {"authenticated": bool(user_id), "user_id": user_id}


@app.get("/api/analysis/multi-timeframe/{ticker}")
async def multi_timeframe_analysis(ticker: str):
    """Generate signal context across multiple timeframes."""
    timeframe_map = {
        "15m": "5d",
        "1h": "1mo",
        "4h": "3mo",
        "1d": "1y",
    }
    result = {}
    for tf, period in timeframe_map.items():
        data = get_stock_data(ticker, period=period)
        close = data["Close"]
        sma20 = safe_float(close.rolling(20).mean().iloc[-1], safe_float(close.iloc[-1]))
        sma50 = safe_float(close.rolling(50).mean().iloc[-1], sma20)
        rsi = safe_float(calculate_rsi(close), 50.0)
        bias = "bullish" if sma20 > sma50 else "bearish"
        signal = "BUY" if bias == "bullish" and rsi < 70 else "SELL" if bias == "bearish" and rsi > 30 else "HOLD"
        result[tf] = {
            "bias": bias,
            "signal": signal,
            "rsi": round(rsi, 1),
            "price": round(safe_float(close.iloc[-1]), 2),
        }

    buy_votes = sum(1 for v in result.values() if v["signal"] == "BUY")
    sell_votes = sum(1 for v in result.values() if v["signal"] == "SELL")
    consensus = "BUY" if buy_votes > sell_votes else "SELL" if sell_votes > buy_votes else "HOLD"

    return {
        "ticker": ticker.upper(),
        "timeframes": result,
        "consensus": consensus,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/analysis/structure/{ticker}")
async def structure_analysis(ticker: str):
    """Support/resistance and recent market structure status."""
    data = get_stock_data(ticker, period="6mo")
    close = data["Close"]
    zones = detect_support_resistance(close)
    price = float(close.iloc[-1])

    if price > zones["resistance"]:
        structure = "breakout"
    elif price < zones["support"]:
        structure = "breakdown"
    else:
        structure = "range"

    return {
        "ticker": ticker.upper(),
        "current_price": round(price, 2),
        "support": zones["support"],
        "resistance": zones["resistance"],
        "structure": structure,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analysis/plan")
async def trade_plan(request: RiskPlanRequest):
    """Risk-aware trade plan with entry/SL/TP and position sizing."""
    data = get_stock_data(request.ticker, period="6mo")
    close = data["Close"]
    current_price = safe_float(close.iloc[-1])
    atr = calculate_atr(data)
    predicted = safe_float(predict_price(close, algorithm="lstm"), current_price)

    direction = "long" if predicted >= current_price else "short"
    stop_distance = max(atr * request.stop_loss_atr_multiplier, current_price * 0.005)

    if direction == "long":
        entry = current_price
        stop_loss = current_price - stop_distance
        take_profit_1 = current_price + stop_distance * 1.5
        take_profit_2 = current_price + stop_distance * 2.5
    else:
        entry = current_price
        stop_loss = current_price + stop_distance
        take_profit_1 = current_price - stop_distance * 1.5
        take_profit_2 = current_price - stop_distance * 2.5

    risk_amount = request.account_size * (request.risk_percent / 100)
    per_share_risk = abs(entry - stop_loss)
    position_size = risk_amount / per_share_risk if per_share_risk > 0 else 0
    rr = abs((take_profit_2 - entry) / (entry - stop_loss)) if entry != stop_loss else 0

    return {
        "ticker": request.ticker.upper(),
        "direction": direction,
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit_1": round(take_profit_1, 2),
        "take_profit_2": round(take_profit_2, 2),
        "risk_reward_ratio": round(rr, 2),
        "position_size_shares": int(position_size),
        "capital_at_risk": round(risk_amount, 2),
        "atr": round(atr, 3),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/analysis/confluence/{ticker}")
async def confluence_analysis(ticker: str):
    """Confluence score combining model, indicators, trend and volume."""
    data = get_stock_data(ticker, period="6mo")
    close = data["Close"]
    current_price = safe_float(close.iloc[-1])
    predicted = safe_float(predict_price(close, algorithm="lstm"), current_price)
    signal = "BUY" if predicted > current_price else "SELL" if predicted < current_price else "HOLD"

    rsi_val = safe_float(calculate_rsi(close), 50.0)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_val = safe_float((ema12 - ema26).iloc[-1])
    sma20 = safe_float(close.rolling(20).mean().iloc[-1], current_price)
    sma50 = safe_float(close.rolling(50).mean().iloc[-1], sma20)
    trend_bias = "bullish" if sma20 > sma50 else "bearish"

    volume_ratio = 1.0
    if "Volume" in data.columns:
        vol = data["Volume"]
        avg = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
        volume_ratio = float(vol.iloc[-1]) / avg if avg > 0 else 1.0

    confluence = compute_confluence_score(signal, rsi_val, macd_val, volume_ratio, trend_bias)

    return {
        "ticker": ticker.upper(),
        "signal": signal,
        "trend_bias": trend_bias,
        "rsi": round(rsi_val, 1),
        "macd": round(macd_val, 4),
        "volume_ratio": round(volume_ratio, 2),
        "confluence": confluence,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/agents/summary/{ticker}")
async def agent_summary(ticker: str):
    """Agent-style combined analysis output for decision support."""
    mtf = await multi_timeframe_analysis(ticker)
    structure = await structure_analysis(ticker)
    confluence = await confluence_analysis(ticker)

    final_bias = confluence["signal"]
    if mtf["consensus"] == "HOLD" and confluence["confluence"]["score"] < 60:
        final_bias = "HOLD"

    rationale = [
        f"Consensus timeframe signal: {mtf['consensus']}",
        f"Structure state: {structure['structure']}",
        f"Confluence score: {confluence['confluence']['score']} ({confluence['confluence']['grade']})",
    ]

    return {
        "ticker": ticker.upper(),
        "final_bias": final_bias,
        "rationale": rationale,
        "market_structure_agent": structure,
        "confluence_agent": confluence,
        "timeframe_agent": mtf,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/user/watchlist")
async def get_watchlist(
    user_id: str = "demo", authorization: Optional[str] = Header(default=None)
):
    user_id = get_authenticated_user(authorization, fallback_user_id=user_id)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, created_at FROM watchlist WHERE user_id = ? ORDER BY ticker",
        (user_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"user_id": user_id, "watchlist": rows, "total": len(rows)}


@app.post("/api/user/watchlist")
async def add_watchlist_item(
    request: WatchlistRequest, authorization: Optional[str] = Header(default=None)
):
    user_id = get_authenticated_user(authorization, fallback_user_id=request.user_id)
    ticker = request.ticker.upper().strip()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO watchlist (user_id, ticker, created_at) VALUES (?, ?, ?)",
        (user_id, ticker, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "user_id": user_id, "ticker": ticker}


@app.delete("/api/user/watchlist/{ticker}")
async def delete_watchlist_item(
    ticker: str,
    user_id: str = "demo",
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_authenticated_user(authorization, fallback_user_id=user_id)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
        (user_id, ticker.upper().strip()),
    )
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return {"ok": True, "deleted": deleted}


@app.get("/api/user/alerts")
async def get_alerts(
    user_id: str = "demo", authorization: Optional[str] = Header(default=None)
):
    user_id = get_authenticated_user(authorization, fallback_user_id=user_id)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, ticker, price, direction, created_at FROM alerts WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"user_id": user_id, "alerts": rows, "total": len(rows)}


@app.post("/api/user/alerts")
async def add_alert(
    request: AlertRequest, authorization: Optional[str] = Header(default=None)
):
    user_id = get_authenticated_user(authorization, fallback_user_id=request.user_id)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alerts (user_id, ticker, price, direction, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            user_id,
            request.ticker.upper().strip(),
            safe_float(request.price),
            request.direction,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    alert_id = cur.lastrowid
    conn.close()
    return {"ok": True, "id": alert_id}


@app.delete("/api/user/alerts/{alert_id}")
async def delete_alert(
    alert_id: int,
    user_id: str = "demo",
    authorization: Optional[str] = Header(default=None),
):
    user_id = get_authenticated_user(authorization, fallback_user_id=user_id)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM alerts WHERE id = ? AND user_id = ?", (alert_id, user_id))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return {"ok": True, "deleted": deleted}
