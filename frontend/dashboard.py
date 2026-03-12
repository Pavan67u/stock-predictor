# ================================================================
# frontend/dashboard.py — StockVision AI Dashboard (Premium)
# ================================================================
# Run: streamlit run dashboard.py
# ================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta

from components.styles import inject_css
from components.charts import (
    create_main_chart,
    create_prediction_chart,
    create_trend_chart,
    create_gauge_chart,
)
from components.sidebar import render_sidebar

# ---- Page Configuration ----
st.set_page_config(
    page_title="StockVision AI — ML Predictions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Inject Custom CSS ----
inject_css()

# ---- Session State Initialization ----
if "page" not in st.session_state:
    st.session_state.page = "start"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"


# ---- Shared Utility ----
@st.cache_data(ttl=300)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


# ================================================================
# START PAGE
# ================================================================
def render_start_page():
    """Render the landing page where users select a stock ticker."""

    # Hide sidebar on start page
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { display: none !important; }
        .stApp > header { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # ---- Vertical spacer to center content ----
    st.markdown("<div style='height: 6vh;'></div>", unsafe_allow_html=True)

    # ---- Logo, Title, Subtitle & Feature Pills (single block) ----
    st.markdown("""
    <div style="text-align:center; animation: fadeInUp 0.8s ease forwards;">
        <div class="start-logo">📊</div>
        <div class="start-title">StockVision AI</div>
        <div class="start-subtitle">
            ML-powered stock market predictions with real-time analytics.<br>
            Choose a stock to begin your analysis.
        </div>
        <div class="start-features">
            <div class="start-feature-pill">🧠 LSTM Neural Network</div>
            <div class="start-feature-pill">🌲 Random Forest</div>
            <div class="start-feature-pill">📐 Support Vector Regression</div>
            <div class="start-feature-pill">📈 Technical Indicators</div>
            <div class="start-feature-pill">🔮 Multi-Day Forecast</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Search Input (Streamlit widgets — must be separate) ----
    spacer_l, center_col, spacer_r = st.columns([1, 2, 1])
    with center_col:
        raw_input = st.text_input(
            "Enter Stock Ticker",
            value=st.session_state.selected_ticker,
            max_chars=10,
            placeholder="e.g. AAPL, TSLA, GOOGL, MSFT",
            label_visibility="collapsed",
        )
        ticker_input = (raw_input or "").upper().strip()

        launch_clicked = st.button(
            "🚀  Launch Dashboard", type="primary", use_container_width=True
        )

    # ---- Popular Stocks Quick-Select ----
    st.markdown("""
    <div style="color:#64748b; font-size:0.78rem; font-weight:500; margin: 1.5rem 0 0.8rem; text-align:center;">
        Or pick a popular stock
    </div>
    """, unsafe_allow_html=True)

    popular_stocks = [
        ("AAPL", "Apple", "🍎"),
        ("MSFT", "Microsoft", "🪟"),
        ("GOOGL", "Alphabet", "🔍"),
        ("AMZN", "Amazon", "📦"),
        ("TSLA", "Tesla", "⚡"),
        ("NVDA", "NVIDIA", "🎮"),
        ("META", "Meta", "👤"),
        ("NFLX", "Netflix", "🎬"),
    ]

    cols = st.columns(len(popular_stocks))
    quick_selected = None
    for i, (sym, name, icon) in enumerate(popular_stocks):
        with cols[i]:
            if st.button(f"{icon} {sym}", key=f"quick_{sym}", use_container_width=True):
                quick_selected = sym

    # ---- Handle Navigation ----
    chosen_ticker = None
    if quick_selected:
        chosen_ticker = quick_selected
    elif launch_clicked and ticker_input:
        chosen_ticker = ticker_input

    if chosen_ticker:
        with st.spinner(f"Validating {chosen_ticker}..."):
            try:
                test = yf.download(chosen_ticker, period="5d", progress=False)
                if test is None or (hasattr(test, 'empty') and test.empty):
                    st.error(
                        f"❌ No data found for **{chosen_ticker}**. "
                        "Please enter a valid stock ticker."
                    )
                    return
            except Exception:
                st.error(
                    f"❌ Could not validate **{chosen_ticker}**. "
                    "Check the symbol and try again."
                )
                return

        st.session_state.selected_ticker = chosen_ticker
        st.session_state.page = "dashboard"
        st.rerun()

    # ---- Footer ----
    st.markdown("""
    <div style="text-align:center; color:#64748b; font-size:0.72rem; margin-top:3rem; opacity:0.7;">
        Powered by TensorFlow, scikit-learn, Plotly & Yahoo Finance &nbsp;·&nbsp;
        For educational purposes only
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# DASHBOARD PAGE
# ================================================================
def render_dashboard():
    """Render the main analytics dashboard."""

    # Show sidebar on dashboard page
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { display: block !important; }
    </style>
    """, unsafe_allow_html=True)

    # ---- Sidebar ----
    (
        ticker_sidebar, period, show_sma, show_bollinger,
        show_rsi, show_macd, show_volume, forecast_days, algorithm,
    ) = render_sidebar(default_ticker=st.session_state.selected_ticker)

    ticker = ticker_sidebar

    # Algorithm display names
    ALGO_DISPLAY = {
        "lstm": ("Long Short-Term Memory (LSTM)", "🧠", "#6366f1"),
        "random_forest": ("Random Forest Regressor", "🌲", "#10b981"),
        "svr": ("Support Vector Regression (SVR)", "📐", "#f59e0b"),
    }
    algo_name, algo_icon, algo_color = ALGO_DISPLAY.get(
        algorithm, ("LSTM", "🧠", "#6366f1")
    )

    # ---- Fetch Data ----
    data = load_data(ticker, period)

    if data.empty:
        st.error(
            f"❌ No data found for ticker **{ticker}**. "
            "Please check the symbol and try again."
        )
        st.stop()

    # ---- Compute Technical Indicators ----
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["EMA_12"] = data["Close"].ewm(span=12).mean()
    data["EMA_26"] = data["Close"].ewm(span=26).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    bb_sma = data["Close"].rolling(20).mean()
    bb_std = data["Close"].rolling(20).std()
    data["BB_Upper"] = bb_sma + 2 * bb_std
    data["BB_Lower"] = bb_sma - 2 * bb_std

    # ---- Key Metrics ----
    current_price = float(data["Close"].iloc[-1])
    prev_price = float(data["Close"].iloc[-2])
    price_change = current_price - prev_price
    change_pct = (price_change / prev_price) * 100
    day_high = float(data["High"].iloc[-1])
    day_low = float(data["Low"].iloc[-1])
    current_rsi = float(data["RSI"].iloc[-1])
    current_volume = (
        int(data["Volume"].iloc[-1]) if "Volume" in data.columns else 0
    )
    avg_volume = (
        int(data["Volume"].rolling(20).mean().iloc[-1])
        if "Volume" in data.columns
        else 0
    )
    sma_20_val = (
        float(data["SMA_20"].iloc[-1])
        if not pd.isna(data["SMA_20"].iloc[-1])
        else 0
    )
    sma_50_val = (
        float(data["SMA_50"].iloc[-1])
        if not pd.isna(data["SMA_50"].iloc[-1])
        else 0
    )

    # ===== HERO HEADER =====
    change_icon = "▲" if change_pct >= 0 else "▼"
    change_color = "#10b981" if change_pct >= 0 else "#ef4444"
    now = datetime.now()
    period_labels = {
        "3mo": "3 Months", "6mo": "6 Months", "1y": "1 Year",
        "2y": "2 Years", "5y": "5 Years",
    }
    period_display = period_labels.get(period, period.upper())

    # Back button
    if st.button("← Back to Stock Selection", key="back_btn"):
        st.session_state.page = "start"
        st.rerun()

    st.markdown(f"""
    <div class="hero-header">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
            <div>
                <div class="hero-title">{ticker} — ${current_price:.2f}
                    <span style="font-size:1rem; color:{change_color}; font-weight:700;">
                        {change_icon} {abs(change_pct):.2f}%
                    </span>
                </div>
                <div class="hero-subtitle">
                    Real-time analytics powered by {algo_name} &nbsp;·&nbsp;
                    Last updated: {now.strftime("%b %d, %Y %H:%M")}
                </div>
                <div class="hero-status">
                    <div class="dot"></div>
                    Live Data Feed
                </div>
            </div>
            <div style="text-align:right;">
                <div style="color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;">
                    Time Period
                </div>
                <div style="color:#f1f5f9; font-size:1.1rem; font-weight:700;">{period_display}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== METRIC CARDS =====
    def metric_card(label, value, delta_val=None, delta_type="neutral", prefix="", suffix=""):
        delta_html = ""
        if delta_val is not None:
            micon = "↑" if delta_type == "positive" else ("↓" if delta_type == "negative" else "→")
            delta_html = f'<div class="metric-delta {delta_type}">{micon} {delta_val}</div>'
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{prefix}{value}{suffix}</div>
            {delta_html}
        </div>
        """

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        dt = "positive" if change_pct >= 0 else "negative"
        st.markdown(
            metric_card("Current Price", f"{current_price:.2f}", f"{change_pct:+.2f}%", dt, prefix="$"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            metric_card("Day High", f"{day_high:.2f}", None, prefix="$"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            metric_card("Day Low", f"{day_low:.2f}", None, prefix="$"),
            unsafe_allow_html=True,
        )
    with col4:
        rsi_type = "negative" if current_rsi > 70 else ("positive" if current_rsi < 30 else "neutral")
        rsi_label = "Overbought" if current_rsi > 70 else ("Oversold" if current_rsi < 30 else "Neutral")
        st.markdown(
            metric_card("Relative Strength Index (14-Day)", f"{current_rsi:.1f}", rsi_label, rsi_type),
            unsafe_allow_html=True,
        )
    with col5:
        vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        vol_type = "positive" if vol_ratio > 1.2 else ("negative" if vol_ratio < 0.8 else "neutral")
        st.markdown(
            metric_card("Trading Volume", f"{current_volume:,.0f}", f"{vol_ratio:.1f}x average", vol_type),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ===== TABBED CONTENT =====
    tab_chart, tab_prediction, tab_indicators = st.tabs([
        "📈  Price Chart",
        "🔮  AI Prediction",
        "📊  Technical Analysis",
    ])

    # ===== TAB 1: PRICE CHART =====
    with tab_chart:
        fig = create_main_chart(data, show_sma, show_bollinger, show_rsi, show_macd, show_volume)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ===== TAB 2: AI PREDICTION =====
    with tab_prediction:
        st.markdown(f"""
        <div class="section-header">
            <div class="icon">{algo_icon}</div>
            <h3>{algo_name} Forecast</h3>
        </div>
        """, unsafe_allow_html=True)

        # Active algorithm badge
        r = int(algo_color[1:3], 16)
        g = int(algo_color[3:5], 16)
        b = int(algo_color[5:7], 16)
        st.markdown(f"""
        <div style="
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba({r},{g},{b},0.1);
            border: 1px solid {algo_color}33;
            border-radius: 20px; padding: 6px 14px; margin-bottom: 16px;
        ">
            <span style="font-size:0.85rem;">{algo_icon}</span>
            <span style="color:{algo_color}; font-size:0.78rem; font-weight:600;">{algo_name}</span>
        </div>
        """, unsafe_allow_html=True)

        pred_col1, pred_col2 = st.columns([2, 1])

        with pred_col1:
            if st.button("🚀  Generate AI Prediction", type="primary"):
                with st.spinner(f"Running {algo_name}..."):
                    try:
                        trend_resp = requests.post(
                            "http://localhost:8000/api/predict/trend",
                            json={"ticker": ticker, "days": forecast_days, "algorithm": algorithm},
                            timeout=30,
                        )
                        pred_resp = requests.post(
                            "http://localhost:8000/api/predict",
                            json={"ticker": ticker, "days": 1, "algorithm": algorithm},
                            timeout=30,
                        )

                        if trend_resp.status_code == 200 and pred_resp.status_code == 200:
                            st.session_state["prediction"] = pred_resp.json()
                            st.session_state["trend"] = trend_resp.json()
                        else:
                            st.warning("⚠️ API returned an error. Ensure the backend is running on port 8000.")

                    except requests.exceptions.ConnectionError:
                        st.error("🔌 Cannot connect to backend. Start it with: `uvicorn app:app --port 8000`")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Request timed out. The model might be loading.")

        # Display results if available
        if "prediction" in st.session_state and "trend" in st.session_state:
            pred = st.session_state["prediction"]
            trend = st.session_state["trend"]

            signal = pred["signal"]
            signal_class = f"signal-{signal.lower()}"
            signal_icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

            st.markdown(f"""
            <div class="prediction-card" style="margin: 1rem 0;">
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
                    <div>
                        <div style="color:#64748b; font-size:0.72rem; font-weight:600;
                             text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                            AI Signal
                        </div>
                        <span class="signal-badge {signal_class}">
                            {signal_icons.get(signal, "⚪")} {signal}
                        </span>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#64748b; font-size:0.72rem; font-weight:600;
                             text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                            Next-Day Price
                        </div>
                        <div style="color:#f1f5f9; font-size:1.6rem; font-weight:800;">
                            ${pred['predicted_price']:.2f}
                        </div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#64748b; font-size:0.72rem; font-weight:600;
                             text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                            Expected Change
                        </div>
                        <div style="color:{'#10b981' if pred['price_change_pct'] >= 0 else '#ef4444'};
                             font-size:1.6rem; font-weight:800;">
                            {pred['price_change_pct']:+.2f}%
                        </div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#64748b; font-size:0.72rem; font-weight:600;
                             text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">
                            Relative Strength Index
                        </div>
                        <div style="color:#a855f7; font-size:1.6rem; font-weight:800;">
                            {pred['rsi']:.1f}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Trend chart
            st.markdown("""
            <div class="section-header">
                <div class="icon">📉</div>
                <h3>Multi-Day Forecast Trend</h3>
            </div>
            """, unsafe_allow_html=True)

            trend_fig = create_trend_chart(
                trend["forecast"], trend["current_price"], ticker
            )
            st.plotly_chart(trend_fig, width="stretch", config={"displayModeBar": False})

            # Forecast table
            with st.expander("📋 Detailed Day-by-Day Forecast", expanded=False):
                forecast_rows = ""
                for f in trend["forecast"]:
                    chg = f["change_pct"]
                    color = "#10b981" if chg >= 0 else "#ef4444"
                    arrow = "↑" if chg >= 0 else "↓"
                    forecast_rows += f"""
                    <div class="trend-row">
                        <span style="color:#94a3b8; font-weight:600; font-size:0.85rem;">Day +{f['day']}</span>
                        <span style="color:#f1f5f9; font-weight:700; font-size:0.95rem;">
                            ${f['predicted_price']:.2f}
                        </span>
                        <span style="color:{color}; font-weight:600; font-size:0.85rem;">
                            {arrow} {abs(chg):.2f}%
                        </span>
                    </div>
                    """
                st.markdown(f'<div class="glass-panel">{forecast_rows}</div>', unsafe_allow_html=True)

        with pred_col2:
            model_info = {
                "lstm": """
                    <div style="margin-bottom:10px;">
                        <span style="color:#6366f1;">●</span>
                        <b style="color:#f1f5f9;">Architecture:</b> 3-Layer Long Short-Term Memory (LSTM)
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#a855f7;">●</span>
                        <b style="color:#f1f5f9;">Layers:</b> 128 → 64 → 32 neurons
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#22d3ee;">●</span>
                        <b style="color:#f1f5f9;">Window:</b> 60-day lookback period
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#10b981;">●</span>
                        <b style="color:#f1f5f9;">Dropout:</b> 20% per layer (regularization)
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#f59e0b;">●</span>
                        <b style="color:#f1f5f9;">Optimizer:</b> Adam (Adaptive Learning Rate)
                    </div>
                    <div>
                        <span style="color:#ef4444;">●</span>
                        <b style="color:#f1f5f9;">Loss Function:</b> Mean Squared Error (MSE)
                    </div>
                """,
                "random_forest": """
                    <div style="margin-bottom:10px;">
                        <span style="color:#10b981;">●</span>
                        <b style="color:#f1f5f9;">Architecture:</b> Ensemble of Decision Trees
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#22d3ee;">●</span>
                        <b style="color:#f1f5f9;">Number of Trees:</b> 100 estimators
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#a855f7;">●</span>
                        <b style="color:#f1f5f9;">Max Depth:</b> 20 levels per tree
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#f59e0b;">●</span>
                        <b style="color:#f1f5f9;">Strategy:</b> Bagging (Bootstrap Aggregation)
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#6366f1;">●</span>
                        <b style="color:#f1f5f9;">Input:</b> 60-day price window (flattened)
                    </div>
                    <div>
                        <span style="color:#ef4444;">●</span>
                        <b style="color:#f1f5f9;">Strength:</b> Handles nonlinear patterns, resistant to overfitting
                    </div>
                """,
                "svr": """
                    <div style="margin-bottom:10px;">
                        <span style="color:#f59e0b;">●</span>
                        <b style="color:#f1f5f9;">Architecture:</b> Support Vector Machine (Regression)
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#a855f7;">●</span>
                        <b style="color:#f1f5f9;">Kernel:</b> Radial Basis Function (RBF)
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#6366f1;">●</span>
                        <b style="color:#f1f5f9;">Regularization (C):</b> 100
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#22d3ee;">●</span>
                        <b style="color:#f1f5f9;">Strategy:</b> Epsilon-tube regression
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#10b981;">●</span>
                        <b style="color:#f1f5f9;">Input:</b> 60-day price window (flattened)
                    </div>
                    <div>
                        <span style="color:#ef4444;">●</span>
                        <b style="color:#f1f5f9;">Strength:</b> Robust to outliers, good generalization
                    </div>
                """,
            }

            info_html = model_info.get(algorithm, model_info["lstm"])

            st.markdown(f"""
            <div class="glass-panel" style="height: 100%;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;
                     text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;">
                    {algo_icon} About the Model — {algo_name}
                </div>
                <div style="color:#64748b; font-size:0.8rem; line-height:1.7;">
                    {info_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ===== TAB 3: TECHNICAL ANALYSIS =====
    with tab_indicators:
        st.markdown("""
        <div class="section-header">
            <div class="icon">📊</div>
            <h3>Technical Indicator Dashboard</h3>
        </div>
        """, unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)

        with g1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            rsi_fig = create_gauge_chart(current_rsi, "Relative Strength Index (14-Day)", 0, 100)
            st.plotly_chart(rsi_fig, width="stretch", config={"displayModeBar": False})
            rsi_status = (
                "🔴 Overbought" if current_rsi > 70
                else ("🟢 Oversold" if current_rsi < 30 else "⚪ Neutral")
            )
            st.markdown(
                f'<div style="text-align:center; color:#94a3b8; font-size:0.8rem;">{rsi_status}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with g2:
            st.markdown(f"""
            <div class="glass-panel" style="padding:1.5rem;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;
                     text-transform:uppercase; letter-spacing:0.08em; margin-bottom:16px;">
                    Moving Averages
                </div>
                <div style="margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="color:#94a3b8; font-size:0.82rem;">Simple Moving Avg (20-Day)</span>
                        <span style="color:#f1f5f9; font-weight:700; font-size:0.95rem;">${sma_20_val:.2f}</span>
                    </div>
                    <div style="height:4px; background:rgba(99,102,241,0.1); border-radius:2px; overflow:hidden;">
                        <div style="height:100%; width:{min(sma_20_val/current_price*100, 100):.0f}%;
                             background:#6366f1; border-radius:2px;"></div>
                    </div>
                </div>
                <div style="margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="color:#94a3b8; font-size:0.82rem;">Simple Moving Avg (50-Day)</span>
                        <span style="color:#f1f5f9; font-weight:700; font-size:0.95rem;">${sma_50_val:.2f}</span>
                    </div>
                    <div style="height:4px; background:rgba(168,85,247,0.1); border-radius:2px; overflow:hidden;">
                        <div style="height:100%; width:{min(sma_50_val/current_price*100, 100):.0f}%;
                             background:#a855f7; border-radius:2px;"></div>
                    </div>
                </div>
                <div style="margin-top:16px; padding-top:12px; border-top:1px solid rgba(99,102,241,0.1);">
                    <div style="color:#94a3b8; font-size:0.75rem;">Price vs Simple Moving Avg (20-Day)</div>
                    <div style="color:{'#10b981' if current_price > sma_20_val else '#ef4444'};
                         font-size:1.1rem; font-weight:700;">
                        {'▲ Above' if current_price > sma_20_val else '▼ Below'}
                        ({abs((current_price - sma_20_val) / sma_20_val * 100):.2f}%)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with g3:
            macd_val = float(data["MACD"].iloc[-1])
            macd_signal_val = float(data["MACD_Signal"].iloc[-1])
            macd_hist = macd_val - macd_signal_val
            bb_upper_val = float(data["BB_Upper"].iloc[-1]) if not pd.isna(data["BB_Upper"].iloc[-1]) else 0
            bb_lower_val = float(data["BB_Lower"].iloc[-1]) if not pd.isna(data["BB_Lower"].iloc[-1]) else 0

            st.markdown(f"""
            <div class="glass-panel" style="padding:1.5rem;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;
                     text-transform:uppercase; letter-spacing:0.08em; margin-bottom:16px;">
                    Key Indicators
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                    <div style="background:rgba(99,102,241,0.06); border-radius:10px; padding:12px; text-align:center;">
                        <div style="color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em;">
                            MACD Line
                        </div>
                        <div style="color:{'#10b981' if macd_val > 0 else '#ef4444'};
                             font-size:1.1rem; font-weight:700;">{macd_val:.4f}</div>
                    </div>
                    <div style="background:rgba(99,102,241,0.06); border-radius:10px; padding:12px; text-align:center;">
                        <div style="color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em;">
                            MACD Signal Line
                        </div>
                        <div style="color:#f59e0b; font-size:1.1rem; font-weight:700;">{macd_signal_val:.4f}</div>
                    </div>
                    <div style="background:rgba(99,102,241,0.06); border-radius:10px; padding:12px; text-align:center;">
                        <div style="color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em;">
                            Bollinger Band (Upper)
                        </div>
                        <div style="color:#22d3ee; font-size:1.1rem; font-weight:700;">${bb_upper_val:.2f}</div>
                    </div>
                    <div style="background:rgba(99,102,241,0.06); border-radius:10px; padding:12px; text-align:center;">
                        <div style="color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em;">
                            Bollinger Band (Lower)
                        </div>
                        <div style="color:#22d3ee; font-size:1.1rem; font-weight:700;">${bb_lower_val:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:12px; padding-top:10px; border-top:1px solid rgba(99,102,241,0.1);">
                    <div style="color:#94a3b8; font-size:0.75rem;">MACD Histogram (MACD − Signal Line)</div>
                    <div style="color:{'#10b981' if macd_hist > 0 else '#ef4444'};
                         font-size:1.1rem; font-weight:700;">
                        {'▲ Bullish' if macd_hist > 0 else '▼ Bearish'} ({macd_hist:.4f})
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ===== FOOTER =====
    st.markdown("""
    <div class="dashboard-footer">
        <div style="margin-bottom:4px;">
            <b style="background: linear-gradient(135deg, #6366f1, #a855f7);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               background-clip: text;">StockVision AI</b>
            &nbsp;·&nbsp; ML-Powered Stock Analytics
        </div>
        <div>
            Built with Streamlit, Plotly, TensorFlow &nbsp;·&nbsp; Data from Yahoo Finance
            &nbsp;·&nbsp; For educational purposes only
        </div>
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# PAGE ROUTER
# ================================================================
if st.session_state.page == "start":
    render_start_page()
else:
    render_dashboard()
