# ================================================================
# frontend/components/sidebar.py — Enhanced Sidebar Configuration
# ================================================================

import streamlit as st
from datetime import datetime


def render_sidebar(default_ticker: str = "AAPL") -> tuple:
    """
    Render the premium sidebar with branded header, styled controls,
    and market status indicator.

    Args:
        default_ticker: The default ticker symbol to pre-fill.

    Returns:
        Tuple of (ticker, period, show_sma, show_bollinger, show_rsi,
                  show_macd, show_volume, forecast_days, algorithm)
    """

    # ---- Branded Header ----
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem;">
        <div style="
            font-size: 2.2rem;
            margin-bottom: 0.2rem;
            filter: drop-shadow(0 0 8px rgba(99,102,241,0.4));
        ">�</div>
        <div style="
            font-size: 1.25rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #22d3ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        ">StockVision AI</div>
        <div style="
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-top: 2px;
        ">ML-Powered Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # ---- Market Status ----
    now = datetime.now()
    hour = now.hour
    is_market_open = 9 <= hour < 16 and now.weekday() < 5

    if is_market_open:
        status_dot = "🟢"
        status_text = "Market Open"
        status_color = "#10b981"
    else:
        status_dot = "🔴"
        status_text = "Market Closed"
        status_color = "#ef4444"

    st.sidebar.markdown(f"""
    <div style="
        display: flex; align-items: center; justify-content: center;
        gap: 8px; padding: 8px 16px;
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 10px; margin-bottom: 1rem;
    ">
        <span style="font-size: 0.7rem;">{status_dot}</span>
        <span style="color: {status_color}; font-size: 0.78rem; font-weight: 600;">{status_text}</span>
        <span style="color: #64748b; font-size: 0.72rem; margin-left: auto;">
            {now.strftime("%H:%M")}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ---- Stock Selection ----
    st.sidebar.markdown("""
    <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;
         text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
        🔍 Stock Selection
    </div>
    """, unsafe_allow_html=True)

    ticker = st.sidebar.text_input(
        "Ticker Symbol",
        value=default_ticker,
        max_chars=10,
        help="Enter any valid stock ticker (e.g., AAPL, MSFT, TSLA, GOOGL)"
    ).upper()

    period = st.sidebar.selectbox(
        "Time Period",
        options=["3mo", "6mo", "1y", "2y", "5y"],
        format_func=lambda x: {"3mo": "3 Months", "6mo": "6 Months", "1y": "1 Year", "2y": "2 Years", "5y": "5 Years"}[x],
        index=2,
        help="Historical data lookback period"
    )

    st.sidebar.markdown("---")

    # ---- Technical Indicators ----
    st.sidebar.markdown("""
    <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;
         text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
        📊 Technical Indicators
    </div>
    """, unsafe_allow_html=True)

    show_sma = st.sidebar.checkbox("Simple Moving Averages (20-Day & 50-Day)", value=True)
    show_bollinger = st.sidebar.checkbox("Bollinger Bands (20-Day, 2 Std Dev)", value=False)
    show_rsi = st.sidebar.checkbox("Relative Strength Index (14-Day)", value=True)
    show_macd = st.sidebar.checkbox("Moving Average Convergence Divergence", value=False)
    show_volume = st.sidebar.checkbox("Trading Volume Bars", value=True)

    st.sidebar.markdown("---")

    # ---- Forecast Settings ----
    st.sidebar.markdown("""
    <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;
         text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
        🔮 AI Forecast Settings
    </div>
    """, unsafe_allow_html=True)

    # Algorithm Selector
    algorithm_options = {
        "lstm": "Long Short-Term Memory (LSTM)",
        "random_forest": "Random Forest Regressor",
        "svr": "Support Vector Regression (SVR)",
    }
    algorithm = st.sidebar.selectbox(
        "Prediction Algorithm",
        options=list(algorithm_options.keys()),
        format_func=lambda x: algorithm_options[x],
        index=0,
        help="Choose the machine learning algorithm to generate predictions"
    )

    # Algorithm description badge
    algo_descriptions = {
        "lstm": ("🧠", "Deep Learning", "3-Layer LSTM Neural Network (128→64→32 neurons, 60-day lookback)"),
        "random_forest": ("🌲", "Ensemble Learning", "100 Decision Trees with max depth 20, bagging strategy"),
        "svr": ("📐", "Kernel Method", "RBF Kernel SVR with C=100, epsilon-tube regression"),
    }
    icon, category, desc = algo_descriptions[algorithm]

    st.sidebar.markdown(f"""
    <div style="
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 10px;
        padding: 10px 12px;
        margin: 6px 0 12px;
    ">
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
            <span style="font-size:0.9rem;">{icon}</span>
            <span style="color:#a855f7; font-size:0.7rem; font-weight:700;
                 text-transform:uppercase; letter-spacing:0.06em;">{category}</span>
        </div>
        <div style="color:#94a3b8; font-size:0.7rem; line-height:1.5;">
            {desc}
        </div>
    </div>
    """, unsafe_allow_html=True)

    forecast_days = st.sidebar.slider(
        "Prediction Horizon (Days)",
        min_value=1, max_value=14, value=7,
        help="Number of days to forecast into the future"
    )

    st.sidebar.markdown("---")

    # ---- Footer Info ----
    st.sidebar.markdown("""
    <div style="
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 0.5rem;
    ">
        <div style="color:#f59e0b; font-size:0.72rem; font-weight:700; margin-bottom:4px;">
            ⚠️ DISCLAIMER
        </div>
        <div style="color:#64748b; font-size:0.68rem; line-height:1.5;">
            Predictions are for <b>educational purposes</b> only.
            This is not financial advice. Always do your own research.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div style="text-align:center; padding:1rem 0 0.5rem; color:#475569; font-size:0.65rem;">
        StockVision AI v1.0 &nbsp;·&nbsp; © {now.year}
    </div>
    """, unsafe_allow_html=True)

    return ticker, period, show_sma, show_bollinger, show_rsi, show_macd, show_volume, forecast_days, algorithm
