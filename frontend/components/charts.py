# ================================================================
# frontend/components/charts.py — Premium Chart Components
# ================================================================

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ---- Shared Dark Theme Layout ----
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10, 14, 23, 0)",
    plot_bgcolor="rgba(17, 24, 39, 0.5)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#94a3b8", size=11),
    legend=dict(
        bgcolor="rgba(17, 24, 39, 0.6)",
        bordercolor="rgba(99, 102, 241, 0.15)",
        borderwidth=1,
        font=dict(size=11, color="#94a3b8"),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(
        bgcolor="rgba(17, 24, 39, 0.95)",
        bordercolor="rgba(99, 102, 241, 0.3)",
        font_size=12,
        font_color="#f1f5f9",
        font_family="Inter, sans-serif",
    ),
    xaxis=dict(
        gridcolor="rgba(99, 102, 241, 0.06)",
        zerolinecolor="rgba(99, 102, 241, 0.1)",
        showgrid=True,
        rangeslider=dict(visible=False),
    ),
    yaxis=dict(
        gridcolor="rgba(99, 102, 241, 0.06)",
        zerolinecolor="rgba(99, 102, 241, 0.1)",
        showgrid=True,
        side="right",
    ),
)

# Color palette
COLORS = dict(
    candle_up="#10b981",
    candle_down="#ef4444",
    candle_up_fill="rgba(16, 185, 129, 0.85)",
    candle_down_fill="rgba(239, 68, 68, 0.85)",
    sma_20="#6366f1",
    sma_50="#a855f7",
    bb_upper="rgba(34, 211, 238, 0.5)",
    bb_lower="rgba(34, 211, 238, 0.5)",
    bb_fill="rgba(34, 211, 238, 0.04)",
    rsi_line="#a855f7",
    rsi_overbought="rgba(239, 68, 68, 0.4)",
    rsi_oversold="rgba(16, 185, 129, 0.4)",
    macd_line="#6366f1",
    macd_signal="#f59e0b",
    macd_hist_pos="rgba(16, 185, 129, 0.6)",
    macd_hist_neg="rgba(239, 68, 68, 0.6)",
    volume_up="rgba(16, 185, 129, 0.35)",
    volume_down="rgba(239, 68, 68, 0.35)",
    actual="#3b82f6",
    forecast="#f59e0b",
    forecast_fill="rgba(245, 158, 11, 0.08)",
)


def create_main_chart(
    data: pd.DataFrame,
    show_sma: bool = True,
    show_bollinger: bool = False,
    show_rsi: bool = True,
    show_macd: bool = False,
    show_volume: bool = True,
) -> go.Figure:
    """
    Create the main interactive stock chart with optional indicators,
    using a premium dark theme with beautiful color coding.
    """
    # Calculate subplot layout
    sub_panels = []
    if show_volume:
        sub_panels.append("volume")
    if show_rsi:
        sub_panels.append("rsi")
    if show_macd:
        sub_panels.append("macd")

    n_rows = 1 + len(sub_panels)
    main_height = 0.55 if sub_panels else 1.0
    sub_height = (1.0 - main_height) / max(len(sub_panels), 1)
    row_heights = [main_height] + [sub_height] * len(sub_panels)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=row_heights,
    )

    # ---- Candlestick ----
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Open-High-Low-Close",
            increasing=dict(line=dict(color=COLORS["candle_up"], width=1), fillcolor=COLORS["candle_up_fill"]),
            decreasing=dict(line=dict(color=COLORS["candle_down"], width=1), fillcolor=COLORS["candle_down_fill"]),
            whiskerwidth=0.5,
        ),
        row=1, col=1,
    )

    # ---- Moving Averages ----
    if show_sma:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["SMA_20"], name="Simple Moving Avg (20-Day)",
                line=dict(color=COLORS["sma_20"], width=1.5, dash="solid"),
                hovertemplate="Simple Moving Avg 20: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["SMA_50"], name="Simple Moving Avg (50-Day)",
                line=dict(color=COLORS["sma_50"], width=1.5, dash="solid"),
                hovertemplate="Simple Moving Avg 50: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ---- Bollinger Bands ----
    if show_bollinger:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["BB_Upper"], name="Bollinger Band (Upper)",
                line=dict(color=COLORS["bb_upper"], width=1, dash="dot"),
                hovertemplate="Bollinger Upper: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["BB_Lower"], name="Bollinger Band (Lower)",
                line=dict(color=COLORS["bb_lower"], width=1, dash="dot"),
                fill="tonexty",
                fillcolor=COLORS["bb_fill"],
                hovertemplate="Bollinger Lower: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ---- Sub-panels ----
    current_row = 2

    # Volume
    if show_volume and "Volume" in data.columns:
        colors = [
            COLORS["volume_up"] if c >= o else COLORS["volume_down"]
            for c, o in zip(data["Close"], data["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=data.index, y=data["Volume"], name="Trading Volume",
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1, showticklabels=False)
        current_row += 1

    # RSI
    if show_rsi:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["RSI"], name="Relative Strength Index (14-Day)",
                line=dict(color=COLORS["rsi_line"], width=1.5),
                hovertemplate="Relative Strength Index: %{y:.1f}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        # Overbought / Oversold zones
        fig.add_hrect(
            y0=70, y1=100, fillcolor=COLORS["rsi_overbought"], layer="below",
            line_width=0, row=current_row, col=1,  # type: ignore[arg-type]
        )
        fig.add_hrect(
            y0=0, y1=30, fillcolor=COLORS["rsi_oversold"], layer="below",
            line_width=0, row=current_row, col=1,  # type: ignore[arg-type]
        )
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.35)", line_width=1, row=current_row, col=1)  # type: ignore[arg-type]
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,0.35)", line_width=1, row=current_row, col=1)  # type: ignore[arg-type]
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(148,163,184,0.15)", line_width=1, row=current_row, col=1)  # type: ignore[arg-type]
        fig.update_yaxes(range=[0, 100], title_text="Relative Strength Index", row=current_row, col=1)
        current_row += 1

    # MACD
    if show_macd:
        macd_hist = data["MACD"] - data["MACD_Signal"]
        hist_colors = [
            COLORS["macd_hist_pos"] if v >= 0 else COLORS["macd_hist_neg"]
            for v in macd_hist
        ]
        fig.add_trace(
            go.Bar(
                x=data.index, y=macd_hist, name="MACD Histogram",
                marker=dict(color=hist_colors, line=dict(width=0)),
                hovertemplate="Histogram: %{y:.4f}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["MACD"], name="MACD Line",
                line=dict(color=COLORS["macd_line"], width=1.5),
                hovertemplate="MACD Line: %{y:.4f}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["MACD_Signal"], name="Signal Line (9-Day EMA)",
                line=dict(color=COLORS["macd_signal"], width=1.5, dash="dash"),
                hovertemplate="Signal Line: %{y:.4f}<extra></extra>",
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(title_text="MACD (Moving Avg Convergence Divergence)", row=current_row, col=1)

    # ---- Apply dark layout ----
    fig.update_layout(
        **DARK_LAYOUT,  # type: ignore[arg-type]
        height=600 if n_rows <= 2 else 720,
        showlegend=True,
    )

    # Style all Y axes to right side with grid
    for i in range(1, n_rows + 1):
        fig.update_yaxes(
            gridcolor="rgba(99, 102, 241, 0.06)",
            zerolinecolor="rgba(99, 102, 241, 0.1)",
            side="right",
            row=i, col=1,
        )
        fig.update_xaxes(
            gridcolor="rgba(99, 102, 241, 0.06)",
            row=i, col=1,
        )

    return fig


def create_prediction_chart(
    actual_prices: list,
    predicted_prices: list,
    ticker: str = "AAPL",
) -> go.Figure:
    """
    Create a sleek actual vs predicted price comparison chart.
    """
    fig = go.Figure()

    actual_x = list(range(-len(actual_prices), 0))
    forecast_x = list(range(0, len(predicted_prices)))

    # Actual prices line
    fig.add_trace(go.Scatter(
        x=actual_x, y=actual_prices,
        mode="lines",
        name="Recent Prices",
        line=dict(color=COLORS["actual"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.06)",
        hovertemplate="Day %{x}: $%{y:.2f}<extra>Actual</extra>",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_x, y=predicted_prices,
        mode="lines+markers",
        name="AI Forecast",
        line=dict(color=COLORS["forecast"], width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond", color=COLORS["forecast"],
                    line=dict(width=2, color="rgba(245, 158, 11, 0.3)")),
        fill="tozeroy",
        fillcolor=COLORS["forecast_fill"],
        hovertemplate="Day +%{x}: $%{y:.2f}<extra>Forecast</extra>",
    ))

    # Today marker
    fig.add_vline(x=-0.5, line_dash="dot", line_color="rgba(148, 163, 184, 0.3)", line_width=1.5)
    fig.add_annotation(
        x=-0.5, y=max(actual_prices + predicted_prices) * 1.01,
        text="Today", showarrow=False,
        font=dict(size=10, color="#94a3b8"),
        bgcolor="rgba(17, 24, 39, 0.8)",
        bordercolor="rgba(99, 102, 241, 0.2)",
        borderwidth=1,
        borderpad=4,
    )

    fig.update_layout(
        **DARK_LAYOUT,  # type: ignore[arg-type]
        height=380,
        showlegend=True,
        xaxis_title="Days Relative to Today",
        yaxis_title="Price (US Dollars)",
    )

    return fig


def create_trend_chart(forecast_data: list, current_price: float, ticker: str = "AAPL") -> go.Figure:
    """
    Create a multi-day trend forecast chart with confidence visual.
    """
    days = [f["day"] for f in forecast_data]
    prices = [f["predicted_price"] for f in forecast_data]
    changes = [f["change_pct"] for f in forecast_data]

    # Add today as day 0
    all_days = [0] + days
    all_prices = [current_price] + prices

    fig = go.Figure()

    # Price trend fill
    fill_color = "rgba(16, 185, 129, 0.06)" if prices[-1] > current_price else "rgba(239, 68, 68, 0.06)"
    line_color = COLORS["candle_up"] if prices[-1] > current_price else COLORS["candle_down"]

    fig.add_trace(go.Scatter(
        x=all_days, y=all_prices,
        mode="lines+markers",
        name="Forecast Trend",
        line=dict(color=line_color, width=3, shape="spline"),
        marker=dict(size=8, color=line_color, line=dict(width=2, color="rgba(255,255,255,0.2)")),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate="Day +%{x}<br>$%{y:.2f}<extra></extra>",
    ))

    # Current price reference line
    fig.add_hline(
        y=current_price, line_dash="dash",
        line_color="rgba(148, 163, 184, 0.3)", line_width=1,
    )
    fig.add_annotation(
        x=max(days), y=current_price,
        text=f"Current: ${current_price:.2f}",
        showarrow=False,
        font=dict(size=10, color="#94a3b8"),
        bgcolor="rgba(17, 24, 39, 0.8)",
        bordercolor="rgba(99, 102, 241, 0.2)",
        borderwidth=1,
        borderpad=4,
        xanchor="right",
    )

    fig.update_layout(
        **DARK_LAYOUT,  # type: ignore[arg-type]
        height=380,
        showlegend=False,
        xaxis_title="Days from Today",
        yaxis_title="Predicted Price (US Dollars)",
        xaxis_dtick=1,
    )

    return fig


def create_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 100) -> go.Figure:
    """
    Create a modern gauge/indicator chart.
    """
    if value > 70:
        bar_color = COLORS["candle_down"]
    elif value < 30:
        bar_color = COLORS["candle_up"]
    else:
        bar_color = COLORS["sma_20"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=13, color="#94a3b8")),
        number=dict(font=dict(size=28, color="#f1f5f9"), suffix=""),
        gauge=dict(
            axis=dict(range=[min_val, max_val], tickcolor="rgba(148,163,184,0.3)",
                      tickwidth=1, tickfont=dict(size=9, color="#64748b")),
            bar=dict(color=bar_color, thickness=0.75),
            bgcolor="rgba(17, 24, 39, 0.5)",
            borderwidth=1,
            bordercolor="rgba(99, 102, 241, 0.15)",
            steps=[
                dict(range=[0, 30], color="rgba(16, 185, 129, 0.08)"),
                dict(range=[30, 70], color="rgba(99, 102, 241, 0.05)"),
                dict(range=[70, 100], color="rgba(239, 68, 68, 0.08)"),
            ],
            threshold=dict(
                line=dict(color="#f1f5f9", width=2),
                thickness=0.8,
                value=value,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )

    return fig
