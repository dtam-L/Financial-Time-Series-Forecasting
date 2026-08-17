"""
dashboard/charts.py
===================
Tất cả Plotly charts cho Real-time Financial Dashboard.

Dark theme nhất quán, interactive, đẹp theo chuẩn TradingView-style.

Charts
------
candlestick_chart(df)      — OHLCV + MA7 + MA25 + Bollinger Bands
volume_chart(df)           — Volume bars (xanh up, đỏ down)
rsi_chart(df)              — RSI(14) + overbought/oversold zones
macd_chart(df)             — MACD line + signal + histogram
forecast_chart(df, pred)   — Lịch sử giá + GBM forecast + CI shading
return_distribution(df)    — Histogram log returns
drawdown_chart(df)         — Drawdown curve
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Dark theme tokens ──────────────────────────────────────────────────────────
_BG         = "#0d1117"
_PAPER      = "#0d1117"
_PLOT_BG    = "#161b22"
_GRID       = "#21262d"
_TEXT       = "#c9d1d9"
_MUTED      = "#8b949e"
_BORDER     = "#30363d"
_UP         = "#3fb950"
_DOWN       = "#f85149"
_BLUE       = "#58a6ff"
_ORANGE     = "#ff9500"
_PURPLE     = "#a371f7"
_YELLOW     = "#f0e68c"
_TEAL       = "#39d353"
_WARN       = "#d29922"

_LAYOUT_BASE = dict(
    paper_bgcolor=_PAPER,
    plot_bgcolor=_PLOT_BG,
    font=dict(color=_TEXT, family="Inter, DejaVu Sans, sans-serif", size=12),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, showgrid=True),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, showgrid=True),
    margin=dict(l=60, r=30, t=50, b=40),
    legend=dict(
        bgcolor=_PLOT_BG,
        bordercolor=_BORDER,
        borderwidth=1,
        font=dict(size=11),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    hovermode="x unified",
)


def _base_fig(title: str, height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=_TEXT)), height=height, **_LAYOUT_BASE)
    return fig


# ─────────────────────────────────────────────────────────────
# 1. Candlestick + MA + Bollinger Bands
# ─────────────────────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> go.Figure:
    """
    Interactive candlestick chart với:
    - MA7 (blue), MA25 (orange)
    - Bollinger Bands (purple shading)
    - Hover tooltip đầy đủ OHLCV
    """
    fig = go.Figure()

    # Bollinger Band fill
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"],
            name="BB Upper", line=dict(color=_PURPLE, width=1, dash="dot"),
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"],
            name="BB Lower", line=dict(color=_PURPLE, width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(163, 113, 247, 0.07)",
            showlegend=False,
        ))
        if "bb_mid" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["bb_mid"],
                name="BB Mid", line=dict(color=_PURPLE, width=0.8, dash="dash"),
                showlegend=False,
            ))

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLCV",
        increasing=dict(line=dict(color=_UP), fillcolor=_UP),
        decreasing=dict(line=dict(color=_DOWN), fillcolor=_DOWN),
    ))

    # Moving averages
    if "ma_7" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ma_7"],
            name="MA7", line=dict(color=_BLUE, width=1.5),
        ))
    if "ma_25" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ma_25"],
            name="MA25", line=dict(color=_ORANGE, width=1.5),
        ))

    fig.update_layout(
        title=dict(text=f"📈 {symbol} / {timeframe} — Price", font=dict(size=14, color=_TEXT)),
        height=480,
        **_LAYOUT_BASE,
        xaxis_rangeslider_visible=False,
        yaxis_title="Price (USDT)",
    )
    fig.update_xaxes(gridcolor=_GRID, showgrid=True)
    fig.update_yaxes(gridcolor=_GRID, showgrid=True)
    return fig


# ─────────────────────────────────────────────────────────────
# 2. Volume
# ─────────────────────────────────────────────────────────────

def volume_chart(df: pd.DataFrame) -> go.Figure:
    """Volume bars, xanh khi close >= open, đỏ khi close < open."""
    colors = [_UP if c >= o else _DOWN for c, o in zip(df["close"], df["open"])]

    fig = _base_fig("📊 Volume", height=200)
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.85,
    ))
    if "volume_ratio" in df.columns:
        # Thêm volume ratio line (axis phụ)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["volume_ratio"],
            name="Vol Ratio", line=dict(color=_YELLOW, width=1),
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(
                overlaying="y", side="right",
                showgrid=False,
                tickfont=dict(color=_MUTED, size=9),
                title="Vol Ratio",
            )
        )
    fig.update_layout(yaxis_title="Volume", bargap=0.1)
    return fig


# ─────────────────────────────────────────────────────────────
# 3. RSI
# ─────────────────────────────────────────────────────────────

def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI(14) với overbought(70) và oversold(30) zones."""
    fig = _base_fig("📉 RSI (14)", height=200)

    if "rsi_14" not in df.columns:
        return fig

    rsi = df["rsi_14"]

    # Overbought zone fill
    fig.add_hrect(y0=70, y1=100, fillcolor=_DOWN, opacity=0.07, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=_UP, opacity=0.07, line_width=0)

    # Overbought / oversold lines
    fig.add_hline(y=70, line=dict(color=_DOWN, dash="dot", width=1))
    fig.add_hline(y=30, line=dict(color=_UP, dash="dot", width=1))
    fig.add_hline(y=50, line=dict(color=_MUTED, dash="dot", width=0.7))

    # RSI line with color segments
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi,
        name="RSI(14)",
        line=dict(color=_PURPLE, width=2),
        fill="tozeroy",
        fillcolor="rgba(163,113,247,0.08)",
    ))

    fig.update_layout(
        yaxis=dict(range=[0, 100], tickvals=[0, 30, 50, 70, 100], gridcolor=_GRID),
        yaxis_title="RSI",
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 4. MACD
# ─────────────────────────────────────────────────────────────

def macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD histogram + MACD line + Signal line."""
    fig = _base_fig("📊 MACD (12, 26, 9)", height=220)

    if "macd_hist" not in df.columns:
        return fig

    hist = df["macd_hist"]
    colors = [_UP if v >= 0 else _DOWN for v in hist]

    fig.add_trace(go.Bar(
        x=df.index, y=hist,
        name="Histogram",
        marker_color=colors,
        opacity=0.75,
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"],
        name="MACD", line=dict(color=_BLUE, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_signal"],
        name="Signal", line=dict(color=_ORANGE, width=1.5),
    ))
    fig.add_hline(y=0, line=dict(color=_MUTED, dash="dot", width=0.7))
    fig.update_layout(yaxis_title="MACD", bargap=0.1)
    return fig


# ─────────────────────────────────────────────────────────────
# 5. Forecast chart
# ─────────────────────────────────────────────────────────────

def forecast_chart(
    df: pd.DataFrame,
    forecast: list,
    symbol: str = "",
    timeframe: str = "",
    n_history_display: int = 60,
) -> go.Figure:
    """
    Vẽ lịch sử giá + GBM forecast + conformal intervals.

    Parameters
    ----------
    df       : DataFrame với index = open_time, cột 'close'
    forecast : list of ForecastStep dict từ API response
    n_history_display : số nến lịch sử hiển thị trước forecast
    """
    fig = go.Figure()

    # History price line
    hist = df.tail(n_history_display)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["close"],
        name="Historical Price",
        line=dict(color=_BLUE, width=2),
        mode="lines",
    ))

    if not forecast:
        fig.update_layout(
            title=f"🔮 {symbol} / {timeframe} — Forecast",
            height=480, **_LAYOUT_BASE,
            yaxis_title="Price (USDT)",
        )
        return fig

    # Build forecast timestamps
    last_ts = hist.index[-1]
    tf_delta = _timeframe_to_delta(timeframe)

    steps    = [f["step"] for f in forecast]
    y_pred   = [f["y_pred"] for f in forecast]
    lo80     = [f["lower_80"] for f in forecast]
    hi80     = [f["upper_80"] for f in forecast]
    lo90     = [f["lower_90"] for f in forecast]
    hi90     = [f["upper_90"] for f in forecast]
    future_ts = [last_ts + tf_delta * s for s in steps]

    # Connection line from last known close to first forecast
    fig.add_trace(go.Scatter(
        x=[hist.index[-1], future_ts[0]],
        y=[hist["close"].iloc[-1], y_pred[0]],
        mode="lines",
        line=dict(color=_BLUE, width=1.5, dash="dot"),
        showlegend=False,
    ))

    # CI 90% fill (outer)
    fig.add_trace(go.Scatter(
        x=future_ts + future_ts[::-1],
        y=hi90 + lo90[::-1],
        fill="toself",
        fillcolor="rgba(88,166,255,0.09)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% CI",
        showlegend=True,
    ))

    # CI 80% fill (inner)
    fig.add_trace(go.Scatter(
        x=future_ts + future_ts[::-1],
        y=hi80 + lo80[::-1],
        fill="toself",
        fillcolor="rgba(88,166,255,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI",
        showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_ts, y=y_pred,
        name="GBM Forecast",
        mode="lines+markers",
        line=dict(color=_ORANGE, width=2.5, dash="dash"),
        marker=dict(size=7, color=_ORANGE, symbol="circle-open", line=dict(color=_ORANGE, width=2)),
    ))

    fig.update_layout(
        title=dict(text=f"🔮 {symbol} / {timeframe} — GBM Forecast (+{len(forecast)} steps)", font=dict(size=14, color=_TEXT)),
        height=480, **_LAYOUT_BASE,
        yaxis_title="Price (USDT)",
    )
    return fig


def _timeframe_to_delta(timeframe: str):
    """Chuyển timeframe string thành timedelta."""
    from datetime import timedelta
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }
    return mapping.get(timeframe, timedelta(days=1))


# ─────────────────────────────────────────────────────────────
# 6. Return distribution
# ─────────────────────────────────────────────────────────────

def return_distribution(df: pd.DataFrame, symbol: str = "") -> go.Figure:
    """Histogram + KDE của log returns."""
    fig = _base_fig(f"📊 Log Return Distribution — {symbol}", height=300)

    if "log_return" not in df.columns:
        return fig

    ret = df["log_return"].dropna()

    fig.add_trace(go.Histogram(
        x=ret,
        name="Log Returns",
        nbinsx=60,
        marker_color=_BLUE,
        opacity=0.7,
        histnorm="probability density",
    ))

    # Normal distribution overlay
    mu, sigma = ret.mean(), ret.std()
    x_range = np.linspace(ret.min(), ret.max(), 200)
    normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_pdf,
        name="Normal Dist",
        line=dict(color=_ORANGE, width=2, dash="dash"),
    ))

    # VaR 5% line
    var_5 = ret.quantile(0.05)
    fig.add_vline(x=var_5, line=dict(color=_DOWN, dash="dot", width=1.5),
                  annotation_text=f"VaR 5%: {var_5:.4f}",
                  annotation_font_color=_DOWN)

    fig.update_layout(yaxis_title="Density", xaxis_title="Log Return", barmode="overlay")
    return fig


# ─────────────────────────────────────────────────────────────
# 7. Drawdown chart
# ─────────────────────────────────────────────────────────────

def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Drawdown curve."""
    fig = _base_fig("📉 Drawdown (%)", height=200)

    if "drawdown_pct" not in df.columns:
        return fig

    dd = df["drawdown_pct"]
    fig.add_trace(go.Scatter(
        x=df.index, y=dd,
        name="Drawdown",
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.15)",
        line=dict(color=_DOWN, width=1.5),
    ))
    fig.update_layout(yaxis_title="Drawdown %", showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────
# 8. Price Z-score
# ─────────────────────────────────────────────────────────────

def zscore_chart(df: pd.DataFrame) -> go.Figure:
    """Price Z-score (30-day rolling)."""
    fig = _base_fig("📊 Price Z-score (30d)", height=200)

    if "price_zscore" not in df.columns:
        return fig

    z = df["price_zscore"]
    colors = [_UP if v >= 0 else _DOWN for v in z]

    fig.add_trace(go.Bar(
        x=df.index, y=z,
        name="Z-score",
        marker_color=colors,
        opacity=0.8,
    ))
    fig.add_hline(y=2,  line=dict(color=_DOWN, dash="dot", width=1))
    fig.add_hline(y=-2, line=dict(color=_UP, dash="dot", width=1))
    fig.add_hline(y=0,  line=dict(color=_MUTED, dash="dot", width=0.7))
    fig.update_layout(yaxis_title="Z-score", showlegend=False)
    return fig
