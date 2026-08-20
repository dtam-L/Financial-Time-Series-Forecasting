"""
dashboard/app.py
================
Real-time Financial Time Series Dashboard — Streamlit App.

Tabs
----
📊 EDA Live   : Candlestick, Volume, RSI, MACD + live stats
🔮 Prediction : GBM forecast + conformal intervals
📈 Diagnostics: Return distribution, Drawdown, Z-score, Stationarity tests

Chạy local:
    streamlit run dashboard/app.py

Trong Docker:
    Được start bởi docker-compose service 'dashboard'
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

# ─────────────────────────────────────────────────────────────
# Page config (PHẢI gọi trước mọi st.* khác)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial TS Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS — dark theme, premium look
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Import Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
  }
  .main .block-container { padding-top: 1.2rem; max-width: 1400px; }

  /* ── Header ── */
  .dash-header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 60%, #1a1f2e 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .dash-title {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #a371f7, #ff9500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .dash-subtitle { color: #8b949e; font-size: 0.82rem; margin-top: 0.2rem; }
  .live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(63,185,80,0.15); border: 1px solid #3fb950;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; color: #3fb950; font-weight: 600;
  }
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #3fb950;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #58a6ff; }
  .metric-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 1.25rem; font-weight: 600; margin-top: 0.2rem; }
  .metric-up   { color: #3fb950; }
  .metric-down { color: #f85149; }
  .metric-neutral { color: #c9d1d9; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #161b22;
    border-radius: 10px;
    padding: 6px;
    border: 1px solid #30363d;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    color: #8b949e;
    transition: all 0.2s;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1c2333, #21262d) !important;
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label { color: #c9d1d9; font-size: 0.85rem; }

  /* ── Info/Warning boxes ── */
  .stAlert { border-radius: 8px; border: 1px solid #30363d; }

  /* ── Tables ── */
  .stat-table { width: 100%; border-collapse: collapse; }
  .stat-table th { background: #1c2333; color: #58a6ff; padding: 8px 12px; text-align: left; font-size: 0.8rem; }
  .stat-table td { padding: 7px 12px; border-bottom: 1px solid #21262d; font-size: 0.85rem; }
  .stat-table tr:hover td { background: #1c2333; }

  /* ── Forecast table ── */
  .forecast-table { width: 100%; border-collapse: collapse; }
  .forecast-table th { background: #1c2333; color: #a371f7; padding: 8px 12px; font-size: 0.8rem; text-align: right; }
  .forecast-table th:first-child { text-align: left; }
  .forecast-table td { padding: 7px 12px; border-bottom: 1px solid #21262d; font-size: 0.85rem; text-align: right; font-variant-numeric: tabular-nums; }
  .forecast-table td:first-child { text-align: center; color: #8b949e; }
  .forecast-table tr:hover td { background: #1c2333; }
  .up   { color: #3fb950; }
  .down { color: #f85149; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
from dashboard.data_service import (
    load_ohlcv_with_features,
    call_predict_api,
    get_descriptive_stats,
    run_stationarity_tests,
)
from dashboard import charts

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    symbol = st.selectbox(
        "Symbol",
        options=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
        index=0,
    )

    timeframe = st.selectbox(
        "Timeframe",
        options=["1d", "1h", "4h", "1w"],
        index=0,
    )

    n_rows = st.slider("History rows", min_value=50, max_value=500, value=200, step=50)

    st.markdown("---")
    st.markdown("#### 🔮 Prediction")

    forecast_steps = st.slider("Forecast steps", min_value=1, max_value=30, value=7, step=1)

    st.markdown("---")
    st.markdown("#### 🔄 Auto-refresh")

    auto_refresh = st.toggle("Enable auto-refresh", value=False)
    refresh_interval = st.select_slider(
        "Interval (seconds)",
        options=[10, 15, 30, 60, 120, 300],
        value=30,
        disabled=not auto_refresh,
    )

    if st.button("🔄 Refresh now", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e; font-size:0.75rem;'>"
        "📡 Data: PostgreSQL<br>"
        "🤖 Model: GBM Stacking<br>"
        "⏱️ Cache TTL: 30s"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">📈 Financial Time Series Dashboard</div>
    <div class="dash-subtitle">{symbol} · {timeframe} · {n_rows} rows · Updated: {now_utc}</div>
  </div>
  <div class="live-badge">
    <div class="live-dot"></div>
    LIVE
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
with st.spinner("📡 Loading data from PostgreSQL..."):
    df = load_ohlcv_with_features(symbol, timeframe, n_rows)

if df.empty:
    st.error("❌ No data found. Make sure the scheduler has ingested data into the DB.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# KPI cards (row of metrics)
# ─────────────────────────────────────────────────────────────
current_price = df["close"].iloc[-1]
prev_price    = df["close"].iloc[-2] if len(df) > 1 else current_price
change_pct    = (current_price - prev_price) / (prev_price + 1e-8) * 100
high_24h      = df["high"].iloc[-1]
low_24h       = df["low"].iloc[-1]
volume_24h    = df["volume"].iloc[-1]
rsi_now       = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 0.0
macd_now      = df["macd_hist"].iloc[-1] if "macd_hist" in df.columns else 0.0

arrow   = "▲" if change_pct >= 0 else "▼"
chg_cls = "metric-up" if change_pct >= 0 else "metric-down"
rsi_cls = "metric-down" if rsi_now > 70 else ("metric-up" if rsi_now < 30 else "metric-neutral")
macd_cls = "metric-up" if macd_now >= 0 else "metric-down"

cols = st.columns(6)
metrics = [
    ("Price", f"${current_price:,.2f}", chg_cls),
    ("24h Change", f"{arrow} {abs(change_pct):.2f}%", chg_cls),
    ("High", f"${high_24h:,.2f}", "metric-neutral"),
    ("Low", f"${low_24h:,.2f}", "metric-neutral"),
    ("RSI(14)", f"{rsi_now:.1f}", rsi_cls),
    ("MACD Hist", f"{macd_now:+.2f}", macd_cls),
]
for col, (label, val, css) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {css}">{val}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_eda, tab_predict, tab_stats = st.tabs([
    "📊 EDA Live",
    "🔮 Prediction",
    "📈 Diagnostics",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — EDA Live
# ══════════════════════════════════════════════════════════════
with tab_eda:
    # Candlestick
    st.plotly_chart(
        charts.candlestick_chart(df, symbol, timeframe),
        use_container_width=True,
    )

    # Volume + RSI side by side
    col_vol, col_rsi = st.columns(2)
    with col_vol:
        st.plotly_chart(charts.volume_chart(df), use_container_width=True)
    with col_rsi:
        st.plotly_chart(charts.rsi_chart(df), use_container_width=True)

    # MACD
    st.plotly_chart(charts.macd_chart(df), use_container_width=True)

    # Summary stats
    with st.expander("📋 Descriptive Statistics", expanded=False):
        stats_dict = get_descriptive_stats(df)
        if stats_dict:
            rows_html = "".join(
                f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                for k, v in stats_dict.items()
            )
            st.markdown(f"""
            <table class="stat-table">
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Prediction
# ══════════════════════════════════════════════════════════════
with tab_predict:
    with st.spinner(f"🤖 Calling GBM Prediction API ({forecast_steps} steps)..."):
        pred_result = call_predict_api(symbol, timeframe, forecast_steps)

    if "error" in pred_result:
        st.warning(pred_result["error"])
        st.info(
            "💡 **Tip:** Bạn cần train GBM model trước:\n"
            "```\ndocker compose --profile training run --rm trainer_gbm\n```\n"
            "Sau đó reload model:\n"
            "```\ncurl -X POST http://localhost:8000/models/reload?model=gbm\n```"
        )
        forecast_steps_data = []
    else:
        forecast_steps_data = pred_result.get("forecast", [])

    # Forecast chart
    st.plotly_chart(
        charts.forecast_chart(df, forecast_steps_data, symbol, timeframe),
        use_container_width=True,
    )

    # Forecast table
    if forecast_steps_data:
        last_close = df["close"].iloc[-1]
        st.markdown("#### 📋 Forecast Details")

        rows_html = ""
        for f in forecast_steps_data:
            y = f["y_pred"]
            diff = y - last_close
            pct  = diff / last_close * 100
            sign_cls = "up" if diff >= 0 else "down"
            sign     = "+" if diff >= 0 else ""
            rows_html += f"""
            <tr>
              <td>t+{f['step']}</td>
              <td class="{sign_cls}">${y:,.2f}</td>
              <td class="{sign_cls}">{sign}{diff:,.2f} ({sign}{pct:.2f}%)</td>
              <td>${f['lower_80']:,.2f} – ${f['upper_80']:,.2f}</td>
              <td>${f['lower_90']:,.2f} – ${f['upper_90']:,.2f}</td>
            </tr>
            """

        st.markdown(f"""
        <table class="forecast-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Predicted Price</th>
              <th>Change vs Now</th>
              <th>80% Interval</th>
              <th>90% Interval</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        if "n_history_rows" in pred_result:
            st.caption(f"ℹ️ Dự báo dựa trên {pred_result['n_history_rows']} candles | Model: GBM Stacking + Conformal Prediction")

# ══════════════════════════════════════════════════════════════
# TAB 3 — Diagnostics
# ══════════════════════════════════════════════════════════════
with tab_stats:
    col_dist, col_dd = st.columns(2)

    with col_dist:
        st.plotly_chart(charts.return_distribution(df, symbol), use_container_width=True)

    with col_dd:
        st.plotly_chart(charts.drawdown_chart(df), use_container_width=True)

    st.plotly_chart(charts.zscore_chart(df), use_container_width=True)

    # Stationarity tests
    with st.expander("🔬 Stationarity & Hurst Tests", expanded=True):
        with st.spinner("Running ADF, KPSS, Hurst..."):
            test_results = run_stationarity_tests(df)

        if "note" in test_results:
            st.info(test_results["note"])
        elif "error" in test_results:
            st.error(test_results["error"])
        else:
            test_cols = st.columns(3)

            if "ADF" in test_results:
                adf = test_results["ADF"]
                test_cols[0].markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">ADF Test (Log Returns)</div>
                  <div class="metric-value metric-neutral" style="font-size:0.95rem;">
                    Stat: {adf['statistic']}<br>
                    p-value: {adf['p-value']}<br>
                    {adf['result']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            if "KPSS" in test_results:
                kpss = test_results["KPSS"]
                test_cols[1].markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">KPSS Test (Log Returns)</div>
                  <div class="metric-value metric-neutral" style="font-size:0.95rem;">
                    Stat: {kpss['statistic']}<br>
                    p-value: {kpss['p-value']}<br>
                    {kpss['result']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            if "Hurst" in test_results:
                hurst = test_results["Hurst"]
                test_cols[2].markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">Hurst Exponent</div>
                  <div class="metric-value metric-neutral" style="font-size:0.95rem;">
                    H = {hurst['value']}<br>
                    {hurst['interpretation']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Footer + Auto-refresh
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:#8b949e; font-size:0.75rem;'>"
    f"Financial Time Series Dashboard · "
    f"GBM Stacking + Conformal Prediction · "
    f"Data: PostgreSQL · "
    f"Last updated: {now_utc}"
    f"</div>",
    unsafe_allow_html=True,
)

# Auto-refresh logic
if auto_refresh:
    countdown = st.empty()
    for remaining in range(refresh_interval, 0, -1):
        countdown.markdown(
            f"<div style='text-align:center; color:#58a6ff; font-size:0.78rem;'>"
            f"🔄 Auto-refresh in {remaining}s...</div>",
            unsafe_allow_html=True,
        )
        time.sleep(1)
    st.cache_data.clear()
    st.rerun()
