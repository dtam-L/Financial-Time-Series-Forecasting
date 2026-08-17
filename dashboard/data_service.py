"""
dashboard/data_service.py
=========================
Data layer cho Streamlit dashboard.

Functions
---------
load_ohlcv_with_features(symbol, timeframe, n_rows)
    Query PostgreSQL → feature engineering → DataFrame

call_predict_api(symbol, timeframe, steps, api_url, n_history)
    POST /predict/gbm trên FastAPI → dict forecast

get_descriptive_stats(df)
    Descriptive statistics cho log returns

run_stationarity_tests(df)
    ADF + KPSS + Hurst exponent

Caching
-------
Dùng @st.cache_data(ttl=30) để tránh query lại DB mỗi render.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st


# ─────────────────────────────────────────────────────────────
# DB / feature helpers
# ─────────────────────────────────────────────────────────────

def _get_db_url() -> str:
    """Xây DB URL từ env vars (được inject bởi docker-compose)."""
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5432")
    name     = os.getenv("DB_NAME", "financial_ts")
    user     = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "123456")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


@st.cache_data(ttl=30, show_spinner=False)
def load_ohlcv_with_features(
    symbol: str,
    timeframe: str,
    n_rows: int = 200,
) -> pd.DataFrame:
    """
    Query PostgreSQL lấy n_rows candles gần nhất, rồi thêm technical indicators.

    Cache 30 giây để giảm tải DB khi auto-refresh.

    Returns
    -------
    DataFrame với cột: open_time, open, high, low, close, volume,
    + tất cả indicators (rsi_14, macd, bb, atr, ...)
    """
    from sqlalchemy import create_engine, text

    query = text("""
        SELECT symbol, timeframe, open_time,
               open::float, high::float, low::float, close::float, volume::float
        FROM ohlcv_data
        WHERE symbol = :symbol
          AND timeframe = :timeframe
        ORDER BY open_time DESC
        LIMIT :n
    """)

    try:
        engine = create_engine(_get_db_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"symbol": symbol, "timeframe": timeframe, "n": n_rows})
        engine.dispose()
    except Exception as exc:
        st.error(f"❌ DB connection failed: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Sort ascending (oldest → newest)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    df = df.set_index("open_time")

    # Feature engineering
    df = _compute_features(df, timeframe)
    return df


def _compute_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Tính technical indicators directly (không phụ thuộc vào clean_feature_engineering_data)."""
    df = df.copy()

    close = df["close"]

    # Returns
    df["return_pct"]  = close.pct_change() * 100
    df["log_return"]  = np.log(close / close.shift(1))

    # Moving averages
    df["ma_7"]  = close.rolling(7).mean()
    df["ma_25"] = close.rolling(25).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # Bollinger Bands (20, 2σ)
    sma20     = close.rolling(20).mean()
    std20     = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20
    df["bb_pct"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-8)

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    macd_line       = df["ema_12"] - df["ema_26"]
    macd_signal     = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"]      = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_line - macd_signal

    # ATR(14)
    hl = df["high"] - df["low"]
    hc = (df["high"] - close.shift(1)).abs()
    lc = (df["low"]  - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # Volume ratio
    vol_ma = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (vol_ma + 1e-8)

    # Rolling vol & zscore
    df["rolling_vol_30"] = df["return_pct"].rolling(30).std()
    sma30 = close.rolling(30).mean()
    std30 = close.rolling(30).std()
    df["price_zscore"] = (close - sma30) / (std30 + 1e-8)

    # Drawdown
    rolling_max = close.cummax()
    df["drawdown_pct"] = (close - rolling_max) / (rolling_max + 1e-8) * 100

    return df.ffill().fillna(0)


# ─────────────────────────────────────────────────────────────
# Prediction API call
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def call_predict_api(
    symbol: str,
    timeframe: str,
    steps: int = 7,
    n_history: int = 100,
    api_url: Optional[str] = None,
) -> dict:
    """
    Gọi FastAPI POST /predict/gbm.

    Cache 30 giây.

    Returns
    -------
    dict với key 'forecast' là list ForecastStep,
    hoặc {'error': str} nếu thất bại.
    """
    import httpx

    base_url = api_url or os.getenv("PREDICT_API_URL", "http://localhost:8000")
    url = f"{base_url}/predict/gbm"

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "steps": steps,
        "from_db": True,
        "n_history": n_history,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 503:
            return {"error": "⚠️ Model chưa được load. Hãy train GBM trước."}
        else:
            return {"error": f"API lỗi {resp.status_code}: {resp.text[:200]}"}
    except httpx.ConnectError:
        return {"error": f"❌ Không kết nối được API tại {base_url}. Đảm bảo service api đang chạy."}
    except Exception as exc:
        return {"error": f"❌ Lỗi: {exc}"}


# ─────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────

def get_descriptive_stats(df: pd.DataFrame) -> dict:
    """Tính descriptive statistics cho log returns."""
    if "log_return" not in df.columns or df["log_return"].dropna().empty:
        return {}

    ret = df["log_return"].dropna()
    close = df["close"]

    return {
        "Số nến":         len(df),
        "Giá hiện tại":   f"${close.iloc[-1]:,.2f}",
        "Giá cao nhất":   f"${close.max():,.2f}",
        "Giá thấp nhất":  f"${close.min():,.2f}",
        "Return TB/ngày": f"{ret.mean():.4f}%",
        "Độ lệch chuẩn":  f"{ret.std():.4f}%",
        "Skewness":       f"{ret.skew():.4f}",
        "Kurtosis":       f"{ret.kurtosis():.4f}",
        "VaR 5%":         f"{ret.quantile(0.05):.4f}%",
        "CVaR 5%":        f"{ret[ret <= ret.quantile(0.05)].mean():.4f}%",
        "Max Drawdown":   f"{df['drawdown_pct'].min():.2f}%",
    }


def run_stationarity_tests(df: pd.DataFrame) -> dict:
    """
    Chạy ADF test, KPSS test và Hurst exponent.

    Returns dict kết quả (có thể empty nếu statsmodels không cài).
    """
    results = {}
    try:
        from statsmodels.tsa.stattools import adfuller, kpss

        price = df["close"].dropna()
        ret   = df["log_return"].dropna()

        # ADF on log returns
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(ret, autolag="AIC")
        results["ADF"] = {
            "statistic": f"{adf_stat:.4f}",
            "p-value":   f"{adf_p:.4f}",
            "result":    "✅ Stationary" if adf_p < 0.05 else "⚠️ Non-stationary",
        }

        # KPSS on log returns
        kpss_stat, kpss_p, _, kpss_crit = kpss(ret, regression="c", nlags="auto")
        results["KPSS"] = {
            "statistic": f"{kpss_stat:.4f}",
            "p-value":   f"{kpss_p:.4f}",
            "result":    "✅ Stationary" if kpss_p > 0.05 else "⚠️ Non-stationary",
        }

    except ImportError:
        results["note"] = "statsmodels không được cài. Chạy: pip install statsmodels"
    except Exception as exc:
        results["error"] = str(exc)

    # Hurst Exponent (không cần statsmodels)
    try:
        h = _hurst_exponent(df["close"].dropna().values)
        interpretation = (
            "Trending (H>0.5)" if h > 0.55
            else "Mean-reverting (H<0.5)" if h < 0.45
            else "Random Walk (H≈0.5)"
        )
        results["Hurst"] = {"value": f"{h:.4f}", "interpretation": interpretation}
    except Exception:
        pass

    return results


def _hurst_exponent(price: np.ndarray, max_lag: int = 100) -> float:
    """Tính Hurst exponent bằng R/S analysis."""
    lags = range(2, min(max_lag, len(price) // 2))
    rs_vals = []
    for lag in lags:
        chunks = [price[i:i+lag] for i in range(0, len(price) - lag, lag)]
        rs = []
        for chunk in chunks:
            mean = chunk.mean()
            deviation = chunk - mean
            cum_dev = np.cumsum(deviation)
            r = cum_dev.max() - cum_dev.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs.append(r / s)
        if rs:
            rs_vals.append((lag, np.mean(rs)))
    if len(rs_vals) < 5:
        return 0.5
    log_lags = np.log([x[0] for x in rs_vals])
    log_rs   = np.log([x[1] for x in rs_vals])
    hurst    = np.polyfit(log_lags, log_rs, 1)[0]
    return float(np.clip(hurst, 0.0, 1.0))
