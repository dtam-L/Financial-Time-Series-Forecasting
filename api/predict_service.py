"""
api/predict_service.py
======================
Business logic cho Prediction API.

Luồng xử lý chính
------------------
1. Nhận PredictRequest
2. Lấy dữ liệu:
   a. from_db=True  → query PostgreSQL, lấy n_history candles gần nhất
   b. from_db=False → dùng candles từ request body
3. Chuyển thành DataFrame, chuẩn hoá
4. Gọi GBMForecaster.recursive_predict()
5. Trả về PredictResponse

DB query
--------
Dùng SQLAlchemy engine từ data_collection_api.config (DB_URL),
không tạo thêm connection mới.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

# Đảm bảo project root trong sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.schemas import (
    CandleResponse,
    ForecastStep,
    OHLCVRecord,
    PredictRequest,
    PredictResponse,
)


# ─────────────────────────────────────────────────────────────
# DB helper
# ─────────────────────────────────────────────────────────────

def _get_engine():
    """Lấy SQLAlchemy engine từ config (lazy import để tránh circular)."""
    from sqlalchemy import create_engine
    from data_collection_api import config
    return create_engine(config.DB_URL, pool_pre_ping=True)


def fetch_candles_from_db(
    symbol: str,
    timeframe: str,
    n: int = 100,
) -> pd.DataFrame:
    """
    Query PostgreSQL lấy n candles OHLCV gần nhất cho (symbol, timeframe).

    Parameters
    ----------
    symbol    : ví dụ "BTC/USDT"
    timeframe : ví dụ "1d"
    n         : số candles cần lấy

    Returns
    -------
    DataFrame có cột: open_time, symbol, open, high, low, close, volume
    (sắp xếp tăng dần theo open_time)
    """
    engine = _get_engine()
    query = """
        SELECT
            symbol, timeframe, open_time,
            open, high, low, close, volume
        FROM ohlcv_data
        WHERE symbol = :symbol
          AND timeframe = :timeframe
        ORDER BY open_time DESC
        LIMIT :n
    """
    try:
        df = pd.read_sql(
            query,
            engine,
            params={"symbol": symbol, "timeframe": timeframe, "n": n},
        )
        df = df.sort_values("open_time").reset_index(drop=True)
        logger.info(f"DB → {len(df)} candles [{symbol} {timeframe}]")
        return df
    except Exception as exc:
        logger.error(f"DB query failed: {exc}")
        raise RuntimeError(f"Không thể query DB: {exc}") from exc
    finally:
        engine.dispose()


def fetch_latest_candles(
    symbol: str,
    timeframe: str,
    n: int = 20,
) -> List[CandleResponse]:
    """Trả về n candles gần nhất dưới dạng list CandleResponse (cho endpoint /data/latest)."""
    df = fetch_candles_from_db(symbol, timeframe, n)
    results = []
    for _, row in df.iterrows():
        results.append(CandleResponse(
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            open_time=row["open_time"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        ))
    return results


# ─────────────────────────────────────────────────────────────
# Feature engineering helper
# ─────────────────────────────────────────────────────────────

def _candles_to_df(candles: List[OHLCVRecord], symbol: str) -> pd.DataFrame:
    """Chuyển list OHLCVRecord → DataFrame đã chuẩn hoá."""
    records = []
    for c in candles:
        row = c.model_dump()
        if row.get("symbol") is None:
            row["symbol"] = symbol
        records.append(row)
    df = pd.DataFrame(records)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    return df.sort_values("open_time").reset_index(drop=True)


def _engineer_features_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nếu DataFrame thiếu technical indicators, tính thủ công (simple version).
    GBMForecaster.build_lag_features() sẽ xử lý lag features sau.
    """
    df = df.copy()

    # Chỉ tính nếu thiếu
    if "return_pct" not in df.columns or df["return_pct"].isna().all():
        df["return_pct"] = df["close"].pct_change() * 100

    if "log_return" not in df.columns or df["log_return"].isna().all():
        import numpy as np
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    if "ma_7" not in df.columns or df["ma_7"].isna().all():
        df["ma_7"] = df["close"].rolling(7).mean()

    if "ma_25" not in df.columns or df["ma_25"].isna().all():
        df["ma_25"] = df["close"].rolling(25).mean()

    if "ema_12" not in df.columns or df["ema_12"].isna().all():
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()

    if "ema_26" not in df.columns or df["ema_26"].isna().all():
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # Calendar cyclical features
    if "hour_sin" not in df.columns or df["hour_sin"].isna().all():
        import numpy as np
        dt = df["open_time"].dt
        df["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * dt.month / 12)
        df["doy_sin"] = np.sin(2 * np.pi * dt.dayofyear / 365)
        df["doy_cos"] = np.cos(2 * np.pi * dt.dayofyear / 365)

    # Volume ratio
    if "volume_ratio" not in df.columns or df["volume_ratio"].isna().all():
        vol_ma = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (vol_ma + 1e-8)

    # RSI (simple)
    if "rsi_14" not in df.columns or df["rsi_14"].isna().all():
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    if "macd_hist" not in df.columns or df["macd_hist"].isna().all():
        macd_line = df["close"].ewm(span=12, adjust=False).mean() - \
                    df["close"].ewm(span=26, adjust=False).mean()
        signal = macd_line.ewm(span=9, adjust=False).mean()
        df["macd"] = macd_line
        df["macd_signal"] = signal
        df["macd_hist"] = macd_line - signal

    # ATR
    if "atr_14" not in df.columns or df["atr_14"].isna().all():
        import numpy as np
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift(1)).abs()
        lc = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

    # Bollinger Bands
    if "bb_pct" not in df.columns or df["bb_pct"].isna().all():
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        df["bb_pct"] = (df["close"] - lower) / (upper - lower + 1e-8)
        df["bb_width"] = (upper - lower) / (sma + 1e-8)

    # Rolling vol & price zscore
    if "rolling_vol_30" not in df.columns or df["rolling_vol_30"].isna().all():
        df["rolling_vol_30"] = df["return_pct"].rolling(30).std()

    if "price_zscore" not in df.columns or df["price_zscore"].isna().all():
        sma30 = df["close"].rolling(30).mean()
        std30 = df["close"].rolling(30).std()
        df["price_zscore"] = (df["close"] - sma30) / (std30 + 1e-8)

    # Drawdown
    if "drawdown_pct" not in df.columns or df["drawdown_pct"].isna().all():
        rolling_max = df["close"].cummax()
        df["drawdown_pct"] = (df["close"] - rolling_max) / (rolling_max + 1e-8) * 100

    df = df.ffill().fillna(0)
    return df


# ─────────────────────────────────────────────────────────────
# GBM predict service
# ─────────────────────────────────────────────────────────────

def predict_gbm(request: PredictRequest) -> PredictResponse:
    """
    Thực hiện dự báo bằng GBM Stacking Ensemble.

    Raises
    ------
    RuntimeError: nếu model chưa load hoặc DB không khả dụng
    ValueError: nếu dữ liệu không đủ
    """
    from api.model_registry import ModelRegistry

    registry = ModelRegistry.get_instance()
    forecaster = registry.gbm_model

    if forecaster is None:
        raise RuntimeError(
            "GBM model chưa được load. "
            "Hãy train model và đặt file tại GBM_MODEL_PATH."
        )

    # ── 1. Lấy dữ liệu ────────────────────────────────────────────────────────
    if request.from_db:
        df = fetch_candles_from_db(request.symbol, request.timeframe, request.n_history)
    else:
        df = _candles_to_df(request.candles, request.symbol)

    if len(df) < 10:
        raise ValueError(f"Cần ít nhất 10 candles, chỉ có {len(df)}.")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    df = _engineer_features_if_needed(df)

    # ── 3. Chuẩn bị DataFrame theo format GBMForecaster ───────────────────────
    from models.gbm_model import GBMForecaster
    df_prep = GBMForecaster.prepare_dataframe(
        df,
        group_col=forecaster.cfg.group_col,
        time_col=forecaster.cfg.time_col,
        default_symbol=request.symbol,
    )

    # ── 4. Recursive predict ───────────────────────────────────────────────────
    result_df = forecaster.recursive_predict(df_prep, steps=request.steps)

    # ── 5. Format response ────────────────────────────────────────────────────
    forecast_steps = [
        ForecastStep(
            step=int(row["step"]),
            y_pred=float(row["y_pred"]),
            lower_80=float(row["lower_80"]),
            upper_80=float(row["upper_80"]),
            lower_90=float(row["lower_90"]),
            upper_90=float(row["upper_90"]),
        )
        for _, row in result_df.iterrows()
    ]

    return PredictResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        model="GBM",
        steps=request.steps,
        forecast=forecast_steps,
        n_history_rows=len(df),
        generated_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────
# TFT predict service (placeholder)
# ─────────────────────────────────────────────────────────────

def predict_tft(request: PredictRequest) -> PredictResponse:
    """
    Dự báo bằng TFT (Temporal Fusion Transformer).
    Hiện tại chỉ trả lỗi nếu model chưa load.
    TFT serialisation đầy đủ sẽ được bổ sung sau khi train trên Colab.
    """
    from api.model_registry import ModelRegistry

    registry = ModelRegistry.get_instance()
    if registry.tft_model is None:
        raise RuntimeError(
            "TFT model chưa được load. "
            "Hãy train model trên Colab và export về tft_output/."
        )

    raise NotImplementedError("TFT predict service đang được phát triển.")
