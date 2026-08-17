

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
_CORE_INDICATORS = ["rsi_14", "macd", "bb_mid"]

_PERIODS_PER_YEAR: dict[str, int] = {
    "1m": 365 * 24 * 60,
    "3m": 365 * 24 * 20,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "2h": 365 * 12,
    "4h": 365 * 6,
    "6h": 365 * 4,
    "8h": 365 * 3,
    "12h": 365 * 2,
    "1d": 252,
    "3d": 84,
    "1w": 52,
    "1M": 12,
}


class FeatureEngineer:
    def __init__(self, drop_warmup: bool = True) -> None:
        self.drop_warmup = drop_warmup

    def transform(
        self,
        df: pd.DataFrame,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a copy of ``df`` enriched with all features."""
        return engineer_features(
            df,
            timeframe=timeframe,
            drop_warmup=self.drop_warmup,
            log=False,
        )

def engineer_features(
    df: pd.DataFrame,
    timeframe: Optional[str] = None,
    *,
    drop_warmup: bool = True,
    log: bool = True,
) :

    if df.empty:
        return df.copy()

    out = _prepare_ohlcv_index(df)
    rows_in = len(out)

    # ── Returns ──────────────────────────────────────────────────────────────
    out["return_pct"] = out["close"].pct_change() * 100
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))

    # ── Moving Averages ──────────────────────────────────────────────────────
    out["ma_7"] = out["close"].rolling(7).mean()
    out["ma_25"] = out["close"].rolling(25).mean()
    out["ma_99"] = out["close"].rolling(99).mean()
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()

    # ── MACD (12/26/9) ───────────────────────────────────────────────────────
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # ── RSI-14 (Wilder smoothing) ────────────────────────────────────────────
    delta = out["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands (20, 2σ) ───────────────────────────────────────────
    bb_std = out["close"].rolling(20).std()
    out["bb_mid"] = out["close"].rolling(20).mean()
    out["bb_upper"] = out["bb_mid"] + 2 * bb_std
    out["bb_lower"] = out["bb_mid"] - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"]
    out["bb_pct"] = (out["close"] - out["bb_lower"]) / (
        out["bb_upper"] - out["bb_lower"]
    ).replace(0, np.nan)

    # ── ATR-14 ───────────────────────────────────────────────────────────────
    hl = out["high"] - out["low"]
    hc = (out["high"] - out["close"].shift(1)).abs()
    lc = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(com=13, adjust=False).mean()

    # ── Volume ───────────────────────────────────────────────────────────────
    out["volume_ma_20"] = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / out["volume_ma_20"].replace(0, np.nan)

    # ── Risk metrics ─────────────────────────────────────────────────────────
    periods = _periods_per_year(timeframe)
    out["rolling_vol_30"] = (
        out["log_return"].rolling(30).std() * np.sqrt(periods)
    )
    out["price_zscore"] = (
        out["close"] - out["close"].rolling(60).mean()
    ) / out["close"].rolling(60).std().replace(0, np.nan)

    rolling_max = out["close"].cummax()
    out["drawdown_pct"] = (out["close"] - rolling_max) / rolling_max * 100

    out["regime"] = "Sideways"
    out.loc[
        (out["ma_7"] > out["ma_25"]) & (out["ma_25"] > out["ma_99"]),
        "regime",
    ] = "Bull"
    out.loc[
        (out["ma_7"] < out["ma_25"]) & (out["ma_25"] < out["ma_99"]),
        "regime",
    ] = "Bear"

    q33, q66 = out["atr_14"].quantile(0.33), out["atr_14"].quantile(0.66)
    out["vol_regime"] = "Mid"
    out.loc[out["atr_14"] <= q33, "vol_regime"] = "Low"
    out.loc[out["atr_14"] >= q66, "vol_regime"] = "High"
    out["day_of_week"] = out.index.dayofweek
    out["hour"] = out.index.hour
    out["month"] = out.index.month
    out["year"] = out.index.year
    out["day_of_month"] = out.index.day
    out["day_of_year"] = out.index.dayofyear

    _add_cyclical_calendar_features(out)

    if drop_warmup:
        out = out.dropna(subset=_CORE_INDICATORS)

    if log:
        logger.info(
            f"Features engineered: {rows_in} → {len(out)} rows, "
            f"{len(out.columns)} columns"
        )

    return out


def clean_and_engineer_features(
    df: pd.DataFrame,
    timeframe: Optional[str] = None,
    *,
    drop_warmup: bool = True,
) :
    from .clean import CleaningReport, clean_ohlcv

    cleaned, report = clean_ohlcv(df, timeframe=timeframe, log=False)
    if cleaned.empty:
        return cleaned, report

    featured = engineer_features(
        cleaned,
        timeframe=timeframe,
        drop_warmup=drop_warmup,
    )
    return featured, report


def encode_cyclical(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def _add_cyclical_calendar_features(df: pd.DataFrame) -> None:
    encodings = [
        ("hour", 24, "hour", 0),
        ("day_of_week", 7, "dow", 0),
        ("month", 12, "month", 1),          # 1–12 → 0–11
        ("day_of_month", 31, "dom", 1),     # 1–31 → 0–30
        ("day_of_year", 365.25, "doy", 1),  # 1–365 → 0–364
    ]
    for col, period, prefix, offset in encodings:
        sin_vals, cos_vals = encode_cyclical(df[col] - offset, period)
        df[f"{prefix}_sin"] = sin_vals
        df[f"{prefix}_cos"] = cos_vals


def _prepare_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize input to OHLCV columns indexed by UTC open_time."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    if "open_time" in out.columns:
        out["open_time"] = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
        out = out.dropna(subset=["open_time"])
        out = out.set_index("open_time")
    elif out.index.name != "open_time":
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out.index.name = "open_time"

    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    missing = [c for c in _OHLCV_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    return out[_OHLCV_COLS].sort_index()


def _periods_per_year(timeframe: Optional[str]) -> int:
    if timeframe and timeframe in _PERIODS_PER_YEAR:
        return _PERIODS_PER_YEAR[timeframe]
    return 252
