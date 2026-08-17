from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

_REQUIRED_COLS = ["open_time", "open", "high", "low", "close", "volume"]
_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class CleaningReport:
    """Summary of rows removed or changed during cleaning."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_nulls: int = 0
    dropped_duplicates: int = 0
    dropped_invalid_ohlc: int = 0
    dropped_non_positive: int = 0
    dropped_inf: int = 0
    gap_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rows_removed(self) -> int:
        return self.rows_in - self.rows_out

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_removed": self.rows_removed,
            "dropped_nulls": self.dropped_nulls,
            "dropped_duplicates": self.dropped_duplicates,
            "dropped_invalid_ohlc": self.dropped_invalid_ohlc,
            "dropped_non_positive": self.dropped_non_positive,
            "dropped_inf": self.dropped_inf,
            "gap_count": self.gap_count,
            "notes": self.notes,
        }


class OHLCVCleaner:
    _TF_MS: dict[str, int] = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
        "3d": 259_200_000,
        "1w": 604_800_000,
        "1M": 2_592_000_000,
    }

    def __init__(
        self,
        drop_invalid: bool = True,
        dedupe_keep: str = "last",
        detect_gaps: bool = True,
    ) -> None:
        self.drop_invalid = drop_invalid
        self.dedupe_keep = dedupe_keep
        self.detect_gaps = detect_gaps

    def clean(
        self,
        df: pd.DataFrame,
        timeframe: Optional[str] = None,
    ) -> tuple[pd.DataFrame, CleaningReport]:
        """
        Return a cleaned copy of `df` plus a cleaning report.

        The output always contains columns:
        open_time, open, high, low, close, volume
        """
        report = CleaningReport(rows_in=len(df))

        if df.empty:
            report.rows_out = 0
            return _empty_ohlcv(), report

        out = _normalize_columns(df)
        out = _ensure_open_time_utc(out)
        out = _coerce_numeric(out, _OHLCV_COLS)

        before = len(out)
        out = out.dropna(subset=_REQUIRED_COLS)
        report.dropped_nulls = before - len(out)

        before = len(out)
        out = out.drop_duplicates(subset=["open_time"], keep=self.dedupe_keep)
        report.dropped_duplicates = before - len(out)

        if self.drop_invalid:
            before = len(out)
            out = out.replace([np.inf, -np.inf], np.nan)
            out = out.dropna(subset=_OHLCV_COLS)
            report.dropped_inf = before - len(out)

            mask_positive = (
                (out["open"] > 0)
                & (out["high"] > 0)
                & (out["low"] > 0)
                & (out["close"] > 0)
                & (out["volume"] >= 0)
            )
            before = len(out)
            out = out[mask_positive]
            report.dropped_non_positive = before - len(out)

            mask_valid = _valid_ohlc_mask(out)
            before = len(out)
            out = out[mask_valid]
            report.dropped_invalid_ohlc = before - len(out)

        out = out.sort_values("open_time").reset_index(drop=True)

        if self.detect_gaps and timeframe and len(out) > 1:
            report.gap_count = _count_gaps(out, timeframe, self._TF_MS)
            if report.gap_count:
                report.notes.append(
                    f"Detected {report.gap_count} missing candle interval(s)"
                )

        report.rows_out = len(out)
        return out[_REQUIRED_COLS], report


def clean_ohlcv(
    df: pd.DataFrame,
    timeframe: Optional[str] = None,
    *,
    log: bool = True,
) :

    cleaner = OHLCVCleaner()
    cleaned, report = cleaner.clean(df, timeframe=timeframe)

    if log and report.rows_in:
        logger.info(
            f"OHLCV cleaned: {report.rows_in} → {report.rows_out} rows "
            f"({report.rows_removed} removed)"
        )
        if report.rows_removed:
            logger.debug(f"Cleaning details: {report.as_dict()}")

    return cleaned, report


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=_REQUIRED_COLS)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in _REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    return out[_REQUIRED_COLS].copy()


def _ensure_open_time_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True, errors="coerce")
    return out


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _valid_ohlc_mask(df: pd.DataFrame) -> pd.Series:
    high_ok = (df["high"] >= df["open"]) & (df["high"] >= df["close"]) & (df["high"] >= df["low"])
    low_ok = (df["low"] <= df["open"]) & (df["low"] <= df["close"]) & (df["low"] <= df["high"])
    return high_ok & low_ok & (df["high"] >= df["low"])


def _count_gaps(df: pd.DataFrame, timeframe: str, tf_ms_map: dict[str, int]) -> int:
    step_ms = tf_ms_map.get(timeframe)
    if step_ms is None:
        return 0

    diffs = df["open_time"].diff().dropna()
    expected = pd.Timedelta(milliseconds=step_ms)
    return int((diffs > expected * 1.5).sum())
