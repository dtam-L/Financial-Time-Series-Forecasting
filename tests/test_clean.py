"""
test_clean.py
=============
Unit tests for OHLCV cleaning before database ingestion.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from clean_feature_engineering_data.clean import OHLCVCleaner, clean_ohlcv


def _row(
    ts: datetime,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 105.0,
    volume: float = 1000.0,
) -> dict:
    return {
        "open_time": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class TestOHLCVCleaner:
    def test_keeps_valid_rows(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame([_row(ts)])

        cleaned, report = clean_ohlcv(df, timeframe="1d", log=False)

        assert len(cleaned) == 1
        assert report.rows_removed == 0
        assert cleaned.iloc[0]["close"] == 105.0

    def test_drops_duplicate_timestamps_keep_last(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame(
            [
                _row(ts, close=100.0),
                _row(ts, high=210.0, close=200.0),
            ]
        )

        cleaned, report = clean_ohlcv(df, log=False)

        assert len(cleaned) == 1
        assert cleaned.iloc[0]["close"] == 200.0
        assert report.dropped_duplicates == 1

    def test_drops_invalid_ohlc(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame(
            [
                _row(ts),
                _row(ts.replace(day=2), high=80.0, low=90.0),  # high < low
            ]
        )

        cleaned, report = clean_ohlcv(df, log=False)

        assert len(cleaned) == 1
        assert report.dropped_invalid_ohlc == 1

    def test_drops_non_positive_prices(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame([_row(ts, close=0.0)])

        cleaned, report = clean_ohlcv(df, log=False)

        assert cleaned.empty
        assert report.dropped_non_positive == 1

    def test_coerces_string_numbers(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        row = _row(ts)
        row["close"] = "105.5"
        df = pd.DataFrame([row])

        cleaned, _ = clean_ohlcv(df, log=False)

        assert cleaned.iloc[0]["close"] == pytest.approx(105.5)

    def test_detects_gaps(self):
        df = pd.DataFrame(
            [
                _row(datetime(2024, 1, 1, tzinfo=timezone.utc)),
                _row(datetime(2024, 1, 3, tzinfo=timezone.utc)),  # skip 1 day
            ]
        )

        cleaner = OHLCVCleaner(detect_gaps=True)
        _, report = cleaner.clean(df, timeframe="1d")

        assert report.gap_count == 1

    def test_empty_input(self):
        cleaned, report = clean_ohlcv(pd.DataFrame(), log=False)

        assert cleaned.empty
        assert report.rows_in == 0
        assert report.rows_out == 0

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)]})

        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            clean_ohlcv(df, log=False)

    def test_drops_inf_values(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame([_row(ts, close=np.inf)])

        cleaned, report = clean_ohlcv(df, log=False)

        assert cleaned.empty
        assert report.dropped_inf == 1
