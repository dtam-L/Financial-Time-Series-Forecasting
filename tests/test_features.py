"""
test_features.py
================
Unit tests for OHLCV feature engineering.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from clean_feature_engineering_data.features import (
    FeatureEngineer,
    clean_and_engineer_features,
    encode_cyclical,
    engineer_features,
)


def _make_ohlcv(n: int = 120, start: datetime | None = None) -> pd.DataFrame:
    """Synthetic daily OHLCV with a gentle upward trend."""
    start = start or datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for i in range(n):
        ts = start + timedelta(days=i)
        open_ = price
        close = price + np.sin(i / 5) * 2 + 0.5
        high = max(open_, close) + 1.5
        low = min(open_, close) - 1.5
        rows.append(
            {
                "open_time": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + i * 10,
            }
        )
        price = close
    return pd.DataFrame(rows)


class TestEngineerFeatures:
    def test_adds_expected_columns(self):
        df = _make_ohlcv()
        out = engineer_features(df, timeframe="1d", log=False)

        expected = {
            "return_pct",
            "log_return",
            "ma_7",
            "ma_25",
            "ma_99",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "macd_hist",
            "rsi_14",
            "bb_mid",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "bb_pct",
            "atr_14",
            "volume_ma_20",
            "volume_ratio",
            "rolling_vol_30",
            "price_zscore",
            "drawdown_pct",
            "regime",
            "vol_regime",
            "day_of_week",
            "hour",
            "month",
            "year",
            "day_of_month",
            "day_of_year",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "dom_sin",
            "dom_cos",
            "doy_sin",
            "doy_cos",
        }
        assert expected.issubset(set(out.columns))

    def test_drops_warmup_rows_by_default(self):
        df = _make_ohlcv(120)
        out = engineer_features(df, log=False)

        assert len(out) < len(df)
        assert out["rsi_14"].notna().all()
        assert out["macd"].notna().all()
        assert out["bb_mid"].notna().all()

    def test_keeps_warmup_when_requested(self):
        df = _make_ohlcv(120)
        out = engineer_features(df, drop_warmup=False, log=False)

        assert len(out) == len(df)

    def test_works_with_open_time_index(self):
        df = _make_ohlcv(120).set_index("open_time")
        out = engineer_features(df, timeframe="1d", log=False)

        assert out.index.name == "open_time"
        assert "rsi_14" in out.columns

    def test_rsi_in_valid_range(self):
        out = engineer_features(_make_ohlcv(), log=False)

        assert out["rsi_14"].between(0, 100).all()

    def test_regime_values(self):
        out = engineer_features(_make_ohlcv(), log=False)

        assert set(out["regime"].unique()).issubset({"Bull", "Bear", "Sideways"})

    def test_empty_input(self):
        out = engineer_features(pd.DataFrame(), log=False)
        assert out.empty

    def test_feature_engineer_class(self):
        engineer = FeatureEngineer()
        out = engineer.transform(_make_ohlcv(), timeframe="1d")

        assert "macd_hist" in out.columns


class TestCyclicalEncoding:
    def test_unit_circle_property(self):
        df = _make_ohlcv()
        out = engineer_features(df, log=False)

        for prefix in ("hour", "dow", "month", "dom", "doy"):
            radius = out[f"{prefix}_sin"] ** 2 + out[f"{prefix}_cos"] ** 2
            np.testing.assert_allclose(radius, 1.0, rtol=1e-5, atol=1e-5)

    def test_hour_wraps_correctly(self):
        """Hour 23 and hour 0 should be close in cyclical space."""
        sin_23, cos_23 = encode_cyclical(pd.Series([23]), 24)
        sin_0, cos_0 = encode_cyclical(pd.Series([0]), 24)

        dist = np.sqrt((sin_23.iloc[0] - sin_0.iloc[0]) ** 2 + (cos_23.iloc[0] - cos_0.iloc[0]) ** 2)
        assert dist < 0.3

    def test_month_december_january_close(self):
        # 0-indexed: Dec=11, Jan=0 — one step apart on the 12-month cycle
        sin_dec, cos_dec = encode_cyclical(pd.Series([11]), 12)
        sin_jan, cos_jan = encode_cyclical(pd.Series([0]), 12)

        dist = np.sqrt((sin_dec.iloc[0] - sin_jan.iloc[0]) ** 2 + (cos_dec.iloc[0] - cos_jan.iloc[0]) ** 2)
        assert dist < 0.55


class TestCleanAndEngineerFeatures:
    def test_pipeline_from_raw_data(self):
        df = _make_ohlcv()
        # Inject one bad row
        bad = df.iloc[0].copy()
        bad["high"] = 50.0
        df = pd.concat([df, pd.DataFrame([bad])], ignore_index=True)

        out, report = clean_and_engineer_features(df, timeframe="1d")

        assert report.dropped_invalid_ohlc >= 1
        assert "rsi_14" in out.columns
        assert len(out) > 0

    def test_empty_after_clean(self):
        out, report = clean_and_engineer_features(pd.DataFrame(), timeframe="1d")

        assert out.empty
        assert report.rows_in == 0
