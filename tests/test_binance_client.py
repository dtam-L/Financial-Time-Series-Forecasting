"""
test_binance_client.py
======================
Unit tests for BinanceClient.
fetch_ohlcv now uses direct requests calls (not ccxt), so we mock
_request_with_retry to return fake JSON responses.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: build a fake Binance klines row
# ---------------------------------------------------------------------------
def make_kline_row(ts_ms: int, price: float = 30000.0) -> list:
    """One row of /api/v3/klines response (12 fields)."""
    return [
        ts_ms,                       # open time
        str(price),                  # open
        str(price + 100),            # high
        str(price - 100),            # low
        str(price + 50),             # close
        "100.0",                     # volume
        ts_ms + 86_399_999,          # close time
        "3000000.0",                 # quote asset volume
        1000,                        # number of trades
        "50.0",                      # taker buy base volume
        "1500000.0",                 # taker buy quote volume
        "0",                         # ignore
    ]


def make_response(rows: list) -> MagicMock:
    """Build a mock requests.Response whose .json() returns `rows`."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = rows
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """BinanceClient with _request_with_retry patched to avoid real HTTP calls."""
    with patch("data_collection_api.binance_client.ccxt"):
        from data_collection_api.binance_client import BinanceClient
        c = BinanceClient(api_key="TEST_KEY", api_secret="TEST_SECRET")
        yield c


# ---------------------------------------------------------------------------
# OHLCV tests — mock _request_with_retry
# ---------------------------------------------------------------------------

class TestFetchOHLCV:

    def test_returns_dataframe_with_correct_columns(self, client):
        """Should return a DataFrame with the expected column names."""
        ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        client._request_with_retry = MagicMock(return_value=make_response([make_kline_row(ts)]))

        df = client.fetch_ohlcv(
            "BTC/USDT", "1d",
            since=datetime(2023, 1, 1, tzinfo=timezone.utc),
            until=datetime(2023, 1, 3, tzinfo=timezone.utc),
        )

        assert list(df.columns) == ["open_time", "open", "high", "low", "close", "volume"]
        assert len(df) == 1

    def test_open_time_is_utc_datetime(self, client):
        """open_time column should be tz-aware UTC datetimes."""
        ts = int(datetime(2023, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
        client._request_with_retry = MagicMock(return_value=make_response([make_kline_row(ts)]))

        df = client.fetch_ohlcv(
            "ETH/USDT", "1d",
            since=datetime(2023, 6, 1, tzinfo=timezone.utc),
            until=datetime(2023, 6, 3, tzinfo=timezone.utc),
        )

        assert df["open_time"].dt.tz is not None
        assert str(df["open_time"].dt.tz) == "UTC"

    def test_empty_response_returns_empty_dataframe(self, client):
        """Empty API response should return an empty DataFrame, not raise."""
        client._request_with_retry = MagicMock(return_value=make_response([]))

        df = client.fetch_ohlcv(
            "BTC/USDT", "1d",
            since=datetime(2023, 1, 1, tzinfo=timezone.utc),
            until=datetime(2023, 1, 2, tzinfo=timezone.utc),
        )

        assert df.empty

    def test_pagination_calls_api_multiple_times(self, client):
        """
        When a batch equals the limit, the client should make another request.
        """
        base_ts = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        day2_ts = base_ts + 86_400_000

        # First call: 1 row (== limit) → triggers second call
        # Second call: empty → stops
        client._request_with_retry = MagicMock(side_effect=[
            make_response([make_kline_row(base_ts)]),
            make_response([make_kline_row(day2_ts)]),
            make_response([]),
        ])

        client.fetch_ohlcv(
            "BTC/USDT", "1d",
            since=datetime(2023, 1, 1, tzinfo=timezone.utc),
            until=datetime(2023, 1, 10, tzinfo=timezone.utc),
            limit=1,
        )

        assert client._request_with_retry.call_count >= 2

    def test_invalid_timeframe_raises_value_error(self, client):
        """Unknown timeframe string should raise ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            client.fetch_ohlcv(
                "BTC/USDT", "99x",
                since=datetime(2023, 1, 1, tzinfo=timezone.utc),
            )

    def test_no_duplicates_in_result(self, client):
        """Duplicate timestamps from API should be deduplicated."""
        ts = int(datetime(2023, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
        duplicate = make_kline_row(ts)
        client._request_with_retry = MagicMock(return_value=make_response([duplicate, duplicate]))

        df = client.fetch_ohlcv(
            "BTC/USDT", "1d",
            since=datetime(2023, 3, 1, tzinfo=timezone.utc),
            until=datetime(2023, 3, 3, tzinfo=timezone.utc),
        )

        assert len(df) == 1

    def test_numeric_columns_are_float(self, client):
        """OHLCV columns should be float64, not strings."""
        ts = int(datetime(2023, 4, 1, tzinfo=timezone.utc).timestamp() * 1000)
        client._request_with_retry = MagicMock(return_value=make_response([make_kline_row(ts)]))

        df = client.fetch_ohlcv(
            "BTC/USDT", "1d",
            since=datetime(2023, 4, 1, tzinfo=timezone.utc),
            until=datetime(2023, 4, 3, tzinfo=timezone.utc),
        )

        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == float, f"{col} should be float"


# ---------------------------------------------------------------------------
# Ticker tests
# ---------------------------------------------------------------------------

class TestFetchTicker:

    def test_returns_dict(self, client):
        client._exchange.fetch_ticker.return_value = {"last": 30000.0, "symbol": "BTC/USDT"}
        ticker = client.fetch_ticker("BTC/USDT")
        assert isinstance(ticker, dict)

    def test_calls_exchange_with_correct_symbol(self, client):
        client._exchange.fetch_ticker.return_value = {"last": 2000.0}
        client.fetch_ticker("ETH/USDT")
        client._exchange.fetch_ticker.assert_called_once_with("ETH/USDT")
