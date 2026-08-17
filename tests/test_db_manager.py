"""
test_db_manager.py
==================
Unit tests for DatabaseManager using a mocked SQLAlchemy engine.
These tests do NOT require a live PostgreSQL instance.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv_df(n: int = 3) -> pd.DataFrame:
    """Create a small test OHLCV DataFrame with n rows."""
    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = pd.Timestamp(base_ts) + pd.Timedelta(days=i)
        rows.append({
            "open_time": ts,
            "open": 30000.0 + i,
            "high": 31000.0 + i,
            "low": 29000.0 + i,
            "close": 30500.0 + i,
            "volume": 100.0 + i,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_manager():
    """
    Return a DatabaseManager where all heavy init is bypassed via mocks.
    """
    with (
        patch("data_collection_api.db_manager.create_engine") as mock_create_engine,
        patch.object(
            __import__("data_collection_api.db_manager", fromlist=["DatabaseManager"]).DatabaseManager,
            "_ensure_database",
        ),
        patch.object(
            __import__("data_collection_api.db_manager", fromlist=["DatabaseManager"]).DatabaseManager,
            "_apply_schema",
        ),
    ):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Make connect() return a usable context manager
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

        from data_collection_api.db_manager import DatabaseManager
        db = DatabaseManager.__new__(DatabaseManager)
        db._engine = mock_engine
        db._db_url = "postgresql+psycopg2://test:test@localhost/test"
        yield db, mock_engine, mock_conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetLatestTimestamp:

    def test_returns_datetime_when_row_exists(self, mock_db_manager):
        db, engine, conn = mock_db_manager
        expected = datetime(2024, 6, 15, tzinfo=timezone.utc)
        conn.execute.return_value.fetchone.return_value = (expected,)

        result = db.get_latest_timestamp("BTC/USDT", "1d")

        assert result == expected

    def test_returns_none_when_no_data(self, mock_db_manager):
        db, engine, conn = mock_db_manager
        conn.execute.return_value.fetchone.return_value = (None,)

        result = db.get_latest_timestamp("ETH/USDT", "1d")

        assert result is None


class TestInsertOHLCV:

    def test_empty_dataframe_returns_zero_counts(self, mock_db_manager):
        db, engine, conn = mock_db_manager
        inserted, updated = db.insert_ohlcv(pd.DataFrame(), "BTC/USDT", "1d")
        assert inserted == 0
        assert updated == 0

    def test_calls_execute_for_each_row(self, mock_db_manager):
        db, engine, conn = mock_db_manager
        # Simulate all rows being new inserts (xmax == 0 → inserted=True)
        conn.execute.return_value.fetchone.return_value = (True,)

        df = make_ohlcv_df(n=5)
        inserted, updated = db.insert_ohlcv(df, "BTC/USDT", "1d")

        assert conn.execute.call_count == 5
        assert inserted == 5
        assert updated == 0

    def test_counts_updates_correctly(self, mock_db_manager):
        db, engine, conn = mock_db_manager
        # Simulate all rows being updates (xmax != 0 → inserted=False)
        conn.execute.return_value.fetchone.return_value = (False,)

        df = make_ohlcv_df(n=3)
        inserted, updated = db.insert_ohlcv(df, "ETH/USDT", "1d")

        assert inserted == 0
        assert updated == 3

    def test_adds_symbol_and_timeframe_columns(self, mock_db_manager):
        """The DF passed into execute should include symbol/timeframe."""
        db, engine, conn = mock_db_manager
        conn.execute.return_value.fetchone.return_value = (True,)

        df = make_ohlcv_df(n=1)
        db.insert_ohlcv(df, "BTC/USDT", "1d")

        # Inspect the params dict passed to conn.execute
        call_args = conn.execute.call_args_list[0]
        params = call_args[0][1]          # Second positional arg is the dict
        assert params["symbol"] == "BTC/USDT"
        assert params["timeframe"] == "1d"


class TestAuditLog:

    def test_log_run_start_returns_integer_id(self, mock_db_manager):
        db, engine, conn = mock_db_manager
        conn.execute.return_value.fetchone.return_value = (42,)

        run_id = db.log_run_start("BTC/USDT", "1d", "initial")

        assert run_id == 42

    def test_log_run_finish_calls_execute(self, mock_db_manager):
        db, engine, conn = mock_db_manager
        db.log_run_finish(
            run_id=1,
            rows_inserted=100,
            rows_updated=5,
            status="success",
        )
        conn.execute.assert_called_once()
