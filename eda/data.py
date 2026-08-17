

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from clean_feature_engineering_data.features import engineer_features
from data_collection_api.db_manager import DatabaseManager


class OHLCVLoader:
    """
    Loads OHLCV data from PostgreSQL and enriches with technical indicators.

    Usage
    -----
    >>> loader = OHLCVLoader()
    >>> df = loader.load_with_features("BTC/USDT", "1d", lookback_days=730)
    >>> loader.close()
    """

    def __init__(self) -> None:
        self._db = DatabaseManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int = 730,
    ) -> pd.DataFrame:
        """
        Load raw OHLCV from PostgreSQL.

        Parameters
        ----------
        symbol : str        e.g. 'BTC/USDT'
        timeframe : str     e.g. '1d', '1h'
        lookback_days : int How many calendar days back from today.

        Returns
        -------
        pd.DataFrame  indexed by open_time (UTC DatetimeIndex)
        Columns: open, high, low, close, volume
        """
        since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        df = self._db.read_ohlcv(symbol, timeframe, since=since)
        if df.empty:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return df
        logger.info(
            f"Loaded {len(df)} rows | {symbol} {timeframe} | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return df

    def compute_features(
        self,
        df: pd.DataFrame,
        timeframe: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add all technical indicators and derived features.

        Delegates to :func:`clean_feature_engineering_data.engineer_features`.
        """
        return engineer_features(df, timeframe=timeframe, log=False)

    def load_with_features(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int = 730,
    ) -> pd.DataFrame:
        """Convenience: load() + compute_features() in one call."""
        raw = self.load(symbol, timeframe, lookback_days)
        if raw.empty:
            return raw
        return self.compute_features(raw, timeframe=timeframe)

    def close(self) -> None:
        """Release database connection pool."""
        self._db.close()
