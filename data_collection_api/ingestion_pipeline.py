
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger

from clean_feature_engineering_data.clean import clean_ohlcv

from . import config
from .binance_client import BinanceClient
from .db_manager import DatabaseManager


class IngestionPipeline:

    def __init__(
        self,
        client: Optional[BinanceClient] = None,
        db: Optional[DatabaseManager] = None,
        symbols: Optional[list[str]] = None,
        timeframes: Optional[list[str]] = None,
    ) -> None:
        self.client = client or BinanceClient()
        self.db = db or DatabaseManager()
        self.symbols = symbols or config.SYMBOLS
        self.timeframes = timeframes or config.TIMEFRAMES

        logger.info(
            f"IngestionPipeline ready | "
            f"symbols={self.symbols} | timeframes={self.timeframes}"
        )

    # ------------------------------------------------------------------
    # Core: single (symbol, timeframe) ingestion
    # ------------------------------------------------------------------

    def _ingest_one(
        self,
        symbol: str,
        timeframe: str,
        mode: str,
        since: Optional[datetime] = None,
    ) -> tuple[int, int]:
        run_id = self.db.log_run_start(symbol, timeframe, mode)
        total_inserted = total_updated = 0

        try:
            raw_df = self.client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
            )

            df, _ = clean_ohlcv(raw_df, timeframe=timeframe)

            if not df.empty:
                total_inserted, total_updated = self.db.insert_ohlcv(
                    df, symbol, timeframe
                )

            self.db.log_run_finish(
                run_id=run_id,
                rows_inserted=total_inserted,
                rows_updated=total_updated,
                status="success",
            )
        except Exception as exc:
            logger.error(f"Failed [{symbol} {timeframe}]: {exc}")
            self.db.log_run_finish(
                run_id=run_id,
                rows_inserted=0,
                rows_updated=0,
                status="error",
                error_message=str(exc),
            )
            raise

        return total_inserted, total_updated

    # ------------------------------------------------------------------
    # Public modes
    # ------------------------------------------------------------------

    def initial_load(self) -> None:
        """
        Pull full history (config.START_DATE → now) for all symbols/timeframes.
        Safe to re-run: upsert prevents duplicates.
        """
        logger.info("=" * 60)
        logger.info("MODE: initial_load")
        logger.info(f"  Start date : {config.START_DATE.date()}")
        logger.info(f"  Symbols    : {self.symbols}")
        logger.info(f"  Timeframes : {self.timeframes}")
        logger.info("=" * 60)

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                logger.info(f"▶ [{symbol}] [{timeframe}] initial load …")
                self._ingest_one(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode="initial",
                    since=config.START_DATE,
                )

        logger.success("initial_load complete.")
        self._print_summary()

    def incremental_update(self) -> None:
        """
        Fetch only candles newer than the latest timestamp in the DB.
        Ideal for scheduled runs (e.g., daily).
        """
        logger.info("=" * 60)
        logger.info("MODE: incremental_update")
        logger.info("=" * 60)

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                latest_ts = self.db.get_latest_timestamp(symbol, timeframe)

                if latest_ts is None:
                    logger.warning(
                        f"[{symbol} {timeframe}] No existing data found. "
                        f"Running initial load instead…"
                    )
                    since = config.START_DATE
                    mode = "initial"
                else:
                    # Start from the next candle after the last stored one
                    since = latest_ts + timedelta(seconds=1)
                    mode = "incremental"
                    logger.info(
                        f"▶ [{symbol}] [{timeframe}] incremental from {since.date()} …"
                    )

                self._ingest_one(
                    symbol=symbol,
                    timeframe=timeframe,
                    mode=mode,
                    since=since,
                )

        logger.success("incremental_update complete.")
        self._print_summary()

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def run_scheduler(self, interval_hours: float = 24.0) -> None:
        """
        Simple blocking scheduler: runs incremental_update in a loop.

        Parameters
        ----------
        interval_hours : float
            How often to poll Binance. Default 24h (daily candles).
        """
        logger.info(
            f"Scheduler started: incremental_update every "
            f"{interval_hours}h. Press Ctrl+C to stop."
        )
        try:
            while True:
                self.incremental_update()
                next_run = datetime.now(timezone.utc) + timedelta(hours=interval_hours)
                logger.info(f"Next run at {next_run.strftime('%Y-%m-%d %H:%M UTC')}")
                time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
        finally:
            self.db.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        """Log the v_data_summary view to console."""
        try:
            summary = self.db.get_data_summary()
            if not summary.empty:
                logger.info("\n" + summary.to_string(index=False))
        except Exception as exc:
            logger.warning(f"Could not fetch summary: {exc}")
