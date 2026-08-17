
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from . import config


class DatabaseManager:
    _SCHEMA_FILE = Path(__file__).parent / "schema.sql"

    def __init__(
        self,
        db_url: str = "",
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        self._db_url = db_url or config.DB_URL
        self._engine = self._create_engine(pool_size, max_overflow)
        self._ensure_database()
        self._apply_schema()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _create_engine(self, pool_size: int, max_overflow: int):
        try:
            engine = create_engine(
                self._db_url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,       # Verify connections before use
                echo=False,
            )
            # Quick connectivity test
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.success(f"Connected to PostgreSQL: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
            return engine
        except OperationalError:
            logger.warning(
                f"Database '{config.DB_NAME}' not found. Attempting to create it…"
            )
            self._create_database()
            engine = create_engine(
                self._db_url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,
                echo=False,
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.success(f"Database '{config.DB_NAME}' created and connected.")
            return engine

    def _create_database(self) -> None:
        admin_conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname="postgres",         
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
        admin_conn.autocommit = True
        with admin_conn.cursor() as cur:
                cur.execute(
                    f'CREATE DATABASE "{config.DB_NAME}" ENCODING \'UTF8\' TEMPLATE template0;'
                )
        admin_conn.close()
        logger.info(f"Database '{config.DB_NAME}' created successfully.")

    def _ensure_database(self) -> None:
        pass

    def _apply_schema(self) -> None:
        sql = self._SCHEMA_FILE.read_text(encoding="utf-8")
        raw_conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
        raw_conn.autocommit = True
        try:
            with raw_conn.cursor() as cur:
                cur.execute(sql)
            logger.info("Schema applied (tables, indexes & triggers ready).")
        except Exception as exc:
            logger.error(f"Schema application failed: {exc}")
            raise
        finally:
            raw_conn.close()

    # ------------------------------------------------------------------
    # Core OHLCV operations
    # ------------------------------------------------------------------

    def get_latest_timestamp(
        self, symbol: str, timeframe: str
    ) -> Optional[datetime]:
        sql = text(
            """
            SELECT MAX(open_time)
            FROM ohlcv_data
            WHERE symbol = :symbol AND timeframe = :timeframe
            """
        )
        with self._engine.connect() as conn:
            result = conn.execute(sql, {"symbol": symbol, "timeframe": timeframe})
            row = result.fetchone()
            return row[0] if row and row[0] else None

    def insert_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> tuple[int, int]:
        if df.empty:
            logger.warning(f"Empty DataFrame for {symbol} {timeframe}. Nothing to insert.")
            return 0, 0

        df = df.copy()
        df["symbol"] = symbol
        df["timeframe"] = timeframe

        # Ensure open_time is timezone-aware
        if df["open_time"].dt.tz is None:
            df["open_time"] = df["open_time"].dt.tz_localize("UTC")

        upsert_sql = text(
            """
            INSERT INTO ohlcv_data
                (symbol, timeframe, open_time, open, high, low, close, volume)
            VALUES
                (:symbol, :timeframe, :open_time, :open, :high, :low, :close, :volume)
            ON CONFLICT (symbol, timeframe, open_time)
            DO UPDATE SET
                open       = EXCLUDED.open,
                high       = EXCLUDED.high,
                low        = EXCLUDED.low,
                close      = EXCLUDED.close,
                volume     = EXCLUDED.volume,
                updated_at = NOW()
            RETURNING (xmax = 0) AS inserted
            """
        )

        inserted = updated = 0
        records = df[
            ["symbol", "timeframe", "open_time", "open", "high", "low", "close", "volume"]
        ].to_dict("records")

        with self._engine.begin() as conn:
            for record in records:
                result = conn.execute(upsert_sql, record)
                row = result.fetchone()
                if row and row[0]:
                    inserted += 1
                else:
                    updated += 1

        logger.info(
            f"[{symbol} {timeframe}] Upserted {len(records)} rows "
            f"→ {inserted} inserted, {updated} updated"
        )
        return inserted, updated

    def read_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> pd.DataFrame:
        conditions = ["symbol = :symbol", "timeframe = :timeframe"]
        params: dict = {"symbol": symbol, "timeframe": timeframe}
        if since:
            conditions.append("open_time >= :since")
            params["since"] = since
        if until:
            conditions.append("open_time <= :until")
            params["until"] = until

        where = " AND ".join(conditions)
        sql = text(
            f"SELECT open_time, open, high, low, close, volume "
            f"FROM ohlcv_data WHERE {where} ORDER BY open_time"
        )
        with self._engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params, parse_dates=["open_time"])

        df = df.set_index("open_time")
        df.index = df.index.tz_convert("UTC")
        return df

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def log_run_start(self, symbol: str, timeframe: str, mode: str) -> int:
        """Insert a new row in ingestion_log and return its id."""
        sql = text(
            """
            INSERT INTO ingestion_log (symbol, timeframe, mode)
            VALUES (:symbol, :timeframe, :mode)
            RETURNING id
            """
        )
        with self._engine.begin() as conn:
            result = conn.execute(sql, {"symbol": symbol, "timeframe": timeframe, "mode": mode})
            return result.fetchone()[0]

    def log_run_finish(
        self,
        run_id: int,
        rows_inserted: int,
        rows_updated: int,
        status: str = "success",
        error_message: str = "",
    ) -> None:
        """Update an existing ingestion_log row with the result."""
        sql = text(
            """
            UPDATE ingestion_log
            SET finished_at    = NOW(),
                rows_inserted  = :rows_inserted,
                rows_updated   = :rows_updated,
                status         = :status,
                error_message  = :error_message
            WHERE id = :run_id
            """
        )
        with self._engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "run_id": run_id,
                    "rows_inserted": rows_inserted,
                    "rows_updated": rows_updated,
                    "status": status,
                    "error_message": error_message,
                },
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_data_summary(self) -> pd.DataFrame:
        """Return the v_data_summary view as a DataFrame."""
        with self._engine.connect() as conn:
            return pd.read_sql(text("SELECT * FROM v_data_summary"), conn)

    def close(self) -> None:
        """Dispose the connection pool gracefully."""
        self._engine.dispose()
        logger.info("DatabaseManager connection pool closed.")
