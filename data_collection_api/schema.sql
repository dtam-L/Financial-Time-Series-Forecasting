-- ============================================================
-- schema.sql  —  Financial Time Series PostgreSQL schema
-- Week 1: Data Ingestion & Database
--
-- Run with:
--   psql -U postgres -d financial_ts -f schema.sql
-- ============================================================

-- Create the database (run as superuser if needed)
-- CREATE DATABASE financial_ts;

-- ============================================================
-- Table: ohlcv_data
-- Stores OHLCV candles for every (symbol, timeframe) pair.
-- ============================================================
CREATE TABLE IF NOT EXISTS ohlcv_data (
    id          BIGSERIAL,
    symbol      VARCHAR(20)        NOT NULL,   -- e.g. 'BTC/USDT'
    timeframe   VARCHAR(5)         NOT NULL,   -- e.g. '1d', '1h'
    open_time   TIMESTAMPTZ        NOT NULL,   -- candle open timestamp (UTC)
    open        NUMERIC(24, 8)     NOT NULL,
    high        NUMERIC(24, 8)     NOT NULL,
    low         NUMERIC(24, 8)     NOT NULL,
    close       NUMERIC(24, 8)     NOT NULL,
    volume      NUMERIC(32, 8)     NOT NULL,
    created_at  TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ        NOT NULL DEFAULT NOW(),

    -- Primary key: one row per (symbol, timeframe, open_time)
    CONSTRAINT ohlcv_data_pkey PRIMARY KEY (symbol, timeframe, open_time)
);

-- Partial index: latest candles retrieved fast
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_time
    ON ohlcv_data (symbol, timeframe, open_time DESC);

-- Index for time-range queries across all symbols
CREATE INDEX IF NOT EXISTS idx_ohlcv_open_time
    ON ohlcv_data (open_time DESC);

-- ============================================================
-- Table: ingestion_log
-- Tracks each pipeline run for audit / resume purposes.
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(20)  NOT NULL,
    timeframe       VARCHAR(5)   NOT NULL,
    mode            VARCHAR(20)  NOT NULL,   -- 'initial' | 'incremental'
    rows_inserted   INTEGER      NOT NULL DEFAULT 0,
    rows_updated    INTEGER      NOT NULL DEFAULT 0,
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    status          VARCHAR(20)  NOT NULL DEFAULT 'running',  -- 'running' | 'success' | 'error'
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_symbol_tf
    ON ingestion_log (symbol, timeframe, started_at DESC);

-- ============================================================
-- Trigger: auto-update updated_at on ohlcv_data
-- ============================================================
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_updated_at ON ohlcv_data;
CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON ohlcv_data
    FOR EACH ROW
    EXECUTE FUNCTION trigger_set_updated_at();

-- ============================================================
-- Convenience views
-- ============================================================

-- Latest close price per symbol/timeframe
CREATE OR REPLACE VIEW v_latest_price AS
SELECT DISTINCT ON (symbol, timeframe)
    symbol,
    timeframe,
    open_time,
    open,
    high,
    low,
    close,
    volume
FROM ohlcv_data
ORDER BY symbol, timeframe, open_time DESC;

-- Row counts per (symbol, timeframe)
CREATE OR REPLACE VIEW v_data_summary AS
SELECT
    symbol,
    timeframe,
    COUNT(*)                    AS total_rows,
    MIN(open_time)              AS earliest,
    MAX(open_time)              AS latest,
    MAX(updated_at)             AS last_updated
FROM ohlcv_data
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;
