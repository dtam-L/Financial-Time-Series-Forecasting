from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> — <level>{message}</level>"
    ),
    colorize=True,
)
logger.add(
    _LOG_DIR / "scheduler_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
)


# ------------------------------------------------------------------
# Job functions (imported lazily to keep startup fast)
# ------------------------------------------------------------------

def _job_hourly() -> None:
    """Incremental update cho dữ liệu 1h — chạy mỗi giờ."""
    from data_collection_api.ingestion_pipeline import IngestionPipeline
    logger.info("⏰ [SCHEDULER] Hourly job triggered")
    pipeline = IngestionPipeline(symbols=None, timeframes=["1h"])
    try:
        pipeline.incremental_update()
    finally:
        pipeline.db.close()


def _job_daily() -> None:
    """Incremental update cho dữ liệu 1d — chạy lúc 00:05 UTC mỗi ngày."""
    from data_collection_api.ingestion_pipeline import IngestionPipeline
    logger.info("📅 [SCHEDULER] Daily job triggered")
    pipeline = IngestionPipeline(symbols=None, timeframes=["1d"])
    try:
        pipeline.incremental_update()
    finally:
        pipeline.db.close()


# ------------------------------------------------------------------
# Scheduler setup
# ------------------------------------------------------------------

def main() -> None:
    scheduler = BlockingScheduler(timezone="UTC")

    # 1h candles: update every 60 minutes (at minute 2 of each hour to ensure candle is closed)
    scheduler.add_job(
        _job_hourly,
        trigger=CronTrigger(minute=2),      # xx:02 every hour
        id="hourly_ohlcv",
        name="BTC/ETH 1h OHLCV update",
        max_instances=1,
        coalesce=True,                       # Skip missed runs (e.g. if PC was off)
        misfire_grace_time=300,              # 5-minute grace period
    )

    # 1d candles: update every day at 00:05 UTC (daily candle closes at 00:00 UTC)
    scheduler.add_job(
        _job_daily,
        trigger=CronTrigger(hour=0, minute=5),
        id="daily_ohlcv",
        name="BTC/ETH 1d OHLCV update",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    logger.info("=" * 60)
    logger.info("🚀 Scheduler started. Jobs:")
    for job in scheduler.get_jobs():
        next_run = getattr(job, "next_run_time", getattr(job, "next_fire_time", "unknown"))
        logger.info(f"   [{job.id}] {job.name} — next: {next_run}")
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 60)

    try:
        # Run hourly job immediately on startup so we have fresh data
        logger.info("▶ Running initial incremental update on startup…")
        from data_collection_api.ingestion_pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        pipeline.incremental_update()
        pipeline.db.close()
        logger.success("Startup update complete. Scheduler now active.")

        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
