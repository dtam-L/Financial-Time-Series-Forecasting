"""
run_ingestion.py
================
Entry point for the Week 1 Data Ingestion pipeline.

Usage
-----
    # Full historical load (2022-01-01 → today):
    python -m data_collection_api.run_ingestion --mode initial

    # Incremental update (only new candles):
    python -m data_collection_api.run_ingestion --mode update

    # Continuous daily scheduler:
    python -m data_collection_api.run_ingestion --mode schedule --interval 24

    # Override symbols / timeframes from CLI:
    python -m data_collection_api.run_ingestion --mode initial --symbols BTC/USDT --timeframes 1d 4h
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# ------------------------------------------------------------------
# Configure loguru: console + rotating file
# ------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger.remove()                              # Remove default handler
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
)
logger.add(
    _LOG_DIR / "ingestion_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",        # New file every midnight
    retention="30 days",
    compression="zip",
    encoding="utf-8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m data_collection_api.run_ingestion",
        description="Binance → PostgreSQL ingestion pipeline (Week 1)",
    )
    parser.add_argument(
        "--mode",
        choices=["initial", "update", "schedule"],
        default="update",
        help=(
            "initial  — pull full history from START_DATE\n"
            "update   — pull only candles newer than latest DB timestamp\n"
            "schedule — run 'update' on a recurring interval"
        ),
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override SYMBOLS from .env (e.g. --symbols BTC/USDT ETH/USDT)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Override TIMEFRAMES from .env (e.g. --timeframes 1d 4h)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=24.0,
        help="Hours between scheduled runs (only for --mode schedule). Default: 24",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import so logger is set up before any module-level code runs
    from data_collection_api.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline(
        symbols=args.symbols,
        timeframes=args.timeframes,
    )

    if args.mode == "initial":
        pipeline.initial_load()

    elif args.mode == "update":
        pipeline.incremental_update()

    elif args.mode == "schedule":
        pipeline.run_scheduler(interval_hours=args.interval)

    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    pipeline.db.close()
    logger.success("Done.")


if __name__ == "__main__":
    main()
