
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

from loguru import logger

# ── Logging setup ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan> — <level>{message}</level>"
    ),
    colorize=True,
)
logger.add(
    _LOG_DIR / "eda_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="00:00",
    retention="30 days",
    encoding="utf-8",
)

sys.path.insert(0, str(_PROJECT_ROOT))


# ── EDA configuration ─────────────────────────────────────────────────────────
_SYMBOLS    = ["BTC/USDT", "ETH/USDT"]
_TIMEFRAMES = ["1d", "1h"]
_LOOKBACK   = {
    "1d": 730,   # 2 years of daily candles
    "1h": 180,   # 7.5 months of hourly candles (180 days)
}
_OUTPUT_ROOT = _PROJECT_ROOT / "reports" / "figures"
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Core EDA runner
# ──────────────────────────────────────────────────────────────────────────────

def run_eda() -> None:
    """
    Load data from PostgreSQL, compute features, and generate all charts
    for every (symbol, timeframe) combination. Also runs diagnostics
    and baseline model comparison.
    """
    from eda.data        import OHLCVLoader
    from eda.visualized  import EDAVisualizer
    from eda.diagnostics import StatisticalDiagnostics
    from eda.modelling   import BaselineModels
    from eda.feature_eda import FeatureEDA

    start_ts = datetime.utcnow()
    logger.info("=" * 64)
    logger.info("🔍 EDA PIPELINE STARTED")
    logger.info(f"   Symbols    : {_SYMBOLS}")
    logger.info(f"   Timeframes : {_TIMEFRAMES}")
    logger.info(f"   Output dir : {_OUTPUT_ROOT}")
    logger.info("=" * 64)

    loader = OHLCVLoader()
    viz    = EDAVisualizer(output_root=_OUTPUT_ROOT)
    diag   = StatisticalDiagnostics()
    bm     = BaselineModels()
    feda   = FeatureEDA(output_root=_OUTPUT_ROOT)

    # Cache loaded DataFrames for cross-symbol correlation chart
    loaded: dict[tuple, object] = {}

    try:
        for symbol in _SYMBOLS:
            for tf in _TIMEFRAMES:
                logger.info(f"\n{'─'*56}")
                logger.info(f"▶ Processing: {symbol} {tf}")
                logger.info(f"{'─'*56}")

                out_dir = _OUTPUT_ROOT / f"{symbol.replace('/', '_')}_{tf}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # ── Load & engineer features ──────────────────────────────────
                df = loader.load_with_features(
                    symbol, tf, lookback_days=_LOOKBACK.get(tf, 365)
                )
                if df.empty:
                    logger.warning(f"No data — skipping {symbol} {tf}")
                    continue

                loaded[(symbol, tf)] = df

                # ── Visualizations ────────────────────────────────────────────
                viz.run_all(df, symbol, tf)

                # ── Statistical diagnostics ───────────────────────────────────
                results = diag.run(df, symbol, tf)
                diag.print_report(results)
                diag.save_report(results, out_dir)

                # ── Baseline models ───────────────────────────────────────────
                bm.run(df, symbol, tf, out_dir)

                # ── Post-FE EDA (Chart 10-15 + text report) ───────────────────
                feda.run_all(df, symbol, tf)

        # ── Cross-symbol correlation (BTC vs ETH) ─────────────────────────────
        for tf in _TIMEFRAMES:
            key_a = ("BTC/USDT", tf)
            key_b = ("ETH/USDT", tf)
            if key_a in loaded and key_b in loaded:
                logger.info(f"\n▶ Cross-symbol correlation: BTC vs ETH [{tf}]")
                viz._plot_correlation(
                    loaded[key_a], "BTC/USDT",
                    loaded[key_b], "ETH/USDT",
                    tf,
                )

    finally:
        loader.close()

    elapsed = (datetime.utcnow() - start_ts).total_seconds()
    logger.success(
        f"\n✅ EDA Pipeline complete in {elapsed:.1f}s\n"
        f"   Charts saved to: {_OUTPUT_ROOT}"
    )
    _print_output_summary()


def _print_output_summary() -> None:
    """List all generated PNG/TXT files."""
    files = sorted(_OUTPUT_ROOT.rglob("*.png")) + sorted(_OUTPUT_ROOT.rglob("*.txt"))
    if not files:
        return
    logger.info(f"\n📁 Output files ({len(files)} total):")
    for f in files:
        rel = f.relative_to(_OUTPUT_ROOT)
        size_kb = f.stat().st_size / 1024
        logger.info(f"   {rel}  ({size_kb:.0f} KB)")


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler (every 48 hours)
# ──────────────────────────────────────────────────────────────────────────────

def run_scheduler() -> None:
    """Start APScheduler: run EDA immediately then repeat every 48 hours."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval  import IntervalTrigger

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_eda,
        trigger=IntervalTrigger(hours=48),
        id="eda_periodic",
        name="EDA Report — every 48h",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
        next_run_time=datetime.utcnow(),   # run immediately on start
    )

    logger.info("=" * 64)
    logger.info("⏰ EDA Scheduler active — runs every 48 hours.")
    logger.info("   First run starting NOW. Press Ctrl+C to stop.")
    logger.info("=" * 64)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eda.action",
        description="Real-Time EDA Pipeline for Financial Time Series",
    )
    parser.add_argument(
        "--mode",
        choices=["run", "schedule"],
        default="run",
        help=(
            "run      → Generate all charts once and exit.\n"
            "schedule → Run now, then repeat automatically every 48 hours."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "run":
        run_eda()
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
