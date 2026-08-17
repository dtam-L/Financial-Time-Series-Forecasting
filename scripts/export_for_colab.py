"""
scripts/export_for_colab.py
============================
Script chạy LOCAL để load dữ liệu từ PostgreSQL và export JSON cho Colab.

Chạy từ project root
---------------------
    python scripts/export_for_colab.py

Hoặc tuỳ chỉnh tham số
-----------------------
    python scripts/export_for_colab.py --symbol ETH/USDT --timeframe 4h --days 365

Output
------
    colab_data/
    ├── train.json      ← upload lên Colab
    ├── test.json       ← upload lên Colab
    └── metadata.json   ← thông tin về data
"""

import argparse
import os
import sys
from pathlib import Path

# Fix Unicode output trên Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Project root tren sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.data_loader import OHLCVDBLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export OHLCV + features từ PostgreSQL → JSON cho Colab"
    )
    parser.add_argument(
        "--symbol", "-s",
        default="BTC/USDT",
        help="Symbol cần export (default: BTC/USDT)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Nhiều symbols: --symbols BTC/USDT ETH/USDT BNB/USDT",
    )
    parser.add_argument(
        "--timeframe", "-tf",
        default="1d",
        help="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d, ... (default: 1d)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=730,
        help="Lookback days (default: 730 = 2 năm)",
    )
    parser.add_argument(
        "--test-ratio", "-t",
        type=float,
        default=0.15,
        help="Tỷ lệ test split (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="colab_data",
        help="Thư mục lưu JSON (default: colab_data/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  OHLCV DB -> JSON Export for Colab")
    print("=" * 60)
    print(f"  Symbol     : {args.symbols or args.symbol}")
    print(f"  Timeframe  : {args.timeframe}")
    print(f"  Lookback   : {args.days} days")
    print(f"  Test ratio : {args.test_ratio:.0%}")
    print(f"  Output     : {args.output_dir}/")
    print("=" * 60)

    loader = OHLCVDBLoader()
    try:
        result = loader.export_for_colab(
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.days,
            output_dir=args.output_dir,
            test_ratio=args.test_ratio,
            symbols=args.symbols,
        )

        print("\nExport thanh cong!")
        print("\nFiles da tao:")
        print(f"   {result['train_path']}")
        print(f"   {result['test_path']}")
        print(f"   {result['meta_path']}")

        meta = result["metadata"]
        print("\nThong tin:")
        print(f"   Tong rows  : {meta['total_rows']:,}")
        print(f"   Train rows : {meta['train_rows']:,}")
        print(f"   Test rows  : {meta['test_rows']:,}")
        print(f"   Features   : {meta['n_features']}")
        print(f"   Period     : {meta['date_range']['start'][:10]} -> {meta['date_range']['end'][:10]}")

        print("\nBuoc tiep theo:")
        print(f"   1. Upload '{args.output_dir}/train.json' len Google Drive")
        print(f"   2. Upload '{args.output_dir}/test.json' len Google Drive")
        print(f"   3. Mo tft_colab.ipynb hoac gbm_ensemble_colab.ipynb tren Colab")
        print(f"   4. Mount Drive va chinh TRAIN_JSON + TEST_JSON")

    except Exception as e:
        print(f"\nLoi: {e}")
        sys.exit(1)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
