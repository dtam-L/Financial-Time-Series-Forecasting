"""
models/data_loader.py
=====================
Data loading từ PostgreSQL + export JSON cho Colab.

Workflow
--------
  [LOCAL — có DB]                      [COLAB — không có DB]
  ─────────────────                    ─────────────────────
  OHLCVDBLoader.load()                 TFTForecaster / GBMForecaster
       ↓                                     ↓
  engineer_features()              load_and_split("train.json", "test.json")
       ↓
  split_and_export()
   → train.json
   → test.json
   → metadata.json
       ↓
  Upload lên Google Drive / Colab

Usage — Export JSON (chạy LOCAL)
---------------------------------
>>> from models.data_loader import OHLCVDBLoader
>>> loader = OHLCVDBLoader()
>>> loader.export_for_colab(
...     symbol="BTC/USDT",
...     timeframe="1d",
...     lookback_days=730,
...     output_dir="colab_data",
...     test_ratio=0.15,
... )
>>> loader.close()

Usage — Dùng trực tiếp (không export)
---------------------------------------
>>> df_train, df_test = loader.load_split(
...     symbol="BTC/USDT",
...     timeframe="1d",
...     lookback_days=730,
... )
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

# Đảm bảo project root trong sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class OHLCVDBLoader:
    """
    Load OHLCV + features từ PostgreSQL và export JSON cho Colab.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy URL. Nếu None, đọc từ .env (DB_HOST, DB_PORT, DB_NAME, ...).

    Methods
    -------
    load(symbol, timeframe, lookback_days)
        Load raw OHLCV + feature engineering từ DB.

    load_split(symbol, timeframe, lookback_days, test_ratio)
        Load + split train/test.

    export_for_colab(symbol, timeframe, lookback_days, output_dir, test_ratio)
        Load → split → save train.json + test.json + metadata.json.

    close()
        Đóng DB connection pool.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        try:
            from eda.data import OHLCVLoader
            self._loader = OHLCVLoader()
            self._db_url = db_url
            logger.info("OHLCVDBLoader: kết nối PostgreSQL thành công.")
        except Exception as exc:
            logger.error(
                f"Không thể kết nối DB: {exc}\n"
                "Đảm bảo PostgreSQL đang chạy và .env đã cấu hình đúng."
            )
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────────────────

    def load(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        lookback_days: int = 730,
    ) -> pd.DataFrame:
        """
        Load OHLCV + engineer_features từ PostgreSQL.

        Parameters
        ----------
        symbol        : e.g. "BTC/USDT", "ETH/USDT"
        timeframe     : e.g. "1d", "4h", "1h"
        lookback_days : số ngày lookback từ hôm nay

        Returns
        -------
        pd.DataFrame  indexed by open_time (UTC), đã có tất cả technical indicators
        """
        logger.info(f"Loading {symbol} {timeframe} | lookback={lookback_days}d ...")
        df = self._loader.load_with_features(symbol, timeframe, lookback_days)

        if df.empty:
            raise ValueError(
                f"Không có data cho {symbol} {timeframe} trong {lookback_days} ngày. "
                "Kiểm tra lại database."
            )

        logger.info(
            f"Loaded: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()} "
            f"| {df.shape[1]} columns"
        )
        return df

    def load_multiple(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        lookback_days: int = 730,
    ) -> pd.DataFrame:
        """
        Load nhiều symbols, gộp thành một DataFrame duy nhất.

        Thêm cột 'symbol' để phân biệt.

        Returns
        -------
        pd.DataFrame với DatetimeIndex và cột 'symbol'
        """
        dfs = []
        for sym in symbols:
            try:
                df = self.load(sym, timeframe, lookback_days)
                df["symbol"] = sym
                dfs.append(df)
            except Exception as exc:
                logger.warning(f"Bỏ qua {sym}: {exc}")

        if not dfs:
            raise ValueError("Không load được data cho bất kỳ symbol nào.")

        combined = pd.concat(dfs).sort_index()
        logger.info(
            f"Combined: {len(combined)} rows | {combined['symbol'].nunique()} symbols"
        )
        return combined

    # ──────────────────────────────────────────────────────────────────────────
    # Split
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame theo thứ tự thời gian (không random).

        Parameters
        ----------
        df         : DataFrame với DatetimeIndex
        test_ratio : tỷ lệ test (phần cuối)

        Returns
        -------
        (df_train, df_test)
        """
        cutoff = int(len(df) * (1 - test_ratio))
        df_train = df.iloc[:cutoff].copy()
        df_test  = df.iloc[cutoff:].copy()

        logger.info(
            f"Split | train={len(df_train)} ({df_train.index[0].date()}→{df_train.index[-1].date()}) "
            f"| test={len(df_test)} ({df_test.index[0].date()}→{df_test.index[-1].date()})"
        )
        return df_train, df_test

    def load_split(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        lookback_days: int = 730,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load từ DB + split train/test.

        Returns
        -------
        (df_train, df_test) — DataFrame đã có features, chưa có lag features
        """
        df = self.load(symbol, timeframe, lookback_days)
        df["symbol"] = symbol
        return self.chronological_split(df, test_ratio)

    # ──────────────────────────────────────────────────────────────────────────
    # Export cho Colab
    # ──────────────────────────────────────────────────────────────────────────

    def export_for_colab(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        lookback_days: int = 730,
        output_dir: str = "colab_data",
        test_ratio: float = 0.15,
        symbols: Optional[list[str]] = None,
    ) -> dict:
        """
        Load từ DB → split → lưu train.json + test.json + metadata.json.

        Dùng trên máy LOCAL có DB, sau đó upload JSON lên Colab.

        Parameters
        ----------
        symbol        : symbol chính (nếu chỉ 1 symbol)
        timeframe     : khung thời gian
        lookback_days : số ngày lookback
        output_dir    : thư mục lưu JSON
        test_ratio    : tỷ lệ test
        symbols       : list symbols nếu muốn load nhiều (override symbol)

        Returns
        -------
        dict : paths và thông tin về files đã tạo
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Load data
        if symbols and len(symbols) > 1:
            df = self.load_multiple(symbols, timeframe, lookback_days)
        else:
            df = self.load(symbol, timeframe, lookback_days)
            df["symbol"] = symbol

        # Reset index để open_time thành column (dễ serialize)
        if df.index.name == "open_time":
            df = df.reset_index()
        df["open_time"] = df["open_time"].astype(str)

        # Split
        cutoff = int(len(df) * (1 - test_ratio))
        df_train = df.iloc[:cutoff]
        df_test  = df.iloc[cutoff:]

        # Lưu JSON
        train_path = out / "train.json"
        test_path  = out / "test.json"
        meta_path  = out / "metadata.json"

        train_records = df_train.to_dict(orient="records")
        test_records  = df_test.to_dict(orient="records")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_records, f, indent=None, default=str)

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_records, f, indent=None, default=str)

        # Metadata
        metadata = {
            "symbol": symbol if not symbols else symbols,
            "timeframe": timeframe,
            "lookback_days": lookback_days,
            "test_ratio": test_ratio,
            "total_rows": len(df),
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "columns": list(df.columns),
            "n_features": len(df.columns),
            "date_range": {
                "start": str(df["open_time"].iloc[0]),
                "end":   str(df["open_time"].iloc[-1]),
                "train_end": str(df_train["open_time"].iloc[-1]),
                "test_start": str(df_test["open_time"].iloc[0]),
            },
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("Export hoàn tất!")
        logger.info(f"  train.json    : {len(df_train):,} rows → {train_path}")
        logger.info(f"  test.json     : {len(df_test):,} rows → {test_path}")
        logger.info(f"  metadata.json : → {meta_path}")
        logger.info(f"  Features      : {len(df.columns)} columns")
        logger.info("=" * 60)
        logger.info("Upload 3 files trên lên Google Drive / Colab.")

        return {
            "train_path": str(train_path),
            "test_path":  str(test_path),
            "meta_path":  str(meta_path),
            "metadata":   metadata,
        }

    def close(self) -> None:
        """Đóng DB connection."""
        try:
            self._loader.close()
            logger.info("DB connection closed.")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Convenience function
# ══════════════════════════════════════════════════════════════════════════════

def export_for_colab(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    lookback_days: int = 730,
    output_dir: str = "colab_data",
    test_ratio: float = 0.15,
) -> dict:
    """
    One-shot function: load từ DB và export JSON cho Colab.

    Usage
    -----
    >>> from models.data_loader import export_for_colab
    >>> export_for_colab("BTC/USDT", "1d", lookback_days=730, output_dir="colab_data")
    """
    loader = OHLCVDBLoader()
    try:
        return loader.export_for_colab(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            output_dir=output_dir,
            test_ratio=test_ratio,
        )
    finally:
        loader.close()
