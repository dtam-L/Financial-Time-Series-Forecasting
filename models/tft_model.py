"""
models/tft_model.py
===================
Temporal Fusion Transformer (TFT) cho Financial Time Series Forecasting.

Kỹ thuật tăng độ chính xác tích hợp
--------------------------------------
  1. Optuna HPO          — tự động tìm learning_rate, hidden_size, dropout, ...
  2. Walk-Forward CV     — cross-validation time-aware, không data leakage
  3. Quantile Forecasting — dự báo khoảng tin cậy q10 / q50 / q90
  4. Checkpoint Ensemble — trung bình hoá top-k checkpoint tốt nhất

Cải tiến v2 (fix thông số xấu)
--------------------------------
  - Target: dự báo `log_return` thay vì `close` tuyệt đối, sau đó tái tạo lại giá
  - Normalizer: EncoderNormalizer (per-sequence) thay GroupNormalizer(softplus)
    → tránh NaN/Infinity loss với BTC price ở mức 60k–90k
  - HPO: gradient_clip_val tunable, trial timeout, best_val_loss finite check
  - batch_size giảm xuống 32 để gradient ổn định hơn với dataset nhỏ (~600 rows)
  - LR scheduler: ReduceLROnPlateau patience tăng lên 5

Input data format (JSON)
------------------------
Train/test đã feature-engineered, lưu dưới dạng JSON records:
  [
    {"open_time": "2024-01-01T00:00:00+00:00", "symbol": "BTC/USDT",
     "close": 42000.0, "rsi_14": 58.3, "macd_hist": 123.4, ...},
    ...
  ]

Usage
-----
>>> from models.tft_model import TFTConfig, TFTForecaster
>>> cfg = TFTConfig()   # mặc định target="log_return", tái tạo close qua close_col
>>> f = TFTForecaster(cfg)
>>> df_train, df_val, df_test = f.load_and_split("train.json", "test.json")
>>> f.tune_hyperparameters(df_train, df_val)          # Optuna HPO (optional)
>>> f.walk_forward_cv(df_train)                        # Walk-Forward CV (optional)
>>> f.train(df_train, df_val)                          # Final training
>>> preds = f.ensemble_predict(df_test)                # Ensemble predict
>>> f.save_results("results/tft_results.json", y_true, preds)
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TFTForecaster")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TFTConfig:
    """
    Configuration cho TFTForecaster.

    Attributes
    ----------
    target                : Cột dự báo. Mặc định "log_return" để tránh
                            scale issue với giá tuyệt đối BTC (~65k USD).
                            Sau khi predict, tái tạo close qua close_col.
    close_col             : Cột giá close để tái tạo giá sau khi dự báo return.
                            Chỉ cần khi target != "close".
    group_col             : Cột nhận dạng chuỗi (symbol / ticker)
    time_col              : Cột thời gian (sẽ parse thành datetime)
    max_encoder_length    : Độ dài cửa sổ encoder (lookback)
    max_prediction_length : Horizon dự báo (số bước phía trước)
    known_real_features   : Features đã biết trước trong tương lai (calendar)
    unknown_real_features : Features chỉ biết trong quá khứ (indicators)
    quantiles             : Các mức quantile cần dự báo
    output_dir            : Thư mục lưu checkpoints + kết quả
    n_optuna_trials       : Số trials Optuna HPO
    n_cv_folds            : Số fold Walk-Forward CV
    max_epochs            : Số epoch tối đa mỗi lần train
    use_encoder_normalizer: True → dùng EncoderNormalizer (per-sequence),
                            False → GroupNormalizer. EncoderNormalizer ổn định
                            hơn với dữ liệu tài chính có drift lớn.
    """

    # ── Target & groups ──────────────────────────────────────────────────────
    # FIX: dự báo log_return thay vì close tuyệt đối
    # → tránh GroupNormalizer(softplus) overflow với BTC ~65k USD
    # → loss hội tụ ổn định, HPO không còn trả về Infinity
    target: str = "log_return"
    close_col: str = "close"          # dùng để reconstruct giá sau predict
    group_col: str = "symbol"
    time_col: str = "open_time"

    # ── Sequence lengths ─────────────────────────────────────────────────────
    max_encoder_length: int = 30      # giảm từ 60 → 30: dataset 600 rows daily
    max_prediction_length: int = 7

    # ── Features ─────────────────────────────────────────────────────────────
    known_real_features: List[str] = field(default_factory=lambda: [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "doy_sin", "doy_cos",
    ])
    # FIX: bỏ "log_return" khỏi unknown_real_features vì nó là target
    # và bỏ các MA/EMA có scale lớn (gây nhiễu khi normalize cùng return)
    unknown_real_features: List[str] = field(default_factory=lambda: [
        "return_pct",
        "rsi_14", "macd_hist", "macd", "macd_signal",
        "bb_pct", "bb_width",
        "atr_14", "volume_ratio",
        "rolling_vol_30", "price_zscore", "drawdown_pct",
    ])

    # ── Quantile output ──────────────────────────────────────────────────────
    quantiles: List[float] = field(default_factory=lambda: [
        0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98
    ])

    # ── Training hyperparameters (defaults — tunable via Optuna) ─────────────
    # FIX: batch_size 32 (dataset nhỏ ~600 rows, 64 quá lớn → ít gradient updates)
    batch_size: int = 32
    max_epochs: int = 80
    early_stopping_patience: int = 15
    gradient_clip_val: float = 0.1    # giữ thấp vì target là return (giá trị nhỏ)
    learning_rate: float = 1e-3       # an toàn hơn 0.03 cho log_return target
    hidden_size: int = 32             # nhỏ hơn: dataset 600 rows, tránh overfit
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 16

    # ── Normalizer choice ────────────────────────────────────────────────────
    # FIX: EncoderNormalizer chuẩn hóa per-sequence, không dùng softplus
    # → tránh hoàn toàn NaN loss với log_return (range ~ -0.1 to 0.1)
    use_encoder_normalizer: bool = True

    # ── Optuna ───────────────────────────────────────────────────────────────
    n_optuna_trials: int = 30
    optuna_timeout_sec: Optional[int] = 1800  # tối đa 30 phút

    # ── Walk-forward CV ──────────────────────────────────────────────────────
    n_cv_folds: int = 3               # giảm từ 5 → 3: dataset nhỏ

    # ── Paths ────────────────────────────────────────────────────────────────
    output_dir: str = "tft_output"

    # ── Val split ratio (khi split từ train) ─────────────────────────────────
    val_ratio: float = 0.15

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @property
    def checkpoint_dir(self) -> str:
        return str(Path(self.output_dir) / "checkpoints")

    @property
    def output_size(self) -> int:
        return len(self.quantiles)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination (R²).
    
    R² = 1 - (SS_res / SS_tot)
    where:
        SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
        SS_tot = Σ(y_true - y_mean)²  (total sum of squares)
    
    Returns
    -------
    float : R² score. Range: (-∞, 1]. Perfect fit = 1.0, baseline = 0.0
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:  # Prevent division by zero
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

def _winkler(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Winkler Score — điểm thấp hơn = dự báo khoảng tốt hơn."""
    width = upper - lower
    penalty = np.where(
        y_true < lower, 2 / alpha * (lower - y_true),
        np.where(y_true > upper, 2 / alpha * (y_true - upper), 0.0),
    )
    return float(np.mean(width + penalty))

def _coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class TFTForecaster:
    """
    Temporal Fusion Transformer Forecaster.

    Accuracy-boosting techniques
    ----------------------------
    1. Optuna HPO          → tune_hyperparameters()
    2. Walk-Forward CV     → walk_forward_cv()
    3. Quantile output     → predict() / ensemble_predict()
    4. Checkpoint Ensemble → ensemble_predict()

    Key fixes (v2)
    --------------
    - Dự báo log_return thay vì close tuyệt đối → loss hội tụ, HPO không còn Infinity
    - EncoderNormalizer thay GroupNormalizer(softplus) → không overflow với BTC prices
    - Sau predict: reconstruct giá close từ log_return predictions
    - max_encoder_length=30, batch_size=32 phù hợp dataset ~600 rows daily

    Quick start
    -----------
    >>> cfg = TFTConfig()  # target="log_return" mặc định
    >>> f = TFTForecaster(cfg)
    >>> df_train, df_val, df_test = f.load_and_split("train.json", "test.json")
    >>> f.tune_hyperparameters(df_train, df_val)
    >>> f.train(df_train, df_val)
    >>> preds = f.ensemble_predict(df_test)
    >>> # preds chứa cả log_return lẫn close_reconstructed
    >>> f.save_results("results.json", y_true=df_test[cfg.close_col], predictions=preds)
    """

    def __init__(self, config: TFTConfig) -> None:
        self.cfg = config
        self.best_hparams: Optional[dict] = None
        self.trainer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.training_dataset: Optional[Any] = None
        self.cv_results: List[dict] = []
        # Lưu giá close cuối cùng của train để reconstruct sau predict
        self._last_train_close: Optional[float] = None

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"TFTForecaster ready | target={config.target} | "
            f"encoder={config.max_encoder_length} | pred_len={config.max_prediction_length} | "
            f"quantiles={config.quantiles} | normalizer="
            f"{'EncoderNormalizer' if config.use_encoder_normalizer else 'GroupNormalizer'}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Data helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def prepare_dataframe(
        df: pd.DataFrame,
        group_col: str = "symbol",
        time_col: str = "open_time",
        default_symbol: str = "BTC/USDT",
    ) -> pd.DataFrame:
        """
        Chuẩn bị DataFrame cho pytorch_forecasting.

        - Thêm cột ``time_idx`` (int, tăng dần per group)
        - Parse cột thời gian thành datetime UTC
        - Forward-fill NaN
        - Đảm bảo cột group_col tồn tại

        Parameters
        ----------
        df             : DataFrame đã feature engineering
        group_col      : Tên cột symbol / group
        time_col       : Tên cột thời gian
        default_symbol : Symbol mặc định nếu cột group không có

        Returns
        -------
        pd.DataFrame sẵn sàng cho TimeSeriesDataSet
        """
        out = df.copy()

        # Đảm bảo cột group
        if group_col not in out.columns:
            out[group_col] = default_symbol
        out[group_col] = out[group_col].astype(str)

        # Parse thời gian
        if time_col in out.columns:
            out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
            out = out.dropna(subset=[time_col])
        else:
            # Index là DatetimeIndex
            out[time_col] = pd.to_datetime(out.index, utc=True, errors="coerce")

        out = out.sort_values([group_col, time_col]).reset_index(drop=True)

        # Tạo time_idx per group
        out["time_idx"] = out.groupby(group_col).cumcount()

        # Fill NaN
        out = out.ffill().fillna(0)

        return out

    @staticmethod
    def load_json(path: str) -> pd.DataFrame:
        """Load JSON file (orient=records) thành DataFrame."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        # Nếu là dict với key "data" hoặc "records"
        for key in ("data", "records", "train", "test"):
            if key in data:
                return pd.DataFrame(data[key])
        return pd.DataFrame(data)

    def load_and_split(
        self,
        train_path: str,
        test_path: str,
        val_ratio: Optional[float] = None,
        symbol: str = "BTC/USDT",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train/test JSON, chuẩn bị DataFrames.

        Tách phần cuối của train làm validation.
        Lưu giá close cuối train để reconstruct sau predict.

        Parameters
        ----------
        train_path : đường dẫn đến train.json
        test_path  : đường dẫn đến test.json
        val_ratio  : tỷ lệ lấy từ train làm val (mặc định cfg.val_ratio)
        symbol     : tên symbol nếu cột chưa có

        Returns
        -------
        (df_train, df_val, df_test) — đã prepare_dataframe
        """
        val_ratio = val_ratio or self.cfg.val_ratio

        logger.info(f"Loading train: {train_path}")
        df_train_raw = self.load_json(train_path)
        logger.info(f"Loading test:  {test_path}")
        df_test_raw = self.load_json(test_path)

        df_full = self.prepare_dataframe(df_train_raw, self.cfg.group_col, self.cfg.time_col, symbol)
        df_test = self.prepare_dataframe(df_test_raw, self.cfg.group_col, self.cfg.time_col, symbol)

        # Tính log_return nếu target là log_return nhưng cột chưa có
        for df_part in [df_full, df_test]:
            if "log_return" not in df_part.columns and self.cfg.close_col in df_part.columns:
                df_part["log_return"] = np.log(
                    df_part[self.cfg.close_col] / df_part[self.cfg.close_col].shift(1)
                ).fillna(0.0)
            if "return_pct" not in df_part.columns and self.cfg.close_col in df_part.columns:
                df_part["return_pct"] = df_part[self.cfg.close_col].pct_change().fillna(0.0)

        # Split train/val (chronological)
        cutoff = int(len(df_full) * (1 - val_ratio))
        df_train = df_full.iloc[:cutoff].copy()
        df_val = df_full.iloc[cutoff:].copy()

        # Lưu giá close cuối train để reconstruct giá từ log_return predictions
        if self.cfg.close_col in df_train.columns:
            self._last_train_close = float(df_train[self.cfg.close_col].iloc[-1])
            logger.info(f"Last train close (for reconstruction): {self._last_train_close:,.2f}")

        logger.info(
            f"Split complete | train={len(df_train)} | val={len(df_val)} | test={len(df_test)}"
        )
        return df_train, df_val, df_test

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: feature spec
    # ──────────────────────────────────────────────────────────────────────────

    def _available(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Lọc chỉ các features thực sự có trong df."""
        return [f for f in features if f in df.columns]

    def _feature_spec(self, df: pd.DataFrame) -> dict:
        """Trả về feature spec cho TimeSeriesDataSet."""
        known = ["time_idx"] + self._available(df, self.cfg.known_real_features)

        unknown = self._available(df, self.cfg.unknown_real_features)
        # Target phải là phần tử đầu tiên của unknown_reals
        if self.cfg.target in unknown:
            unknown = [self.cfg.target] + [f for f in unknown if f != self.cfg.target]
        else:
            unknown = [self.cfg.target] + unknown

        # Loại bỏ close_col khỏi unknown_reals nếu target là log_return
        # (close có scale ~65k, nhiễu khi cùng normalize với log_return ~0.01)
        if self.cfg.target != self.cfg.close_col and self.cfg.close_col in unknown:
            unknown = [f for f in unknown if f != self.cfg.close_col]

        static_cats = [self.cfg.group_col]

        logger.info(
            f"Features — known_reals: {len(known)}, "
            f"unknown_reals: {len(unknown)}, target: {self.cfg.target}"
        )
        return {"known_reals": known, "unknown_reals": unknown, "static_cats": static_cats}

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset & DataLoader
    # ──────────────────────────────────────────────────────────────────────────

    def build_datasets(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> Tuple[Any, Any]:
        """
        Tạo TimeSeriesDataSet cho train và validation.

        FIX v2: dùng EncoderNormalizer (per-sequence standardization) thay
        GroupNormalizer(softplus) để tránh NaN loss với log_return target.

        Returns
        -------
        (training_dataset, validation_dataset)
        """
        from pytorch_forecasting import TimeSeriesDataSet

        spec = self._feature_spec(df_train)

        # ── Chọn normalizer ────────────────────────────────────────────────
        if self.cfg.use_encoder_normalizer:
            # EncoderNormalizer: chuẩn hóa target per-sequence (mean=0, std=1)
            # Phù hợp với log_return (range ~-0.15 to 0.15, không có drift dài hạn)
            from pytorch_forecasting.data import EncoderNormalizer
            target_normalizer = EncoderNormalizer(
                method="standard",   # z-score: (x - mean) / std
                center=True,
            )
            logger.info("Using EncoderNormalizer(method='standard')")
        else:
            from pytorch_forecasting.data import GroupNormalizer
            target_normalizer = GroupNormalizer(
                groups=[self.cfg.group_col],
                transformation=None,   # None thay softplus: tránh overflow với BTC price
                center=True,
                scale_by_group=True,
            )
            logger.info("Using GroupNormalizer(transformation=None)")

        training = TimeSeriesDataSet(
            data=df_train,
            time_idx="time_idx",
            target=self.cfg.target,
            group_ids=[self.cfg.group_col],
            min_encoder_length=max(1, self.cfg.max_encoder_length // 2),
            max_encoder_length=self.cfg.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.cfg.max_prediction_length,
            static_categoricals=spec["static_cats"],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=spec["known_reals"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=spec["unknown_reals"],
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            df_val,
            predict=True,
            stop_randomization=True,
        )

        self.training_dataset = training
        return training, validation

    def _make_dataloaders(
        self,
        training: Any,
        validation: Any,
    ) -> Tuple[Any, Any]:
        train_dl = training.to_dataloader(
            train=True,
            batch_size=self.cfg.batch_size,
            num_workers=0,
            persistent_workers=False,
        )
        val_dl = validation.to_dataloader(
            train=False,
            batch_size=self.cfg.batch_size * 2,
            num_workers=0,
            persistent_workers=False,
        )
        return train_dl, val_dl

    # ──────────────────────────────────────────────────────────────────────────
    # Model builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_model(
        self,
        dataset: Any,
        hparams: Optional[dict] = None,
    ) -> Any:
        """Tạo TFT model từ dataset + hparams."""
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        hp = hparams or {}
        model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=hp.get("learning_rate", self.cfg.learning_rate),
            hidden_size=hp.get("hidden_size", self.cfg.hidden_size),
            attention_head_size=hp.get("attention_head_size", self.cfg.attention_head_size),
            dropout=hp.get("dropout", self.cfg.dropout),
            hidden_continuous_size=hp.get("hidden_continuous_size", self.cfg.hidden_continuous_size),
            output_size=self.cfg.output_size,
            loss=QuantileLoss(quantiles=self.cfg.quantiles),
            log_interval=10,
            log_val_interval=1,
            # FIX: tăng patience để LR scheduler không giảm LR quá sớm
            # (với log_return target và dataset nhỏ, loss dao động nhiều hơn)
            reduce_on_plateau_patience=5,
        )
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"TFT model: {n_params:,} parameters")
        return model

    def _build_trainer(
        self,
        ckpt_dir: Optional[str] = None,
        max_epochs: Optional[int] = None,
        ckpt_prefix: str = "tft",
    ) -> Any:
        """Tạo PyTorch Lightning Trainer với callbacks."""
        import torch
        try:
            from lightning.pytorch import Trainer
            from lightning.pytorch.callbacks import (
                EarlyStopping,
                LearningRateMonitor,
                ModelCheckpoint,
            )
        except ImportError:
            from pytorch_lightning import Trainer
            from pytorch_lightning.callbacks import (
                EarlyStopping,
                LearningRateMonitor,
                ModelCheckpoint,
            )

        _ckpt_dir = ckpt_dir or self.cfg.checkpoint_dir
        _epochs = max_epochs or self.cfg.max_epochs
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.early_stopping_patience,
                mode="min",
                verbose=False,
            ),
            ModelCheckpoint(
                dirpath=_ckpt_dir,
                filename=f"{ckpt_prefix}_{{epoch:03d}}_valloss{{val_loss:.4f}}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
            ),
        ]

        trainer = Trainer(
            max_epochs=_epochs,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=self.cfg.gradient_clip_val,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
        )
        logger.info(f"Trainer ready | accelerator={accelerator} | max_epochs={_epochs}")
        return trainer

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        hparams: Optional[dict] = None,
    ) -> dict:
        """
        Full training run.

        Sử dụng best_hparams từ Optuna nếu đã tune, hoặc hparams truyền vào,
        hoặc mặc định trong config.

        Parameters
        ----------
        df_train : prepared train DataFrame
        df_val   : prepared val DataFrame
        hparams  : override hparams (optional)

        Returns
        -------
        dict : {val_loss, best_checkpoint, n_epochs}
        """
        hp = hparams or self.best_hparams or {}

        training_ds, val_ds = self.build_datasets(df_train, df_val)
        train_dl, val_dl = self._make_dataloaders(training_ds, val_ds)

        self.model = self._build_model(training_ds, hp)
        self.trainer = self._build_trainer(ckpt_prefix="tft_final")

        logger.info("=" * 60)
        logger.info("Starting final TFT training ...")
        logger.info("=" * 60)

        self.trainer.fit(
            self.model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
        )

        best_ckpt = self.trainer.checkpoint_callback.best_model_path
        best_loss = self.trainer.checkpoint_callback.best_model_score
        n_epochs = self.trainer.current_epoch

        logger.info(f"Training done | val_loss={best_loss:.4f} | epochs={n_epochs}")
        logger.info(f"Best checkpoint: {best_ckpt}")

        return {
            "val_loss": float(best_loss or 0),
            "best_checkpoint": best_ckpt,
            "n_epochs": n_epochs,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Technique 1: Optuna Hyperparameter Optimisation
    # ──────────────────────────────────────────────────────────────────────────

    def tune_hyperparameters(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        n_trials: Optional[int] = None,
        timeout_sec: Optional[int] = None,
    ) -> dict:
        """
        [Kỹ thuật 1] Optuna Hyperparameter Optimisation.

        FIX v2: với log_return target và EncoderNormalizer, loss không còn NaN/Infinity.
        Search space được thu hẹp để phù hợp dataset nhỏ (~600 rows daily).

        Search space
        ------------
        - learning_rate        : [5e-4, 5e-3]  (log scale — conservative range)
        - hidden_size          : {16, 32, 64}   (bỏ 128/256: overfit với 600 rows)
        - attention_head_size  : {1, 2, 4}
        - dropout              : [0.05, 0.3]
        - hidden_continuous_size: {8, 16, 32}
        - gradient_clip_val    : [0.05, 0.5]   (tunable, log_return cần clip thấp)

        Mỗi trial train tối đa 20 epoch.

        Returns
        -------
        dict : best hyperparameters
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Cài optuna: pip install optuna")

        n_trials = n_trials or self.cfg.n_optuna_trials
        timeout = timeout_sec or self.cfg.optuna_timeout_sec

        training_ds, val_ds = self.build_datasets(df_train, df_val)
        train_dl, val_dl = self._make_dataloaders(training_ds, val_ds)

        # FIX: tăng trial_epochs để loss có đủ thời gian hội tụ
        trial_epochs = min(20, self.cfg.max_epochs)

        def objective(trial: "optuna.Trial") -> float:
            hp = {
                # FIX: range hẹp hơn, an toàn hơn cho log_return target
                "learning_rate": trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True),
                # FIX: bỏ 128, 256 — overfit với dataset 600 rows
                "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
                "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4]),
                "dropout": trial.suggest_float("dropout", 0.05, 0.3),
                "hidden_continuous_size": trial.suggest_categorical(
                    "hidden_continuous_size", [8, 16, 32]
                ),
                # FIX: gradient_clip_val tunable — ảnh hưởng lớn đến stability
                "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.05, 0.5, log=True),
            }
            try:
                model = self._build_model(training_ds, hp)
                # Override gradient_clip_val cho trainer
                _original_clip = self.cfg.gradient_clip_val
                self.cfg.gradient_clip_val = hp["gradient_clip_val"]
                trainer = self._build_trainer(
                    ckpt_dir=str(Path(self.cfg.checkpoint_dir) / f"trial_{trial.number}"),
                    max_epochs=trial_epochs,
                    ckpt_prefix=f"trial{trial.number}",
                )
                self.cfg.gradient_clip_val = _original_clip
                trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
                score = trainer.checkpoint_callback.best_model_score
                # FIX: reject nan/inf explicitly
                if score is None or not np.isfinite(float(score)):
                    logger.warning(f"Trial {trial.number}: non-finite score={score}, pruned.")
                    raise optuna.exceptions.TrialPruned()
                return float(score)
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as exc:
                logger.warning(f"Trial {trial.number} failed: {exc}")
                raise optuna.exceptions.TrialPruned()

        logger.info(f"Optuna HPO: {n_trials} trials, {trial_epochs} epochs each ...")
        study = optuna.create_study(
            direction="minimize",
            study_name="tft_hpo",
            # FIX: MedianPruner để dừng sớm các trial tệ
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # FIX: kiểm tra có trial thành công không
        completed = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
        if not completed:
            logger.warning(
                "Tất cả HPO trials đều fail. Dùng default hyperparameters.\n"
                "Gợi ý: kiểm tra log_return column có NaN không, hoặc dataset quá nhỏ."
            )
            self.best_hparams = {}
        else:
            self.best_hparams = study.best_params
            best_val = study.best_value
            logger.info(f"Best hparams : {self.best_hparams}")
            logger.info(f"Best val_loss: {best_val:.6f}")

        # Lưu HPO results
        hpo_path = Path(self.cfg.output_dir) / "hpo_results.json"
        with open(hpo_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_hparams": self.best_hparams,
                    "best_val_loss": study.best_value if completed else None,
                    "n_trials": len(study.trials),
                    "n_completed": len(completed),
                    "trials": [
                        {"number": t.number, "params": t.params, "val_loss": t.value}
                        for t in study.trials
                        if t.value is not None and np.isfinite(t.value)
                    ],
                },
                f,
                indent=2,
            )
        logger.info(f"HPO results saved → {hpo_path}")
        return self.best_hparams

    # ──────────────────────────────────────────────────────────────────────────
    # Technique 2: Walk-Forward Cross-Validation
    # ──────────────────────────────────────────────────────────────────────────

    def walk_forward_cv(
        self,
        df: pd.DataFrame,
        n_splits: Optional[int] = None,
    ) -> List[dict]:
        """
        [Kỹ thuật 2] Walk-Forward Cross-Validation.

        Phân chia chronological (không data leakage):
          Fold 1: train [0→T1)  | val [T1→T2)
          Fold 2: train [0→T2)  | val [T2→T3)
          ...

        Mỗi fold train tối đa 20 epoch.

        Returns
        -------
        List[dict] : metrics từng fold {fold, train_size, val_size, val_loss}
        """
        n_splits = n_splits or self.cfg.n_cv_folds
        time_idx_vals = sorted(df["time_idx"].unique())
        n = len(time_idx_vals)

        # Tính fold boundaries
        min_train_len = self.cfg.max_encoder_length * 3
        available = n - min_train_len
        if available <= 0:
            logger.error("Không đủ dữ liệu cho Walk-Forward CV.")
            return []

        fold_size = available // n_splits
        if fold_size < self.cfg.max_prediction_length:
            n_splits = max(2, available // max(self.cfg.max_prediction_length, 1))
            fold_size = available // n_splits
            logger.warning(f"Giảm số fold xuống {n_splits}")

        self.cv_results = []
        fold_epochs = min(20, self.cfg.max_epochs)

        for fold in range(n_splits):
            # Boundaries của fold
            train_end_pos = min_train_len + fold * fold_size
            val_start_pos = train_end_pos
            val_end_pos = min(val_start_pos + fold_size, n)

            if val_end_pos >= n:
                break

            train_max_idx = time_idx_vals[train_end_pos - 1]
            val_min_idx = time_idx_vals[val_start_pos]
            val_max_idx = time_idx_vals[val_end_pos - 1]

            # Cần thêm encoder context vào validation
            context_pos = max(0, val_start_pos - self.cfg.max_encoder_length)
            context_min_idx = time_idx_vals[context_pos]

            df_fold_train = df[df["time_idx"] <= train_max_idx].copy()
            df_fold_val = df[
                (df["time_idx"] >= context_min_idx) &
                (df["time_idx"] <= val_max_idx)
            ].copy()

            logger.info(
                f"[Fold {fold+1}/{n_splits}] "
                f"train: {len(df_fold_train)} rows | "
                f"val: {len(df_fold_val)} rows"
            )

            try:
                training_ds, val_ds = self.build_datasets(df_fold_train, df_fold_val)
                train_dl, val_dl = self._make_dataloaders(training_ds, val_ds)
                model = self._build_model(training_ds, self.best_hparams)
                trainer = self._build_trainer(
                    ckpt_dir=str(Path(self.cfg.checkpoint_dir) / f"cv_fold{fold+1}"),
                    max_epochs=fold_epochs,
                    ckpt_prefix=f"cv{fold+1}",
                )
                trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

                val_loss = float(trainer.checkpoint_callback.best_model_score or 0)
                result = {
                    "fold": fold + 1,
                    "train_rows": len(df_fold_train),
                    "val_rows": len(df_fold_val),
                    "val_loss": val_loss,
                    "train_end_time_idx": int(train_max_idx),
                    "val_start_time_idx": int(val_min_idx),
                }
                self.cv_results.append(result)
                logger.info(f"Fold {fold+1} val_loss = {val_loss:.4f}")

            except Exception as exc:
                logger.error(f"Fold {fold+1} lỗi: {exc}")
                self.cv_results.append({"fold": fold + 1, "error": str(exc)})

        # Summary
        valid = [r for r in self.cv_results if "val_loss" in r]
        if valid:
            losses = [r["val_loss"] for r in valid]
            logger.info(
                f"CV Summary: mean={np.mean(losses):.4f} ± {np.std(losses):.4f} "
                f"| folds={len(valid)}"
            )
        return self.cv_results

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def _load_checkpoint(self, ckpt_path: str) -> Any:
        """Load TFT từ checkpoint file."""
        from pytorch_forecasting import TemporalFusionTransformer
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
        logger.info(f"Loaded: {Path(ckpt_path).name}")
        return model

    def _predict_single(
        self,
        model: Any,
        df_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Dự báo với một model cụ thể.

        df_test phải chứa ít nhất max_encoder_length rows lịch sử
        (thường là phần đuôi của train + test rows).

        Returns
        -------
        np.ndarray shape (n_windows, pred_length, n_quantiles)
        """
        from pytorch_forecasting import TimeSeriesDataSet

        test_ds = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            df_test,
            predict=True,
            stop_randomization=True,
        )
        test_dl = test_ds.to_dataloader(
            train=False,
            batch_size=self.cfg.batch_size * 2,
            num_workers=0,
            persistent_workers=False,
        )

        import torch
        model.eval()
        with torch.no_grad():
            raw = model.predict(test_dl, mode="raw", return_x=True)

        # shape: (n_windows, pred_length, n_quantiles)
        pred_np = raw.output.prediction.cpu().numpy()
        return pred_np

    def predict(
        self,
        df_test: pd.DataFrame,
        model: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        [Kỹ thuật 3] Quantile Forecasting.

        Dự báo tất cả các quantile đã cấu hình.
        Nếu target == "log_return", tự động thêm cột close_reconstructed
        bằng cách tích lũy log_return từ giá close cuối cùng của train.

        Parameters
        ----------
        df_test : DataFrame test (đã prepare_dataframe)
                  Phải bao gồm encoder history (tail của train)
        model   : dùng model cụ thể (mặc định: self.model)

        Returns
        -------
        pd.DataFrame columns: [window, step, q02, q10, q25, q50, q75, q90, q98]
                               + close_q50 (nếu target == log_return)
        """
        _model = model or self.model
        if _model is None:
            raise RuntimeError("Chưa có model. Gọi train() trước.")

        pred_np = self._predict_single(_model, df_test)
        qnames = [f"q{int(q*100):02d}" for q in self.cfg.quantiles]

        rows = []
        n_windows, pred_len, _ = pred_np.shape
        for w in range(n_windows):
            for t in range(pred_len):
                row = {"window": w, "step": t + 1}
                for qi, qn in enumerate(qnames):
                    row[qn] = float(pred_np[w, t, qi])
                rows.append(row)

        df_pred = pd.DataFrame(rows)

        # ── Reconstruct close price từ log_return predictions ───────────────
        df_pred = self._reconstruct_close(df_pred, df_test)

        return df_pred

    def _reconstruct_close(
        self,
        df_pred: pd.DataFrame,
        df_context: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Tái tạo giá close tuyệt đối từ log_return predictions.

        Chỉ áp dụng khi target == "log_return".
        Dùng giá close cuối cùng của encoder context (hoặc _last_train_close)
        làm điểm khởi đầu.

        Công thức: close[t] = close[t-1] * exp(log_return[t])
        Áp dụng cho từng window độc lập (step 1→max_prediction_length).

        Returns
        -------
        df_pred với thêm cột: close_q50, close_q10, close_q90
        """
        if self.cfg.target != "log_return":
            return df_pred
        if "q50" not in df_pred.columns:
            return df_pred

        # Lấy giá close khởi đầu cho mỗi window
        base_closes: dict = {}  # window_id → base_close

        if df_context is not None and self.cfg.close_col in df_context.columns:
            # Mỗi window bắt đầu từ giá close cuối của encoder context
            # Sắp xếp theo time_idx để lấy close đúng cho mỗi window
            sorted_ctx = df_context.sort_values("time_idx")
            # Lấy unique windows từ df_pred
            for w in df_pred["window"].unique():
                # Window w tương ứng với timestep w trong df_context
                # (mỗi window shift thêm 1 bước)
                ctx_idx = min(w + self.cfg.max_encoder_length - 1, len(sorted_ctx) - 1)
                if ctx_idx >= 0:
                    base_closes[w] = float(sorted_ctx[self.cfg.close_col].iloc[ctx_idx])
        
        # Fallback: dùng last_train_close
        if not base_closes and self._last_train_close is not None:
            for w in df_pred["window"].unique():
                base_closes[w] = self._last_train_close

        if not base_closes:
            logger.warning("Không có base close price để reconstruct. Bỏ qua reconstruction.")
            return df_pred

        # Reconstruct per window
        close_q50_list = []
        close_q10_list = []
        close_q90_list = []

        for w in df_pred["window"].unique():
            w_rows = df_pred[df_pred["window"] == w].sort_values("step")
            base = base_closes.get(w, self._last_train_close or 1.0)
            prev_close = base
            prev_close_q10 = base
            prev_close_q90 = base

            for _, row in w_rows.iterrows():
                c50 = prev_close * np.exp(row["q50"])
                c10 = prev_close_q10 * np.exp(row.get("q10", row["q50"]))
                c90 = prev_close_q90 * np.exp(row.get("q90", row["q50"]))
                close_q50_list.append(c50)
                close_q10_list.append(c10)
                close_q90_list.append(c90)
                prev_close = c50
                prev_close_q10 = c10
                prev_close_q90 = c90

        df_out = df_pred.copy()
        df_out["close_q50"] = close_q50_list
        df_out["close_q10"] = close_q10_list
        df_out["close_q90"] = close_q90_list

        logger.info(
            f"Close reconstruction | sample close_q50: "
            f"{df_out['close_q50'].iloc[0]:.2f} → {df_out['close_q50'].iloc[-1]:.2f}"
        )
        return df_out

    # ──────────────────────────────────────────────────────────────────────────
    # Technique 4: Checkpoint Ensemble
    # ──────────────────────────────────────────────────────────────────────────

    def ensemble_predict(
        self,
        df_test: pd.DataFrame,
        n_best: int = 3,
    ) -> pd.DataFrame:
        """
        [Kỹ thuật 4] Checkpoint Ensemble.

        Load top-k checkpoints (theo val_loss trong tên file),
        dự báo với từng checkpoint, rồi trung bình hoá kết quả.
        Nếu target == "log_return", thêm close_q50/close_q10/close_q90.

        Parameters
        ----------
        df_test : DataFrame test (đã prepare_dataframe + encoder history)
        n_best  : số checkpoints tốt nhất dùng để ensemble

        Returns
        -------
        pd.DataFrame với quantile predictions đã được ensemble
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        # Tìm checkpoints từ final training (prefix "tft_final")
        checkpoints = sorted(
            list(ckpt_dir.glob("tft_final_*.ckpt")),
            key=lambda p: self._extract_val_loss(p.stem),
        )[:n_best]

        if not checkpoints:
            logger.warning("Không tìm thấy checkpoint. Dùng self.model.")
            return self.predict(df_test)

        logger.info(f"Ensembling {len(checkpoints)} checkpoints:")
        for c in checkpoints:
            logger.info(f"  {c.name}")

        all_preds: List[np.ndarray] = []
        for ckpt in checkpoints:
            try:
                m = self._load_checkpoint(str(ckpt))
                pred_np = self._predict_single(m, df_test)
                all_preds.append(pred_np)
            except Exception as exc:
                logger.warning(f"Bỏ qua {ckpt.name}: {exc}")

        if not all_preds:
            return self.predict(df_test)

        # Average ensemble
        ensemble_np = np.mean(all_preds, axis=0)  # (n_windows, pred_len, n_quantiles)

        qnames = [f"q{int(q*100):02d}" for q in self.cfg.quantiles]
        rows = []
        n_windows, pred_len, _ = ensemble_np.shape
        for w in range(n_windows):
            for t in range(pred_len):
                row = {"window": w, "step": t + 1}
                for qi, qn in enumerate(qnames):
                    row[qn] = float(ensemble_np[w, t, qi])
                rows.append(row)

        df_pred = pd.DataFrame(rows)

        # FIX: reconstruct close price sau ensemble
        df_pred = self._reconstruct_close(df_pred, df_test)

        logger.info(f"Ensemble complete | models={len(all_preds)} | rows={len(df_pred)}")
        return df_pred

    @staticmethod
    def _extract_val_loss(stem: str) -> float:
        """Parse val_loss từ tên file checkpoint."""
        try:
            return float(stem.split("valloss")[-1])
        except Exception:
            return float("inf")

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.Series, List[float]],
        predictions: pd.DataFrame,
    ) -> dict:
        """
        Tính evaluation metrics.

        Point metrics   : MAE, RMSE, MAPE, sMAPE  (dùng q50 làm point forecast)
        Interval metrics: Winkler Score (80%), Coverage Rate (80% và 96%)

        Nếu target == "log_return" và predictions có cột "close_q50",
        dùng close_q50 làm point forecast để tính metrics trên giá tuyệt đối.
        Nếu y_true là giá tuyệt đối (close), evaluate trên close_q50.

        Parameters
        ----------
        y_true      : giá trị thực tế — nên là close (tuyệt đối) để dễ so sánh
        predictions : DataFrame từ predict() hoặc ensemble_predict()

        Returns
        -------
        dict : tất cả metrics
        """
        y = np.asarray(y_true).flatten()

        # Chọn point forecast column
        step1 = predictions[predictions["step"] == 1]

        # Ưu tiên close_q50 nếu target là log_return (metrics trên giá tuyệt đối dễ đọc hơn)
        if "close_q50" in predictions.columns and len(step1) > 0:
            y_hat_raw = step1["close_q50"].values
            lo_col = "close_q10"
            hi_col = "close_q90"
            logger.info("Evaluating on reconstructed close prices (close_q50)")
        elif "q50" in predictions.columns:
            y_hat_raw = step1["q50"].values
            lo_col = "q10"
            hi_col = "q90"
        else:
            col = predictions.columns[-1]
            y_hat_raw = predictions[col].values[:len(y)]
            lo_col = hi_col = None

        n = min(len(y), len(y_hat_raw))
        y_hat = y_hat_raw[:n]
        y = y[:n]

        metrics: dict = {
            "R2": _r2(y, y_hat),  # R² as first metric
            "MAE": _mae(y, y_hat),
            "RMSE": _rmse(y, y_hat),
            "MAPE_%": _mape(y, y_hat),
            "sMAPE_%": _smape(y, y_hat),
        }

        # 80% interval
        if lo_col and hi_col and lo_col in step1.columns and hi_col in step1.columns:
            lo = step1[lo_col].values[:n]
            hi = step1[hi_col].values[:n]
            metrics["Winkler_80"] = _winkler(y, lo, hi, alpha=0.2)
            metrics["Coverage_80_%"] = _coverage(y, lo, hi) * 100

        # 96% interval: q02 → q98 (chỉ khi dùng log_return quantile trực tiếp)
        if "q02" in step1.columns and "q98" in step1.columns and "close_q50" not in step1.columns:
            lo96 = step1["q02"].values[:n]
            hi96 = step1["q98"].values[:n]
            metrics["Winkler_96"] = _winkler(y, lo96, hi96, alpha=0.04)
            metrics["Coverage_96_%"] = _coverage(y, lo96, hi96) * 100

        logger.info("── Evaluation Metrics ─────────────────────")
        for k, v in metrics.items():
            logger.info(f"  {k:<20}: {v:.4f}")
        logger.info("───────────────────────────────────────────")

        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Interpretability
    # ──────────────────────────────────────────────────────────────────────────

    def get_variable_importance(self) -> dict:
        """
        Lấy variable importance từ TFT attention mechanism.

        Returns
        -------
        dict : {encoder_variables, decoder_variables, static_variables}
        """
        if self.model is None:
            raise RuntimeError("Chưa có model. Gọi train() trước.")
        if self.training_dataset is None:
            raise RuntimeError("Chưa có training_dataset.")

        import torch

        try:
            val_dl = self.training_dataset.to_dataloader(
                train=False, batch_size=128, num_workers=0
            )
            raw = self.model.predict(val_dl, mode="raw", return_x=True)
            interp = self.model.interpret_output(raw, reduction="sum")

            importance = {}
            # encoder variables
            if hasattr(interp, "encoder_variables") and interp.encoder_variables is not None:
                enc_names = self.model.hparams.get("time_varying_encoders_reals", [])
                enc_vals = interp.encoder_variables.cpu().numpy().tolist()
                importance["encoder_variables"] = dict(zip(enc_names, enc_vals))
            # static variables
            if hasattr(interp, "static_variables") and interp.static_variables is not None:
                importance["static_variables"] = interp.static_variables.cpu().numpy().tolist()

            return importance

        except Exception as exc:
            logger.warning(f"Variable importance thất bại: {exc}")
            return {}

    # ──────────────────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────────────────

    def save_results(
        self,
        output_path: str,
        y_true: Union[np.ndarray, pd.Series, List],
        predictions: pd.DataFrame,
        train_result: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """
        Tính metrics và lưu tất cả kết quả vào JSON.

        Output JSON structure
        ---------------------
        {
          "config"      : TFTConfig as dict,
          "best_hparams": {},
          "cv_results"  : [...],
          "train_result": {},
          "metrics"     : {MAE, RMSE, MAPE, sMAPE, Winkler_80, Coverage_80_%},
          "predictions" : {
            "y_true" : [...],
            "q02": [...], "q10": [...], "q25": [...],
            "q50": [...], "q75": [...], "q90": [...], "q98": [...],
            "step": [...], "window": [...]
          },
          ...extra
        }

        Parameters
        ----------
        output_path  : đường dẫn file JSON đầu ra
        y_true       : giá trị thực tế (test set target)
        predictions  : DataFrame từ ensemble_predict()
        train_result : dict trả về từ train()
        extra        : any dict muốn thêm vào output

        Returns
        -------
        dict : evaluation metrics
        """
        metrics = self.evaluate(y_true, predictions)
        y_list = np.asarray(y_true).flatten().tolist()

        output = {
            "config": self.cfg.to_dict(),
            "best_hparams": self.best_hparams or {},
            "cv_results": self.cv_results,
            "train_result": train_result or {},
            "metrics": metrics,
            "predictions": {
                col: predictions[col].tolist()
                for col in predictions.columns
            },
            "y_true": y_list,
        }

        if extra:
            output.update(extra)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results saved → {out_path}")
        return metrics

    @staticmethod
    def load_results(path: str) -> dict:
        """Load kết quả đã lưu từ JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
