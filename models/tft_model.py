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
>>> cfg = TFTConfig(target="close", max_encoder_length=60, max_prediction_length=7)
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
    target                : Tên cột cần dự báo, e.g. "close" hoặc "return_pct"
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
    """

    # ── Target & groups ──────────────────────────────────────────────────────
    target: str = "close"
    group_col: str = "symbol"
    time_col: str = "open_time"

    # ── Sequence lengths ─────────────────────────────────────────────────────
    max_encoder_length: int = 60
    max_prediction_length: int = 7

    # ── Features ─────────────────────────────────────────────────────────────
    known_real_features: List[str] = field(default_factory=lambda: [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "doy_sin", "doy_cos",
    ])
    unknown_real_features: List[str] = field(default_factory=lambda: [
        "return_pct", "log_return",
        "rsi_14", "macd_hist", "macd", "macd_signal",
        "bb_pct", "bb_width",
        "atr_14", "volume_ratio",
        "rolling_vol_30", "price_zscore", "drawdown_pct",
        "ma_7", "ma_25", "ema_12", "ema_26",
    ])

    # ── Quantile output ──────────────────────────────────────────────────────
    quantiles: List[float] = field(default_factory=lambda: [
        0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98
    ])

    # ── Training hyperparameters (defaults — tunable via Optuna) ─────────────
    batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 10
    gradient_clip_val: float = 0.1
    learning_rate: float = 0.03
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 16

    # ── Optuna ───────────────────────────────────────────────────────────────
    n_optuna_trials: int = 20
    optuna_timeout_sec: Optional[int] = None

    # ── Walk-forward CV ──────────────────────────────────────────────────────
    n_cv_folds: int = 5

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

    Quick start
    -----------
    >>> cfg = TFTConfig(target="close", max_encoder_length=60, max_prediction_length=7)
    >>> f = TFTForecaster(cfg)
    >>> df_train, df_val, df_test = f.load_and_split("train.json", "test.json")
    >>> f.tune_hyperparameters(df_train, df_val)
    >>> f.train(df_train, df_val)
    >>> preds = f.ensemble_predict(df_test)
    >>> f.save_results("results.json", y_true=df_test[cfg.target], predictions=preds)
    """

    def __init__(self, config: TFTConfig) -> None:
        self.cfg = config
        self.best_hparams: Optional[dict] = None
        self.trainer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.training_dataset: Optional[Any] = None
        self.cv_results: List[dict] = []

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"TFTForecaster ready | target={config.target} | "
            f"encoder={config.max_encoder_length} | pred_len={config.max_prediction_length} | "
            f"quantiles={config.quantiles}"
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

        # Split train/val (chronological)
        cutoff = int(len(df_full) * (1 - val_ratio))
        # Align time_idx so val continues from train
        df_train = df_full.iloc[:cutoff].copy()
        df_val = df_full.iloc[cutoff:].copy()

        # Rebuild time_idx for val (must be contiguous with train for from_dataset)
        # Keep the original time_idx from df_full — they are already global

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

        Returns
        -------
        (training_dataset, validation_dataset)
        """
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        spec = self._feature_spec(df_train)

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
            target_normalizer=GroupNormalizer(
                groups=[self.cfg.group_col],
                transformation="softplus",
            ),
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
            reduce_on_plateau_patience=4,
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

        Search space
        ------------
        - learning_rate       : [1e-4, 0.1]  (log scale)
        - hidden_size         : {16, 32, 64, 128}
        - attention_head_size : {1, 2, 4}
        - dropout             : [0.05, 0.4]
        - hidden_continuous_size : {8, 16, 32}

        Mỗi trial train tối đa 15 epoch để tiết kiệm thời gian.
        Kết quả được lưu vào output_dir/hpo_results.json.

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

        trial_epochs = min(15, self.cfg.max_epochs)

        def objective(trial: "optuna.Trial") -> float:
            hp = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
                "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
                "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4]),
                "dropout": trial.suggest_float("dropout", 0.05, 0.4),
                "hidden_continuous_size": trial.suggest_categorical(
                    "hidden_continuous_size", [8, 16, 32]
                ),
            }
            try:
                model = self._build_model(training_ds, hp)
                trainer = self._build_trainer(
                    ckpt_dir=str(Path(self.cfg.checkpoint_dir) / f"trial_{trial.number}"),
                    max_epochs=trial_epochs,
                    ckpt_prefix=f"trial{trial.number}",
                )
                trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
                score = trainer.checkpoint_callback.best_model_score
                return float(score) if score is not None else float("inf")
            except Exception as exc:
                logger.warning(f"Trial {trial.number} failed: {exc}")
                return float("inf")

        logger.info(f"Optuna HPO: {n_trials} trials, {trial_epochs} epochs each ...")
        study = optuna.create_study(direction="minimize", study_name="tft_hpo")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        self.best_hparams = study.best_params
        best_val = study.best_value

        logger.info(f"Best hparams : {self.best_hparams}")
        logger.info(f"Best val_loss: {best_val:.4f}")

        # Lưu HPO results
        hpo_path = Path(self.cfg.output_dir) / "hpo_results.json"
        with open(hpo_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_hparams": self.best_hparams,
                    "best_val_loss": best_val,
                    "n_trials": len(study.trials),
                    "trials": [
                        {"number": t.number, "params": t.params, "val_loss": t.value}
                        for t in study.trials
                        if t.value is not None
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

        Parameters
        ----------
        df_test : DataFrame test (đã prepare_dataframe)
                  Phải bao gồm encoder history (tail của train)
        model   : dùng model cụ thể (mặc định: self.model)

        Returns
        -------
        pd.DataFrame columns: [window, step, q02, q10, q25, q50, q75, q90, q98]
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

        return pd.DataFrame(rows)

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

        Parameters
        ----------
        y_true      : giá trị thực tế (aligned với q50 predictions)
        predictions : DataFrame từ predict() hoặc ensemble_predict()

        Returns
        -------
        dict : tất cả metrics
        """
        y = np.asarray(y_true).flatten()

        # Lấy q50 làm point forecast (một step cho mỗi window)
        if "q50" in predictions.columns:
            # Lấy step=1 từ mỗi window → point forecast
            step1 = predictions[predictions["step"] == 1]["q50"].values
            n = min(len(y), len(step1))
            y_hat = step1[:n]
            y = y[:n]
        else:
            col = predictions.columns[-1]
            y_hat = predictions[col].values[:len(y)]

        metrics: dict = {
            "MAE": _mae(y, y_hat),
            "RMSE": _rmse(y, y_hat),
            "MAPE_%": _mape(y, y_hat),
            "sMAPE_%": _smape(y, y_hat),
        }

        # 80% interval: q10 → q90
        if "q10" in predictions.columns and "q90" in predictions.columns:
            step1_rows = predictions[predictions["step"] == 1]
            lo = step1_rows["q10"].values[:len(y)]
            hi = step1_rows["q90"].values[:len(y)]
            metrics["Winkler_80"] = _winkler(y, lo, hi, alpha=0.2)
            metrics["Coverage_80_%"] = _coverage(y, lo, hi) * 100

        # 96% interval: q02 → q98
        if "q02" in predictions.columns and "q98" in predictions.columns:
            step1_rows = predictions[predictions["step"] == 1]
            lo96 = step1_rows["q02"].values[:len(y)]
            hi96 = step1_rows["q98"].values[:len(y)]
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
