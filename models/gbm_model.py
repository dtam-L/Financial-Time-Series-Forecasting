"""
models/gbm_model.py
===================
XGBoost + LightGBM Stacking Ensemble cho Financial Time Series Forecasting.

Kỹ thuật tăng độ chính xác (KHÁC với TFT)
-----------------------------------------
  1. Auto Lag-Feature Engineering  — tự động tạo lag, rolling, EWMA từ target
  2. Optuna + MedianPruner         — Bayesian HPO với early trial pruning
  3. Stacking Ensemble (OOF)       — XGB + LGB base → Ridge meta-learner
  4. Conformal Prediction Intervals— non-parametric intervals với coverage guarantee

Input data format (JSON)
------------------------
  [{"open_time": "...", "symbol": "BTC/USDT", "close": 42000.0,
    "rsi_14": 58.3, "macd_hist": 123.4, ...}, ...]

Usage
-----
>>> from models.gbm_model import GBMConfig, GBMForecaster
>>> cfg = GBMConfig(target="close", max_prediction_length=7)
>>> f = GBMForecaster(cfg)
>>> df_train, df_cal, df_test = f.load_and_split("train.json", "test.json")
>>> df_train = f.build_lag_features(df_train)
>>> f.tune_all(df_train)                    # Optuna + MedianPruner
>>> f.fit_stacking(df_train, df_cal)        # OOF Stacking
>>> f.calibrate_conformal(df_cal)           # Conformal intervals
>>> result = f.recursive_predict(df_test, steps=cfg.max_prediction_length)
>>> f.save_results("results/gbm_results.json", y_true, result)
"""

from __future__ import annotations

import json
import logging
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
logger = logging.getLogger("GBMForecaster")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GBMConfig:
    """
    Configuration cho GBMForecaster.

    Attributes
    ----------
    target                : Cột cần dự báo
    group_col             : Cột symbol/group
    time_col              : Cột thời gian
    lags                  : Các bậc lag của target
    rolling_windows       : Cửa sổ rolling statistics
    ewma_spans            : Spans cho EWMA features
    max_prediction_length : Số bước dự báo (recursive)
    n_optuna_trials       : Số trials Optuna HPO (mỗi model)
    n_cv_folds            : Số fold cho OOF stacking
    conformal_alpha       : Mức tin cậy conformal intervals (0.1 → 90% coverage)
    calibration_ratio     : Tỷ lệ train dùng làm calibration set
    val_ratio             : Tỷ lệ train dùng làm val (trong Optuna)
    output_dir            : Thư mục lưu kết quả
    """

    # ── Target ───────────────────────────────────────────────────────────────
    target: str = "close"
    group_col: str = "symbol"
    time_col: str = "open_time"

    # ── Kỹ thuật 1: Lag Features ─────────────────────────────────────────────
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7, 14, 21])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    ewma_spans: List[int] = field(default_factory=lambda: [7, 14])

    # ── External features (từ feature engineering đã làm) ────────────────────
    external_features: List[str] = field(default_factory=lambda: [
        "rsi_14", "macd_hist", "macd", "macd_signal",
        "bb_pct", "bb_width", "atr_14", "volume_ratio",
        "rolling_vol_30", "price_zscore", "drawdown_pct",
        "ma_7", "ma_25", "ema_12", "ema_26",
        "return_pct", "log_return",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "doy_sin", "doy_cos",
    ])

    # ── Forecast horizon ─────────────────────────────────────────────────────
    max_prediction_length: int = 7

    # ── Kỹ thuật 2: Optuna + MedianPruner ────────────────────────────────────
    n_optuna_trials: int = 30
    optuna_timeout_sec: Optional[int] = None
    optuna_cv_folds: int = 3          # inner CV folds trong objective

    # ── Kỹ thuật 3: OOF Stacking ─────────────────────────────────────────────
    n_cv_folds: int = 5               # outer CV folds cho OOF
    meta_learner: str = "ridge"       # "ridge" | "lasso" | "elasticnet"

    # ── Kỹ thuật 4: Conformal Prediction ─────────────────────────────────────
    conformal_alpha: float = 0.1      # 0.1 → 90% coverage
    calibration_ratio: float = 0.15   # % cuối train dùng làm calibration

    # ── Data split ───────────────────────────────────────────────────────────
    val_ratio: float = 0.1

    # ── Paths ────────────────────────────────────────────────────────────────
    output_dir: str = "gbm_output"

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics  (dùng chung với gbm_model, không import từ tft_model)
# ══════════════════════════════════════════════════════════════════════════════

def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))

def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def _mape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y - yhat) / (np.abs(y) + eps))) * 100)

def _smape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> float:
    d = (np.abs(y) + np.abs(yhat)) / 2.0 + eps
    return float(np.mean(np.abs(y - yhat) / d) * 100)

def _winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    w = hi - lo
    pen = np.where(y < lo, 2 / alpha * (lo - y),
                   np.where(y > hi, 2 / alpha * (y - hi), 0.0))
    return float(np.mean(w + pen))

def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class GBMForecaster:
    """
    XGBoost + LightGBM Stacking Ensemble Forecaster.

    Accuracy-boosting techniques (khác với TFT)
    -------------------------------------------
    1. Auto Lag-Feature Engineering   → build_lag_features()
    2. Optuna + MedianPruner HPO       → tune_all()
    3. OOF Stacking Ensemble           → fit_stacking()
    4. Conformal Prediction Intervals  → calibrate_conformal() + get_intervals()

    Quick start
    -----------
    >>> cfg = GBMConfig(target="close", max_prediction_length=7)
    >>> f = GBMForecaster(cfg)
    >>> df_train, df_cal, df_test = f.load_and_split("train.json", "test.json")
    >>> df_train_feat = f.build_lag_features(df_train)
    >>> f.tune_all(df_train_feat)
    >>> f.fit_stacking(df_train_feat, f.build_lag_features(df_cal))
    >>> f.calibrate_conformal(f.build_lag_features(df_cal))
    >>> result = f.recursive_predict(df_test, steps=cfg.max_prediction_length)
    >>> f.save_results("results.json", y_true, result)
    """

    def __init__(self, config: GBMConfig) -> None:
        self.cfg = config

        # Models
        self.xgb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.meta_model: Optional[Any] = None

        # HPO results
        self.best_xgb_params: dict = {}
        self.best_lgb_params: dict = {}

        # Conformal state
        self._conformal_residuals: Optional[np.ndarray] = None

        # Feature info
        self.feature_names_: Optional[List[str]] = None

        # Results log
        self.oof_scores_: List[dict] = []

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"GBMForecaster ready | target={config.target} | "
            f"pred_len={config.max_prediction_length} | "
            f"lags={config.lags} | rolling={config.rolling_windows}"
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
        """Chuẩn bị DataFrame: parse datetime, sort, reset index."""
        out = df.copy()

        if group_col not in out.columns:
            out[group_col] = default_symbol
        out[group_col] = out[group_col].astype(str)

        if time_col in out.columns:
            out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
            out = out.dropna(subset=[time_col])
        else:
            out[time_col] = pd.to_datetime(out.index, utc=True, errors="coerce")

        out = out.sort_values([group_col, time_col]).reset_index(drop=True)
        out = out.ffill().fillna(0)
        return out

    @staticmethod
    def load_json(path: str) -> pd.DataFrame:
        """Load JSON (orient=records) thành DataFrame."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        for key in ("data", "records", "train", "test"):
            if key in data:
                return pd.DataFrame(data[key])
        return pd.DataFrame(data)

    def load_and_split(
        self,
        train_path: str,
        test_path: str,
        calibration_ratio: Optional[float] = None,
        symbol: str = "BTC/USDT",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train/test JSON, tách calibration set từ cuối train.

        Returns
        -------
        (df_train, df_cal, df_test) — đã prepare_dataframe, chưa lag features
        """
        cal_ratio = calibration_ratio or self.cfg.calibration_ratio

        logger.info(f"Loading train: {train_path}")
        df_tr_raw = self.load_json(train_path)
        logger.info(f"Loading test:  {test_path}")
        df_te_raw = self.load_json(test_path)

        df_train_full = self.prepare_dataframe(
            df_tr_raw, self.cfg.group_col, self.cfg.time_col, symbol
        )
        df_test = self.prepare_dataframe(
            df_te_raw, self.cfg.group_col, self.cfg.time_col, symbol
        )

        # Tách calibration set từ cuối train (để dùng cho conformal prediction)
        cal_start = int(len(df_train_full) * (1 - cal_ratio))
        df_train = df_train_full.iloc[:cal_start].copy().reset_index(drop=True)
        df_cal = df_train_full.iloc[cal_start:].copy().reset_index(drop=True)

        logger.info(
            f"Split | train={len(df_train)} | calibration={len(df_cal)} | test={len(df_test)}"
        )
        return df_train, df_cal, df_test

    # ──────────────────────────────────────────────────────────────────────────
    # Technique 1: Auto Lag-Feature Engineering
    # ──────────────────────────────────────────────────────────────────────────

    def build_lag_features(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        [Kỹ thuật 1] Tự động tạo lag + rolling + EWMA features từ target.

        Features được tạo
        -----------------
        Lag          : target_lag_1, target_lag_2, ..., target_lag_21
        Rolling mean : target_rolling_mean_7, _14, _30
        Rolling std  : target_rolling_std_7, _14, _30
        Rolling min  : target_rolling_min_7, _14
        Rolling max  : target_rolling_max_7, _14
        EWMA         : target_ewma_7, target_ewma_14
        Diff         : target_diff_1, target_diff_7
        Momentum     : target_mom_7  (target - target_lag_7)

        Parameters
        ----------
        df       : DataFrame đã prepare_dataframe
        drop_na  : Bỏ các rows NaN do lag (mặc định True)

        Returns
        -------
        pd.DataFrame với thêm lag/rolling features
        """
        out = df.copy()
        tgt = self.cfg.target

        if tgt not in out.columns:
            raise ValueError(f"Cột target '{tgt}' không tồn tại trong DataFrame")

        series = out[tgt]

        # Lags
        for lag in self.cfg.lags:
            out[f"{tgt}_lag_{lag}"] = series.shift(lag)

        # Rolling statistics
        for w in self.cfg.rolling_windows:
            roll = series.rolling(w, min_periods=max(1, w // 2))
            out[f"{tgt}_roll_mean_{w}"] = roll.mean()
            out[f"{tgt}_roll_std_{w}"]  = roll.std()
            if w <= 14:
                out[f"{tgt}_roll_min_{w}"] = roll.min()
                out[f"{tgt}_roll_max_{w}"] = roll.max()

        # EWMA
        for span in self.cfg.ewma_spans:
            out[f"{tgt}_ewma_{span}"] = series.ewm(span=span, adjust=False).mean()

        # Differences & momentum
        out[f"{tgt}_diff_1"] = series.diff(1)
        out[f"{tgt}_diff_7"] = series.diff(7)
        out[f"{tgt}_mom_7"]  = series - series.shift(7)

        if drop_na:
            out = out.dropna().reset_index(drop=True)

        lag_cols = [c for c in out.columns if c.startswith(f"{tgt}_")]
        logger.info(
            f"Lag features built: {len(lag_cols)} new columns | "
            f"rows: {len(df)} → {len(out)}"
        )
        return out

    def _make_xy(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Tạo (X, y, feature_names) cho supervised learning.

        y = target shifted by `horizon` (predict horizon steps ahead).
        X = tất cả features trừ target gốc và time/group columns.
        """
        exclude = {
            self.cfg.target,
            self.cfg.group_col,
            self.cfg.time_col,
            "open_time", "symbol",
        }

        feat_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]

        # Tất cả external features có trong df
        ext_available = [c for c in self.cfg.external_features if c in df.columns]
        all_features = list(dict.fromkeys(feat_cols + ext_available))  # unique, preserve order

        y = df[self.cfg.target].shift(-horizon).values
        X = df[all_features].values

        # Loại bỏ rows cuối bị NaN do shift
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]

        self.feature_names_ = all_features
        return X.astype(np.float32), y.astype(np.float32), all_features

    # ──────────────────────────────────────────────────────────────────────────
    # Technique 2: Optuna + MedianPruner HPO
    # ──────────────────────────────────────────────────────────────────────────

    def _tune_xgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int,
    ) -> dict:
        """
        [Kỹ thuật 2a] Optuna HPO cho XGBoost với MedianPruner.

        MedianPruner dừng sớm các trial có val_loss
        tệ hơn median của n_startup_trials trước.

        Search space
        ------------
        n_estimators     : [200, 2000]
        max_depth        : [3, 10]
        learning_rate    : [0.005, 0.3] log-scale
        subsample        : [0.5, 1.0]
        colsample_bytree : [0.5, 1.0]
        reg_alpha        : [1e-8, 10] log-scale
        reg_lambda       : [1e-8, 10] log-scale
        min_child_weight : [1, 10]
        """
        try:
            import optuna
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_squared_error
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(f"Cài thư viện: pip install xgboost optuna scikit-learn ({e})")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                "verbosity": 0,
            }
            tscv = TimeSeriesSplit(n_splits=self.cfg.optuna_cv_folds)
            fold_scores = []

            for fold_idx, (tr_idx, vl_idx) in enumerate(tscv.split(X)):
                X_tr, X_vl = X[tr_idx], X[vl_idx]
                y_tr, y_vl = y[tr_idx], y[vl_idx]

                # XGBoost >= 2.0: early_stopping dung callbacks
                from xgboost.callback import EarlyStopping
                model = XGBRegressor(**params, callbacks=[EarlyStopping(50, save_best=True)])
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_vl, y_vl)],
                    verbose=False,
                )
                pred = model.predict(X_vl)
                rmse = float(np.sqrt(mean_squared_error(y_vl, pred)))
                fold_scores.append(rmse)

                # Report intermediate -> MedianPruner kiem tra
                trial.report(np.mean(fold_scores), step=fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(fold_scores))

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        )
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name="xgb_hpo",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        logger.info(f"XGB best params: {best} | val_RMSE={study.best_value:.4f}")
        return best

    def _tune_lgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int,
    ) -> dict:
        """
        [Kỹ thuật 2b] Optuna HPO cho LightGBM với MedianPruner.

        Search space
        ------------
        n_estimators   : [200, 2000]
        num_leaves     : [20, 200]
        learning_rate  : [0.005, 0.3] log-scale
        subsample      : [0.5, 1.0]
        colsample_bytree : [0.5, 1.0]
        reg_alpha      : [1e-8, 10] log-scale
        reg_lambda     : [1e-8, 10] log-scale
        min_child_samples : [5, 50]
        """
        try:
            import optuna
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_squared_error
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(f"Cài thư viện: pip install lightgbm optuna ({e})")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            }
            tscv = TimeSeriesSplit(n_splits=self.cfg.optuna_cv_folds)
            fold_scores = []

            for fold_idx, (tr_idx, vl_idx) in enumerate(tscv.split(X)):
                X_tr, X_vl = X[tr_idx], X[vl_idx]
                y_tr, y_vl = y[tr_idx], y[vl_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_vl, y_vl)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
                )
                pred = model.predict(X_vl, num_iteration=model.best_iteration_)
                rmse = float(np.sqrt(mean_squared_error(y_vl, pred)))
                fold_scores.append(rmse)

                # Report intermediate → MedianPruner
                trial.report(np.mean(fold_scores), step=fold_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(fold_scores))

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        )
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name="lgb_hpo",
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        logger.info(f"LGB best params: {best} | val_RMSE={study.best_value:.4f}")
        return best

    def tune_all(
        self,
        df_train: pd.DataFrame,
        n_trials: Optional[int] = None,
    ) -> dict:
        """
        [Kỹ thuật 2] Tune cả XGBoost và LightGBM với Optuna + MedianPruner.

        Kết quả được lưu vào output_dir/hpo_results.json.

        Returns
        -------
        dict : {"xgb": best_xgb_params, "lgb": best_lgb_params}
        """
        n = n_trials or self.cfg.n_optuna_trials
        X, y, _ = self._make_xy(df_train)

        logger.info(f"Tuning XGBoost ({n} trials) ...")
        self.best_xgb_params = self._tune_xgb(X, y, n)

        logger.info(f"Tuning LightGBM ({n} trials) ...")
        self.best_lgb_params = self._tune_lgb(X, y, n)

        hpo_path = Path(self.cfg.output_dir) / "hpo_results.json"
        with open(hpo_path, "w", encoding="utf-8") as f:
            json.dump(
                {"xgb": self.best_xgb_params, "lgb": self.best_lgb_params},
                f,
                indent=2,
            )
        logger.info(f"HPO results saved → {hpo_path}")
        return {"xgb": self.best_xgb_params, "lgb": self.best_lgb_params}

    # ──────────────────────────────────────────────────────────────────────────
    # Technique 3: OOF Stacking Ensemble
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sinh OOF (Out-of-Fold) predictions từ XGBoost và LightGBM.

        Dùng TimeSeriesSplit để đảm bảo không leakage.
        OOF predictions dung lam meta-features de train Ridge.

        Returns
        -------
        (oof_xgb, oof_lgb) - shape: (n_valid_samples,)
        """
        from sklearn.model_selection import TimeSeriesSplit
        from xgboost import XGBRegressor
        from xgboost.callback import EarlyStopping as XGBEarlyStopping
        import lightgbm as lgb

        tscv = TimeSeriesSplit(n_splits=self.cfg.n_cv_folds)

        oof_xgb = np.full(len(y), np.nan)
        oof_lgb = np.full(len(y), np.nan)

        logger.info(f"Generating OOF predictions ({self.cfg.n_cv_folds} folds)...")

        for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X)):
            X_tr, X_vl = X[tr_idx], X[vl_idx]
            y_tr, y_vl = y[tr_idx], y[vl_idx]

            # XGBoost (v2+ dung callbacks cho early stopping)
            xgb_params = {**self.best_xgb_params, "random_state": 42,
                          "n_jobs": -1, "tree_method": "hist", "verbosity": 0,
                          "callbacks": [XGBEarlyStopping(50, save_best=True)]}
            xgb_m = XGBRegressor(**xgb_params)
            xgb_m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            oof_xgb[vl_idx] = xgb_m.predict(X_vl)

            # LightGBM
            lgb_params = {**self.best_lgb_params, "random_state": 42,
                          "n_jobs": -1, "verbosity": -1}
            lgb_m = lgb.LGBMRegressor(**lgb_params)
            lgb_m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            oof_lgb[vl_idx] = lgb_m.predict(X_vl, num_iteration=lgb_m.best_iteration_)

            from sklearn.metrics import mean_squared_error
            xgb_rmse = np.sqrt(mean_squared_error(y_vl, oof_xgb[vl_idx]))
            lgb_rmse = np.sqrt(mean_squared_error(y_vl, oof_lgb[vl_idx]))
            logger.info(
                f"  Fold {fold+1}/{self.cfg.n_cv_folds} | "
                f"XGB_RMSE={xgb_rmse:.4f} | LGB_RMSE={lgb_rmse:.4f}"
            )
            self.oof_scores_.append(
                {"fold": fold + 1, "xgb_rmse": xgb_rmse, "lgb_rmse": lgb_rmse}
            )

        # Loại bỏ NaN (đầu array — do TimeSeriesSplit không có fold 0 val)
        valid = ~np.isnan(oof_xgb) & ~np.isnan(oof_lgb)
        return oof_xgb[valid], oof_lgb[valid], y[valid]

    def fit_stacking(
        self,
        df_train: pd.DataFrame,
        df_cal: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        [Kỹ thuật 3] Fit Stacking Ensemble: XGB + LGB → Ridge meta-learner.

        Pipeline
        --------
        1. Generate OOF predictions (TimeSeriesSplit, no leakage)
        2. Train Ridge meta-learner trên OOF matrix [xgb_oof | lgb_oof]
        3. Retrain XGB + LGB trên toàn bộ train data
        4. Lưu feature importance

        Parameters
        ----------
        df_train : train DataFrame đã build_lag_features
        df_cal   : calibration DataFrame (optional, thêm vào OOF)
        """
        from xgboost import XGBRegressor
        import lightgbm as lgb
        from sklearn.linear_model import Ridge, Lasso, ElasticNet

        # Combine train + cal nếu có
        df_all = pd.concat([df_train, df_cal], ignore_index=True) if df_cal is not None else df_train
        X, y, _ = self._make_xy(df_all)
        X_tr, y_tr, _ = self._make_xy(df_train)

        # ── Step 1: OOF predictions ───────────────────────────────────────────
        oof_xgb, oof_lgb, y_oof = self._generate_oof(X_tr, y_tr)
        meta_X = np.column_stack([oof_xgb, oof_lgb])

        # ── Step 2: Meta-learner ──────────────────────────────────────────────
        meta_cls = {"ridge": Ridge(alpha=1.0), "lasso": Lasso(alpha=0.01),
                    "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5)}
        self.meta_model = meta_cls.get(self.cfg.meta_learner, Ridge(alpha=1.0))
        self.meta_model.fit(meta_X, y_oof)
        logger.info(
            f"Meta-learner ({self.cfg.meta_learner}) fitted on "
            f"OOF matrix shape {meta_X.shape}"
        )

        # ── Step 3: Retrain base models trên toàn bộ data ────────────────────
        xgb_params = {**self.best_xgb_params, "random_state": 42,
                      "n_jobs": -1, "tree_method": "hist", "verbosity": 0}
        self.xgb_model = XGBRegressor(**xgb_params)
        self.xgb_model.fit(X, y)
        logger.info("XGBoost retrained on full data.")

        lgb_params = {**self.best_lgb_params, "random_state": 42,
                      "n_jobs": -1, "verbosity": -1}
        self.lgb_model = lgb.LGBMRegressor(**lgb_params)
        self.lgb_model.fit(X, y)
        logger.info("LightGBM retrained on full data.")

        # ── Step 4: Feature importance ────────────────────────────────────────
        self._save_feature_importance()

    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """
        Dự báo bằng Stacking Ensemble.

        XGB(X) → base_xgb
        LGB(X) → base_lgb
        Ridge([base_xgb, base_lgb]) → final_prediction
        """
        if self.xgb_model is None or self.lgb_model is None or self.meta_model is None:
            raise RuntimeError("Chưa fit stacking. Gọi fit_stacking() trước.")

        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        meta_X = np.column_stack([xgb_pred, lgb_pred])
        return self.meta_model.predict(meta_X)

    def _save_feature_importance(self) -> None:
        """Lưu feature importance của XGB + LGB vào JSON."""
        if self.feature_names_ is None:
            return
        importance = {}
        if self.xgb_model is not None:
            xgb_imp = self.xgb_model.feature_importances_
            importance["xgb"] = {
                name: float(val)
                for name, val in sorted(
                    zip(self.feature_names_, xgb_imp),
                    key=lambda x: x[1], reverse=True
                )
            }
        if self.lgb_model is not None:
            lgb_imp = self.lgb_model.feature_importances_
            importance["lgb"] = {
                name: float(val)
                for name, val in sorted(
                    zip(self.feature_names_, lgb_imp),
                    key=lambda x: x[1], reverse=True
                )
            }
        path = Path(self.cfg.output_dir) / "feature_importance.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(importance, f, indent=2)
        logger.info(f"Feature importance saved → {path}")
        self._feature_importance = importance

    def get_feature_importance(self) -> dict:
        """Trả về dict feature importance đã tính."""
        return getattr(self, "_feature_importance", {})

    # ──────────────────────────────────────────────────────────────────────────
    # Technique 4: Conformal Prediction Intervals
    # ──────────────────────────────────────────────────────────────────────────

    def calibrate_conformal(self, df_cal: pd.DataFrame) -> None:
        """
        [Kỹ thuật 4] Calibrate conformal prediction trên calibration set.

        Lý thuyết Conformal Prediction
        --------------------------------
        Cho trước alpha (e.g. 0.1):
          1. Predict trên calibration set → ŷ_cal
          2. Tính residuals: r_i = |y_i - ŷ_i|
          3. Quantile = np.quantile(residuals, 1 - alpha)
          4. Với bất kỳ test point:
             PI = [ŷ - quantile, ŷ + quantile]
          → Coverage ≥ 1 - alpha (đảm bảo theo lý thuyết)

        Khác với Quantile Loss của TFT:
          - Không cần train lại model
          - Coverage guarantee không phụ thuộc vào phân phối

        Parameters
        ----------
        df_cal : calibration DataFrame đã build_lag_features
        """
        if self.xgb_model is None:
            raise RuntimeError("Chưa fit model. Gọi fit_stacking() trước.")

        X_cal, y_cal, _ = self._make_xy(df_cal)
        y_hat_cal = self.predict_stacking(X_cal)

        residuals = np.abs(y_cal - y_hat_cal)
        self._conformal_residuals = residuals

        # Tính quantile cho các mức alpha phổ biến
        alpha = self.cfg.conformal_alpha
        self._q80 = float(np.quantile(residuals, 0.80))
        self._q90 = float(np.quantile(residuals, 1 - alpha))  # alpha=0.1 → 90%
        self._q95 = float(np.quantile(residuals, 0.95))

        logger.info(
            f"Conformal calibrated on {len(residuals)} samples | "
            f"q80={self._q80:.4f} | q90={self._q90:.4f} | q95={self._q95:.4f}"
        )

    def get_intervals(
        self,
        y_pred: np.ndarray,
        coverage: float = 0.90,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính Conformal Prediction Intervals.

        Parameters
        ----------
        y_pred   : point predictions (array)
        coverage : mức coverage mong muốn (0.90 → 90%)

        Returns
        -------
        (lower, upper) : prediction interval arrays
        """
        if self._conformal_residuals is None:
            raise RuntimeError("Chưa calibrate. Gọi calibrate_conformal() trước.")

        alpha = 1 - coverage
        q = float(np.quantile(self._conformal_residuals, 1 - alpha))
        return y_pred - q, y_pred + q

    # ──────────────────────────────────────────────────────────────────────────
    # Recursive Multi-step Forecasting
    # ──────────────────────────────────────────────────────────────────────────

    def recursive_predict(
        self,
        df_test: pd.DataFrame,
        steps: Optional[int] = None,
        history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Dự báo nhiều bước theo kiểu Recursive (khác TFT direct multi-step).

        Recursive strategy
        ------------------
        1. Lấy features từ last known row (end of history)
        2. Predict 1 step → ŷ₁
        3. Cập nhật lag features: lag_1 = ŷ₁, lag_2 = lag_1_old, ...
        4. Repeat cho step 2, 3, ..., steps

        Parameters
        ----------
        df_test    : test DataFrame (đã prepare, chưa cần lag features)
        steps      : số bước dự báo (mặc định cfg.max_prediction_length)
        history_df : tail của train để khởi tạo lag (nếu df_test không có history)

        Returns
        -------
        pd.DataFrame columns: [step, y_pred, lower_90, upper_90, lower_80, upper_80]
        """
        n_steps = steps or self.cfg.max_prediction_length
        if self.feature_names_ is None:
            raise RuntimeError("Chưa fit model. Gọi fit_stacking() trước.")

        # Lấy tail của test (hoặc history) để build lag features
        context = pd.concat(
            [h for h in [history_df, df_test] if h is not None],
            ignore_index=True
        )
        context_feat = self.build_lag_features(context, drop_na=False)

        # Lấy hàng cuối cùng có đủ data
        last_row_idx = context_feat.dropna(subset=[self.cfg.target]).index[-1]
        current_features = context_feat.loc[last_row_idx, self.feature_names_].values.copy()

        # Index các cột lag để cập nhật
        lag_col_indices = {
            lag: self.feature_names_.index(f"{self.cfg.target}_lag_{lag}")
            for lag in self.cfg.lags
            if f"{self.cfg.target}_lag_{lag}" in self.feature_names_
        }
        lag1_idx = lag_col_indices.get(1)

        predictions = []
        sorted_lags = sorted(self.cfg.lags, reverse=True)

        for step in range(1, n_steps + 1):
            X_step = current_features.reshape(1, -1).astype(np.float32)
            y_hat = float(self.predict_stacking(X_step)[0])

            lo90, hi90 = self.get_intervals(np.array([y_hat]), coverage=0.90)
            lo80, hi80 = self.get_intervals(np.array([y_hat]), coverage=0.80)

            predictions.append({
                "step": step,
                "y_pred": y_hat,
                "lower_90": float(lo90[0]),
                "upper_90": float(hi90[0]),
                "lower_80": float(lo80[0]),
                "upper_80": float(hi80[0]),
            })

            # Cập nhật lag features cho bước tiếp theo
            for lag in sorted_lags:
                if lag in lag_col_indices and (lag - 1) in lag_col_indices:
                    current_features[lag_col_indices[lag]] = \
                        current_features[lag_col_indices[lag - 1]]
            if lag1_idx is not None:
                current_features[lag1_idx] = y_hat

        df_pred = pd.DataFrame(predictions)
        logger.info(f"Recursive predict complete | steps={n_steps}")
        return df_pred

    # ──────────────────────────────────────────────────────────────────────────
    # Batch prediction (toàn bộ test set)
    # ──────────────────────────────────────────────────────────────────────────

    def predict_batch(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Dự báo toàn bộ test DataFrame (1-step ahead cho từng row).

        Dùng khi muốn evaluate trên tập test có ground truth.
        Kết quả có cả conformal intervals.

        Returns
        -------
        pd.DataFrame columns: [y_pred, lower_90, upper_90, lower_80, upper_80]
        """
        X, y, _ = self._make_xy(df_test, horizon=0)
        y_hat = self.predict_stacking(X)

        lo90, hi90 = self.get_intervals(y_hat, coverage=0.90)
        lo80, hi80 = self.get_intervals(y_hat, coverage=0.80)

        return pd.DataFrame({
            "y_pred": y_hat,
            "lower_90": lo90,
            "upper_90": hi90,
            "lower_80": lo80,
            "upper_80": hi80,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        predictions: pd.DataFrame,
        pred_col: str = "y_pred",
    ) -> dict:
        """
        Tính evaluation metrics cho batch predictions.

        Metrics
        -------
        Point    : MAE, RMSE, MAPE, sMAPE
        Interval : Winkler Score (90%), Coverage (90%), Winkler (80%), Coverage (80%)

        Returns
        -------
        dict : tất cả metrics
        """
        y = np.asarray(y_true).flatten()
        y_hat = predictions[pred_col].values

        n = min(len(y), len(y_hat))
        y, y_hat = y[:n], y_hat[:n]

        metrics: dict = {
            "MAE": _mae(y, y_hat),
            "RMSE": _rmse(y, y_hat),
            "MAPE_%": _mape(y, y_hat),
            "sMAPE_%": _smape(y, y_hat),
        }

        if "lower_90" in predictions.columns and "upper_90" in predictions.columns:
            lo = predictions["lower_90"].values[:n]
            hi = predictions["upper_90"].values[:n]
            metrics["Winkler_90"] = _winkler(y, lo, hi, alpha=0.10)
            metrics["Coverage_90_%"] = _coverage(y, lo, hi) * 100

        if "lower_80" in predictions.columns and "upper_80" in predictions.columns:
            lo80 = predictions["lower_80"].values[:n]
            hi80 = predictions["upper_80"].values[:n]
            metrics["Winkler_80"] = _winkler(y, lo80, hi80, alpha=0.20)
            metrics["Coverage_80_%"] = _coverage(y, lo80, hi80) * 100

        # OOF fold scores summary
        if self.oof_scores_:
            xgb_rmse_list = [s["xgb_rmse"] for s in self.oof_scores_]
            lgb_rmse_list = [s["lgb_rmse"] for s in self.oof_scores_]
            metrics["OOF_XGB_RMSE_mean"] = float(np.mean(xgb_rmse_list))
            metrics["OOF_LGB_RMSE_mean"] = float(np.mean(lgb_rmse_list))

        logger.info("── Evaluation Metrics ──────────────────────")
        for k, v in metrics.items():
            logger.info(f"  {k:<25}: {v:.4f}")
        logger.info("────────────────────────────────────────────")
        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save_results(
        self,
        output_path: str,
        y_true: Union[np.ndarray, pd.Series, List],
        predictions: pd.DataFrame,
        extra: Optional[dict] = None,
    ) -> dict:
        """
        Tính metrics và lưu tất cả kết quả vào JSON.

        Output JSON structure
        ---------------------
        {
          "config"           : GBMConfig as dict,
          "best_hparams"     : {"xgb": {...}, "lgb": {...}},
          "oof_scores"       : [{fold, xgb_rmse, lgb_rmse}, ...],
          "conformal"        : {q80, q90, q95, n_calibration},
          "metrics"          : {MAE, RMSE, MAPE, ...},
          "feature_importance": {xgb: {...}, lgb: {...}},
          "predictions"      : {y_pred, lower_90, upper_90, ...},
          "y_true"           : [...],
        }
        """
        metrics = self.evaluate(y_true, predictions)
        y_list = np.asarray(y_true).flatten().tolist()

        output = {
            "config": self.cfg.to_dict(),
            "best_hparams": {
                "xgb": self.best_xgb_params,
                "lgb": self.best_lgb_params,
            },
            "oof_scores": self.oof_scores_,
            "conformal": {
                "q80": getattr(self, "_q80", None),
                "q90": getattr(self, "_q90", None),
                "q95": getattr(self, "_q95", None),
                "n_calibration": (
                    len(self._conformal_residuals)
                    if self._conformal_residuals is not None else 0
                ),
            },
            "metrics": metrics,
            "feature_importance": self.get_feature_importance(),
            "predictions": {
                col: predictions[col].tolist() for col in predictions.columns
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
        """Load kết quả từ JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ──────────────────────────────────────────────────────────────────────────
    # Serialisation (dùng cho Prediction API)
    # ──────────────────────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """
        Lưu toàn bộ state của GBMForecaster ra file (dùng joblib).

        Lưu: xgb_model, lgb_model, meta_model, best params,
             conformal residuals, feature_names_, config.

        Parameters
        ----------
        path : str
            Đường dẫn file đầu ra, ví dụ ``"gbm_output/gbm_forecaster.joblib"``
        """
        try:
            import joblib
        except ImportError as e:
            raise ImportError("Cần cài joblib: pip install joblib") from e

        state = {
            "config": self.cfg,
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "meta_model": self.meta_model,
            "best_xgb_params": self.best_xgb_params,
            "best_lgb_params": self.best_lgb_params,
            "_conformal_residuals": self._conformal_residuals,
            "_q80": getattr(self, "_q80", None),
            "_q90": getattr(self, "_q90", None),
            "_q95": getattr(self, "_q95", None),
            "feature_names_": self.feature_names_,
            "oof_scores_": self.oof_scores_,
        }

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, out_path, compress=3)
        logger.info(f"Model saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    @classmethod
    def load_model(cls, path: str) -> "GBMForecaster":
        """
        Load GBMForecaster đã lưu từ file joblib.

        Parameters
        ----------
        path : str
            Đường dẫn file đã lưu bởi ``save_model()``.

        Returns
        -------
        GBMForecaster instance đã sẵn sàng dự báo.
        """
        try:
            import joblib
        except ImportError as e:
            raise ImportError("Cần cài joblib: pip install joblib") from e

        state = joblib.load(path)
        instance = cls(config=state["config"])

        instance.xgb_model = state["xgb_model"]
        instance.lgb_model = state["lgb_model"]
        instance.meta_model = state["meta_model"]
        instance.best_xgb_params = state["best_xgb_params"]
        instance.best_lgb_params = state["best_lgb_params"]
        instance._conformal_residuals = state["_conformal_residuals"]
        instance._q80 = state.get("_q80")
        instance._q90 = state.get("_q90")
        instance._q95 = state.get("_q95")
        instance.feature_names_ = state["feature_names_"]
        instance.oof_scores_ = state.get("oof_scores_", [])

        logger.info(f"Model loaded ← {path}")
        return instance
