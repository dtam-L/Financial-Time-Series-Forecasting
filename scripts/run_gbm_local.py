"""
scripts/run_gbm_local.py
=========================
Chay full GBM pipeline tren may LOCAL (khong can Colab).
Load du lieu tu PostgreSQL hoac tu colab_data/ JSON.

Chay:
    python scripts/run_gbm_local.py
    python scripts/run_gbm_local.py --from-json          # dung JSON da export
    python scripts/run_gbm_local.py --symbol ETH/USDT
    python scripts/run_gbm_local.py --no-hpo             # bo qua Optuna
    python scripts/run_gbm_local.py --no-cv              # bo qua OOF
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.gbm_model import GBMConfig, GBMForecaster


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run GBM Forecaster pipeline locally")
    parser.add_argument("--symbol",     default="BTC/USDT")
    parser.add_argument("--timeframe",  default="1d")
    parser.add_argument("--days",       type=int, default=730)
    parser.add_argument("--target",     default="close")
    parser.add_argument("--steps",      type=int, default=7, help="Forecast horizon")
    parser.add_argument("--trials",     type=int, default=30, help="Optuna trials per model")
    parser.add_argument("--cv-folds",   type=int, default=5,  help="OOF CV folds")
    parser.add_argument("--from-json",  action="store_true",
                        help="Load du lieu tu colab_data/ thay vi DB")
    parser.add_argument("--json-dir",   default="colab_data",
                        help="Thu muc chua train.json va test.json")
    parser.add_argument("--output-dir", default="gbm_output")
    parser.add_argument("--no-hpo",     action="store_true", help="Bo qua Optuna HPO")
    parser.add_argument("--no-cv",      action="store_true", help="Bo qua OOF stacking")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_from_db(args) -> tuple:
    """Load truc tiep tu PostgreSQL."""
    from models.data_loader import OHLCVDBLoader
    loader = OHLCVDBLoader()
    df_train, df_cal, df_test = _split_three(
        *loader.load_split(args.symbol, args.timeframe, args.days)
    )
    loader.close()
    return df_train, df_cal, df_test


def load_from_json(json_dir: str) -> tuple:
    """Load tu colab_data/ JSON da export san."""
    import pandas as pd

    train_path = Path(json_dir) / "train.json"
    test_path  = Path(json_dir) / "test.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} khong tim thay.\n"
            "Hay chay truoc: python scripts/export_for_colab.py"
        )

    with open(train_path) as f:
        df_train_full = pd.DataFrame(json.load(f))
    with open(test_path) as f:
        df_test = pd.DataFrame(json.load(f))

    # Tach calibration tu cuoi train
    cal_start = int(len(df_train_full) * 0.85)
    df_train = df_train_full.iloc[:cal_start].reset_index(drop=True)
    df_cal   = df_train_full.iloc[cal_start:].reset_index(drop=True)

    print(f"Loaded from JSON:")
    print(f"  train.json : {len(df_train_full)} rows total -> train={len(df_train)}, cal={len(df_cal)}")
    print(f"  test.json  : {len(df_test)} rows")
    return df_train, df_cal, df_test


def _split_three(df_train_full, df_test):
    """Tach calibration set tu cuoi train."""
    cal_start = int(len(df_train_full) * 0.85)
    df_train = df_train_full.iloc[:cal_start].reset_index(drop=True)
    df_cal   = df_train_full.iloc[cal_start:].reset_index(drop=True)
    return df_train, df_cal, df_test


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  XGBoost + LightGBM Stacking Ensemble — Local Run")
    print("=" * 60)
    print(f"  Symbol     : {args.symbol}")
    print(f"  Target     : {args.target}")
    print(f"  Horizon    : {args.steps} steps")
    print(f"  Data source: {'JSON (' + args.json_dir + ')' if args.from_json else 'PostgreSQL'}")
    print(f"  Output     : {args.output_dir}/")
    print("=" * 60 + "\n")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("[1/6] Loading data ...")
    if args.from_json:
        df_train_raw, df_cal_raw, df_test_raw = load_from_json(args.json_dir)
    else:
        df_train_raw, df_cal_raw, df_test_raw = load_from_db(args)

    # ── 2. Config ─────────────────────────────────────────────────────────────
    cfg = GBMConfig(
        target=args.target,
        group_col="symbol",
        time_col="open_time",
        lags=[1, 2, 3, 5, 7, 14, 21],
        rolling_windows=[7, 14, 30],
        ewma_spans=[7, 14],
        max_prediction_length=args.steps,
        n_optuna_trials=args.trials,
        n_cv_folds=args.cv_folds,
        meta_learner="ridge",
        conformal_alpha=0.10,
        output_dir=args.output_dir,
    )
    forecaster = GBMForecaster(cfg)

    # ── 3. Lag features ───────────────────────────────────────────────────────
    print("\n[2/6] Building lag features ...")
    df_train = forecaster.build_lag_features(df_train_raw, drop_na=True)
    df_cal   = forecaster.build_lag_features(df_cal_raw,   drop_na=True)
    df_test  = forecaster.build_lag_features(df_test_raw,  drop_na=True)

    print(f"  train: {df_train.shape} | cal: {df_cal.shape} | test: {df_test.shape}")
    lag_cols = [c for c in df_train.columns if f"{cfg.target}_" in c]
    print(f"  Lag/rolling features created: {len(lag_cols)}")

    # ── 4. HPO ────────────────────────────────────────────────────────────────
    if not args.no_hpo:
        print(f"\n[3/6] Optuna HPO ({args.trials} trials each, MedianPruner) ...")
        forecaster.tune_all(df_train, n_trials=args.trials)
    else:
        print("\n[3/6] Skipping HPO (--no-hpo) — using default params")

    # ── 5. Stacking ───────────────────────────────────────────────────────────
    if not args.no_cv:
        print(f"\n[4/6] OOF Stacking ({args.cv_folds} folds) ...")
        forecaster.fit_stacking(df_train, df_cal)
    else:
        print("\n[4/6] Skipping OOF stacking (--no-cv) — fitting base models only ...")
        from xgboost import XGBRegressor
        import lightgbm as lgb
        from sklearn.linear_model import Ridge
        import numpy as np

        X, y, _ = forecaster._make_xy(df_train)
        forecaster.xgb_model = XGBRegressor(
            **{**forecaster.best_xgb_params, "random_state": 42, "verbosity": 0}
        )
        forecaster.lgb_model = lgb.LGBMRegressor(
            **{**forecaster.best_lgb_params, "random_state": 42, "verbosity": -1}
        )
        forecaster.xgb_model.fit(X, y)
        forecaster.lgb_model.fit(X, y)

        # Fit trivial meta-model
        X_cal_arr, y_cal, _ = forecaster._make_xy(df_cal)
        xgb_cal = forecaster.xgb_model.predict(X_cal_arr)
        lgb_cal = forecaster.lgb_model.predict(X_cal_arr)
        import numpy as np
        meta_X = np.column_stack([xgb_cal, lgb_cal])
        forecaster.meta_model = Ridge(alpha=1.0).fit(meta_X, y_cal)

    # ── 6. Conformal calibration ──────────────────────────────────────────────
    print("\n[5/6] Calibrating conformal intervals ...")
    forecaster.calibrate_conformal(df_cal)
    print(f"  q90 (90% coverage) = +/-{forecaster._q90:.4f}")

    # ── 7. Prediction + evaluation ────────────────────────────────────────────
    print("\n[6/6] Predicting & evaluating ...")

    # Batch prediction (evaluate trên test set)
    predictions_batch = forecaster.predict_batch(df_test)
    X_te, y_true, _ = forecaster._make_xy(df_test, horizon=0)

    metrics = forecaster.evaluate(y_true, predictions_batch)

    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  MAE         : {metrics['MAE']:.4f}")
    print(f"  RMSE        : {metrics['RMSE']:.4f}")
    print(f"  MAPE%       : {metrics['MAPE_%']:.2f}%")
    print(f"  sMAPE%      : {metrics['sMAPE_%']:.2f}%")
    if "Winkler_90" in metrics:
        print(f"  Winkler 90% : {metrics['Winkler_90']:.4f}")
        print(f"  Coverage 90%: {metrics['Coverage_90_%']:.1f}%  (target >= 90%)")

    # Recursive multi-step forecast
    print(f"\n  Recursive {args.steps}-step forecast:")
    recursive_pred = forecaster.recursive_predict(
        df_test=df_test,
        steps=args.steps,
        history_df=df_train.tail(30),
    )
    print(recursive_pred[["step", "y_pred", "lower_90", "upper_90"]].to_string(index=False))

    # ── 8. Save ───────────────────────────────────────────────────────────────
    result_path = Path(args.output_dir) / "gbm_results.json"
    forecaster.save_results(
        output_path=str(result_path),
        y_true=y_true,
        predictions=predictions_batch,
        extra={
            "symbol": args.symbol,
            "recursive_forecast": recursive_pred.to_dict(orient="records"),
        },
    )
    print(f"\nResults saved -> {result_path}")
    print("Done!")


if __name__ == "__main__":
    main()
