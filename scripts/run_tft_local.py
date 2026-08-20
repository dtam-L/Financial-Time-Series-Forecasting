"""
scripts/run_tft_local.py
=========================
Chạy full TFT pipeline trên máy LOCAL (hoặc Colab qua CLI).
Load dữ liệu từ PostgreSQL hoặc từ colab_data/ JSON.

Chạy:
    python scripts/run_tft_local.py --from-json --no-hpo --no-cv --epochs 10
    python scripts/run_tft_local.py --from-json --trials 10 --epochs 30
    python scripts/run_tft_local.py --symbol BTC/USDT --epochs 30
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

from models.tft_model import TFTConfig, TFTForecaster


def parse_args():
    parser = argparse.ArgumentParser(description="Run TFT Forecaster pipeline locally")
    parser.add_argument("--symbol",        default="BTC/USDT")
    parser.add_argument("--timeframe",     default="1d")
    parser.add_argument("--days",          type=int, default=730)
    parser.add_argument("--target",        default="close")
    parser.add_argument("--encoder-len",   type=int, default=60, help="Lookback window")
    parser.add_argument("--pred-len",      type=int, default=7,  help="Forecast horizon")
    parser.add_argument("--epochs",        type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--trials",        type=int, default=15, help="Optuna trials")
    parser.add_argument("--cv-folds",      type=int, default=5,  help="CV folds")
    parser.add_argument("--from-json",     action="store_true",
                        help="Load dữ liệu từ colab_data/ thay vì DB")
    parser.add_argument("--json-dir",      default="colab_data",
                        help="Thư mục chứa train.json và test.json")
    parser.add_argument("--output-dir",    default="tft_output")
    parser.add_argument("--no-hpo",        action="store_true", help="Bỏ qua Optuna HPO")
    parser.add_argument("--no-cv",         action="store_true", help="Bỏ qua Walk-Forward CV")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Temporal Fusion Transformer (TFT) — Local Run")
    print("=" * 60)
    print(f"  Symbol     : {args.symbol}")
    print(f"  Target     : {args.target}")
    print(f"  Encoder len: {args.encoder_len} steps")
    print(f"  Pred len   : {args.pred_len} steps")
    print(f"  Max Epochs : {args.epochs}")
    print(f"  Data source: {'JSON (' + args.json_dir + ')' if args.from_json else 'PostgreSQL'}")
    print(f"  Output     : {args.output_dir}/")
    print("=" * 60 + "\n")

    cfg = TFTConfig(
        target=args.target,
        group_col="symbol",
        time_col="open_time",
        max_encoder_length=args.encoder_len,
        max_prediction_length=args.pred_len,
        known_real_features=[
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "doy_sin", "doy_cos",
        ],
        unknown_real_features=[
            "return_pct", "log_return",
            "rsi_14", "macd_hist", "macd", "macd_signal",
            "bb_pct", "bb_width", "atr_14", "volume_ratio",
            "rolling_vol_30", "price_zscore", "drawdown_pct",
            "ma_7", "ma_25", "ema_12", "ema_26",
        ],
        quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        early_stopping_patience=8,
        val_ratio=0.15,
        n_optuna_trials=args.trials,
        n_cv_folds=args.cv_folds,
        output_dir=args.output_dir,
    )

    forecaster = TFTForecaster(cfg)

    # 1. Load data
    print("[1/5] Loading data ...")
    if args.from_json:
        train_path = os.path.join(args.json_dir, "train.json")
        test_path  = os.path.join(args.json_dir, "test.json")
        df_train, df_val, df_test = forecaster.load_and_split(
            train_path=train_path,
            test_path=test_path,
            val_ratio=cfg.val_ratio,
            symbol=args.symbol,
        )
    else:
        from models.data_loader import OHLCVDBLoader
        loader = OHLCVDBLoader()
        df_tr_raw, df_test = loader.load_split(args.symbol, args.timeframe, args.days)
        loader.close()
        
        # Split train / val
        cutoff = int(len(df_tr_raw) * (1 - cfg.val_ratio))
        df_train = df_tr_raw.iloc[:cutoff].copy()
        df_val   = df_tr_raw.iloc[cutoff:].copy()

        df_train = forecaster.prepare_dataframe(df_train, cfg.group_col, cfg.time_col, args.symbol)
        df_val   = forecaster.prepare_dataframe(df_val,   cfg.group_col, cfg.time_col, args.symbol)
        df_test  = forecaster.prepare_dataframe(df_test,  cfg.group_col, cfg.time_col, args.symbol)

    print(f"  train: {df_train.shape} | val: {df_val.shape} | test: {df_test.shape}")

    # 2. Optuna HPO
    if not args.no_hpo:
        print(f"\n[2/5] Optuna HPO ({args.trials} trials) ...")
        forecaster.tune_hyperparameters(df_train=df_train, df_val=df_val, n_trials=args.trials)
    else:
        print("\n[2/5] Skipping HPO (--no-hpo)")

    # 3. Walk-Forward CV
    if not args.no_cv:
        print(f"\n[3/5] Walk-Forward CV ({args.cv_folds} folds) ...")
        forecaster.walk_forward_cv(df=df_train, n_splits=args.cv_folds)
    else:
        print("\n[3/5] Skipping CV (--no-cv)")

    # 4. Final Training
    print("\n[4/5] Training final TFT model ...")
    train_res = forecaster.train(df_train=df_train, df_val=df_val)
    print(f"  Best val_loss: {train_res['val_loss']:.4f}")

    # 5. Prediction with Checkpoint Ensemble
    print("\n[5/5] Predicting with Checkpoint Ensemble ...")
    import pandas as pd
    encoder_tail = df_train.tail(cfg.max_encoder_length).copy()
    df_test_with_history = pd.concat([encoder_tail, df_test], ignore_index=True)

    predictions = forecaster.ensemble_predict(df_test=df_test_with_history, n_best=3)
    y_true = df_test[cfg.target].values

    metrics = forecaster.evaluate(y_true=y_true, predictions=predictions)

    print("\n" + "=" * 50)
    print("  TFT EVALUATION RESULTS")
    print("=" * 50)
    print(f"  R²         : {metrics.get('R2', 0.0):.4f}")
    print(f"  MAE        : {metrics['MAE']:.4f}")
    print(f"  RMSE       : {metrics['RMSE']:.4f}")
    print(f"  MAPE%      : {metrics['MAPE_%']:.2f}%")
    print(f"  sMAPE%     : {metrics['sMAPE_%']:.2f}%")
    if "Coverage_80_%" in metrics:
        print(f"  Coverage80%: {metrics['Coverage_80_%']:.1f}%")

    result_path = os.path.join(args.output_dir, "tft_results.json")
    forecaster.save_results(
        output_path=result_path,
        y_true=y_true,
        predictions=predictions,
        train_result=train_res,
        extra={"symbol": args.symbol},
    )
    print(f"\nResults saved -> {result_path}")
    print("Done!")


if __name__ == "__main__":
    main()
