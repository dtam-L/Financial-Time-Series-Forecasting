"""
api/main.py
===========
FastAPI application — Financial Time Series Prediction API.

Endpoints
---------
GET  /health              → Health check + model status
GET  /models/status       → Chi tiết model đã load
POST /models/reload       → Hot-reload model từ file (không cần restart)
GET  /data/latest         → Lấy N candles gần nhất từ PostgreSQL
POST /predict/gbm         → Dự báo GBM (XGB+LGB Stacking + Conformal Intervals)
POST /predict/tft         → Dự báo TFT (Temporal Fusion Transformer)

Chạy local
----------
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI
----------
  http://localhost:8000/docs
  http://localhost:8000/redoc
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.model_registry import ModelRegistry
from api.schemas import (
    CandleResponse,
    HealthResponse,
    ModelStatusResponse,
    PredictRequest,
    PredictResponse,
)

# ─────────────────────────────────────────────────────────────
# Lifespan: load models on startup
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chạy khi startup và shutdown."""
    logger.info("=" * 60)
    logger.info("🚀 Financial Time Series Prediction API starting...")

    registry = ModelRegistry.get_instance()

    # Load GBM model
    gbm_ok = registry.load_gbm()
    if not gbm_ok:
        logger.warning("⚠️  GBM model not loaded. /predict/gbm sẽ trả 503.")

    # Load TFT model (optional)
    tft_ok = registry.load_tft()
    if not tft_ok:
        logger.warning("⚠️  TFT model not loaded. /predict/tft sẽ trả 503.")

    logger.info("✅ API ready.")
    logger.info("=" * 60)

    yield

    logger.info("👋 API shutting down.")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial Time Series Prediction API",
    description=(
        "REST API dự báo giá tài sản tài chính (BTC, ETH, ...) "
        "sử dụng GBM Stacking Ensemble và TFT.\n\n"
        "- **GBM**: XGBoost + LightGBM Stacking + Conformal Prediction Intervals\n"
        "- **TFT**: Temporal Fusion Transformer (Quantile Forecasting)\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — cho phép mọi origin (dev mode; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _check_db() -> bool:
    """Kiểm tra kết nối DB có hoạt động không."""
    try:
        from sqlalchemy import create_engine, text
        from data_collection_api import config
        engine = create_engine(config.DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Routes — Health & Status
# ─────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Status"],
)
async def health_check() -> HealthResponse:
    """
    Kiểm tra trạng thái API, models đã load, và kết nối DB.

    - `status`: **ok** (tất cả sẵn sàng), **degraded** (thiếu model), **error** (nghiêm trọng)
    - `models_loaded`: dict model_name → bool
    - `db_connected`: kết nối PostgreSQL có hoạt động không
    """
    registry = ModelRegistry.get_instance()
    models_loaded = {
        "GBM": registry.gbm_model is not None,
        "TFT": registry.tft_model is not None,
    }
    db_ok = _check_db()

    if all(models_loaded.values()) and db_ok:
        api_status = "ok"
    elif db_ok or any(models_loaded.values()):
        api_status = "degraded"
    else:
        api_status = "error"

    return HealthResponse(
        status=api_status,
        models_loaded=models_loaded,
        db_connected=db_ok,
    )


@app.get(
    "/models/status",
    summary="Chi tiết model đã load",
    tags=["Status"],
)
async def models_status() -> dict:
    """Trả về thông tin chi tiết về từng model: path, target, số features, conformal q90."""
    registry = ModelRegistry.get_instance()
    return registry.status()


@app.post(
    "/models/reload",
    summary="Hot-reload model từ file",
    tags=["Status"],
)
async def reload_models(
    model: str = Query(default="gbm", enum=["gbm", "tft", "all"], description="Model cần reload"),
    path: Optional[str] = Query(default=None, description="Đường dẫn file (để trống dùng env var)"),
) -> dict:
    """
    Hot-reload model mà không cần restart API.
    Hữu ích sau khi train xong model mới.
    """
    registry = ModelRegistry.get_instance()
    results = {}

    if model in ("gbm", "all"):
        ok = registry.reload_gbm(path)
        results["GBM"] = "loaded" if ok else "failed (file not found)"

    if model in ("tft", "all"):
        ok = registry.reload_tft(path)
        results["TFT"] = "loaded" if ok else "failed (file not found)"

    return {"reloaded": results, "timestamp": datetime.now(timezone.utc).isoformat()}


# ─────────────────────────────────────────────────────────────
# Routes — Data
# ─────────────────────────────────────────────────────────────

@app.get(
    "/data/latest",
    response_model=List[CandleResponse],
    summary="Lấy N candles gần nhất từ DB",
    tags=["Data"],
)
async def get_latest_candles(
    symbol: str = Query(default="BTC/USDT", description="Symbol tài sản"),
    timeframe: str = Query(default="1d", description="Khung thời gian: 1d, 1h, ..."),
    n: int = Query(default=20, ge=1, le=500, description="Số candles cần lấy"),
) -> List[CandleResponse]:
    """
    Query PostgreSQL và trả về N candles OHLCV gần nhất.
    Hữu ích để kiểm tra dữ liệu trong DB.
    """
    try:
        from api.predict_service import fetch_latest_candles
        return fetch_latest_candles(symbol, timeframe, n)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )


# ─────────────────────────────────────────────────────────────
# Routes — Predict
# ─────────────────────────────────────────────────────────────

@app.post(
    "/predict/gbm",
    response_model=PredictResponse,
    summary="Dự báo GBM (XGB + LGB Stacking)",
    tags=["Predict"],
)
async def predict_gbm(request: PredictRequest) -> PredictResponse:
    """
    Dự báo giá đóng cửa nhiều bước sử dụng **GBM Stacking Ensemble**
    (XGBoost + LightGBM → Ridge meta-learner) với **Conformal Prediction Intervals**.

    ### Cách dùng

    **Từ DB (mặc định):**
    ```json
    {
      "symbol": "BTC/USDT",
      "timeframe": "1d",
      "steps": 7,
      "from_db": true,
      "n_history": 100
    }
    ```

    **Từ candles thủ công:**
    ```json
    {
      "symbol": "BTC/USDT",
      "timeframe": "1d",
      "steps": 3,
      "from_db": false,
      "candles": [
        {"open_time": "2024-01-01T00:00:00Z", "open": 42000, "high": 43000,
         "low": 41500, "close": 42800, "volume": 1234.5},
        ...
      ]
    }
    ```

    ### Response
    - `forecast[i].y_pred`: giá trị dự báo tại bước i+1
    - `forecast[i].lower_90` / `upper_90`: conformal interval 90% coverage
    - `forecast[i].lower_80` / `upper_80`: conformal interval 80% coverage
    """
    registry = ModelRegistry.get_instance()
    if registry.gbm_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GBM model chưa được load. "
                "Hãy train model và gọi POST /models/reload sau khi có file."
            ),
        )

    try:
        from api.predict_service import predict_gbm as _predict_gbm
        return await _run_in_threadpool(_predict_gbm, request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception(f"predict_gbm error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi nội bộ: {exc}",
        )


@app.post(
    "/predict/tft",
    response_model=PredictResponse,
    summary="Dự báo TFT (Temporal Fusion Transformer)",
    tags=["Predict"],
)
async def predict_tft(request: PredictRequest) -> PredictResponse:
    """
    Dự báo sử dụng **Temporal Fusion Transformer** với Quantile Forecasting.

    > **Lưu ý**: Cần train TFT trên Colab trước, sau đó export và reload model.
    """
    registry = ModelRegistry.get_instance()
    if registry.tft_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TFT model chưa được load. Hãy train trên Colab và gọi POST /models/reload.",
        )

    try:
        from api.predict_service import predict_tft as _predict_tft
        return await _run_in_threadpool(_predict_tft, request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception(f"predict_tft error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi nội bộ: {exc}",
        )


# ─────────────────────────────────────────────────────────────
# Async thread pool helper (CPU-bound predict không block event loop)
# ─────────────────────────────────────────────────────────────

import asyncio
from functools import partial


async def _run_in_threadpool(func, *args, **kwargs):
    """Chạy hàm đồng bộ trong thread pool để không block async event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


# ─────────────────────────────────────────────────────────────
# Entry point (development)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info",
    )
