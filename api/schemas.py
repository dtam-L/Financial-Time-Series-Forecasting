

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────

class OHLCVRecord(BaseModel):
    """Một nến OHLCV kèm các technical indicators (optional)."""

    open_time: datetime = Field(..., description="Thời điểm mở nến (ISO 8601 UTC)")
    symbol: Optional[str] = Field(None, description="Symbol, ví dụ 'BTC/USDT'. Nếu None dùng symbol trong request.")
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Technical indicators (optional — nếu thiếu sẽ được tính trong pipeline)
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_pct: Optional[float] = None
    bb_width: Optional[float] = None
    atr_14: Optional[float] = None
    volume_ratio: Optional[float] = None
    rolling_vol_30: Optional[float] = None
    price_zscore: Optional[float] = None
    drawdown_pct: Optional[float] = None
    ma_7: Optional[float] = None
    ma_25: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    return_pct: Optional[float] = None
    log_return: Optional[float] = None
    hour_sin: Optional[float] = None
    hour_cos: Optional[float] = None
    dow_sin: Optional[float] = None
    dow_cos: Optional[float] = None
    month_sin: Optional[float] = None
    month_cos: Optional[float] = None
    doy_sin: Optional[float] = None
    doy_cos: Optional[float] = None


class PredictRequest(BaseModel):
    """Request body cho /predict/gbm và /predict/tft."""

    symbol: str = Field(
        default="BTC/USDT",
        description="Ký hiệu tài sản, ví dụ 'BTC/USDT'",
        examples=["BTC/USDT", "ETH/USDT"],
    )
    timeframe: str = Field(
        default="1d",
        description="Khung thời gian: '1d', '1h', ...",
        examples=["1d", "1h"],
    )
    steps: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Số bước dự báo (1–30)",
    )
    from_db: bool = Field(
        default=True,
        description="Nếu True, API tự query PostgreSQL lấy dữ liệu. Nếu False, dùng trường `candles`.",
    )
    n_history: int = Field(
        default=100,
        ge=30,
        le=1000,
        description="Số candles lấy từ DB khi from_db=True (cần đủ để tính lag features).",
    )
    candles: Optional[List[OHLCVRecord]] = Field(
        default=None,
        description="Danh sách candles OHLCV khi from_db=False. Cần ít nhất 30 rows.",
    )

    @model_validator(mode="after")
    def check_candles_when_not_from_db(self) -> "PredictRequest":
        if not self.from_db and (not self.candles or len(self.candles) < 10):
            raise ValueError(
                "Khi from_db=False, cần cung cấp ít nhất 10 candles trong trường 'candles'."
            )
        return self


# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────

class ForecastStep(BaseModel):
    """Một bước dự báo với conformal prediction intervals."""

    step: int = Field(..., description="Bước dự báo (1 = t+1, 2 = t+2, ...)")
    y_pred: float = Field(..., description="Giá trị dự báo điểm (point forecast)")
    lower_80: float = Field(..., description="Giới hạn dưới 80% conformal interval")
    upper_80: float = Field(..., description="Giới hạn trên 80% conformal interval")
    lower_90: float = Field(..., description="Giới hạn dưới 90% conformal interval")
    upper_90: float = Field(..., description="Giới hạn trên 90% conformal interval")


class PredictResponse(BaseModel):
    """Kết quả dự báo đầy đủ."""

    symbol: str
    timeframe: str
    model: Literal["GBM", "TFT"]
    steps: int
    forecast: List[ForecastStep]
    n_history_rows: int = Field(..., description="Số candles thực sự dùng để predict")
    generated_at: datetime = Field(..., description="Thời điểm tạo dự báo (UTC)")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok", "degraded", "error"]
    models_loaded: dict[str, bool]
    db_connected: bool
    version: str = "1.0.0"


class ModelStatusResponse(BaseModel):
    """Thông tin chi tiết về model đã load."""

    name: str
    loaded: bool
    path: Optional[str] = None
    target: Optional[str] = None
    max_prediction_length: Optional[int] = None
    features_count: Optional[int] = None
    conformal_q90: Optional[float] = None


class CandleResponse(BaseModel):
    """Một nến OHLCV từ DB."""

    symbol: str
    timeframe: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
