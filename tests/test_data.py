"""
test_data.py
============
Unit tests for data schemas and data loader export structures.
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from api.schemas import (
    OHLCVRecord,
    PredictRequest,
    ForecastStep,
    PredictResponse,
    HealthResponse,
    CandleResponse,
)


class TestAPISchemas:
    def test_valid_ohlcv_record(self):
        record = OHLCVRecord(
            open_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            symbol="BTC/USDT",
            open=50000.0,
            high=51000.0,
            low=49500.0,
            close=50500.0,
            volume=1200.0,
        )
        assert record.close == 50500.0
        assert record.symbol == "BTC/USDT"

    def test_predict_request_default_db(self):
        req = PredictRequest(symbol="BTC/USDT", timeframe="1d", steps=7, from_db=True)
        assert req.from_db is True
        assert req.steps == 7
        assert req.n_history == 100

    def test_predict_request_without_db_valid(self):
        candles = [
            OHLCVRecord(
                open_time=datetime(2026, 1, i, tzinfo=timezone.utc),
                open=100.0 + i,
                high=110.0 + i,
                low=90.0 + i,
                close=105.0 + i,
                volume=1000.0,
            )
            for i in range(1, 15)
        ]
        req = PredictRequest(from_db=False, candles=candles)
        assert len(req.candles) == 14

    def test_predict_request_without_db_invalid_candles(self):
        with pytest.raises(ValidationError):
            PredictRequest(from_db=False, candles=[])

    def test_forecast_step_and_predict_response(self):
        step1 = ForecastStep(
            step=1,
            y_pred=51000.0,
            lower_80=49000.0,
            upper_80=53000.0,
            lower_90=48000.0,
            upper_90=54000.0,
        )
        resp = PredictResponse(
            symbol="BTC/USDT",
            timeframe="1d",
            model="GBM",
            steps=1,
            forecast=[step1],
            n_history_rows=100,
            generated_at=datetime.now(timezone.utc),
        )
        assert resp.model == "GBM"
        assert len(resp.forecast) == 1
        assert resp.forecast[0].y_pred == 51000.0

    def test_health_response_schema(self):
        health = HealthResponse(
            status="ok",
            models_loaded={"GBM": True, "TFT": True},
            db_connected=True,
            version="1.0.0",
        )
        assert health.status == "ok"
        assert health.models_loaded["GBM"] is True
