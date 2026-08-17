"""
api/model_registry.py
=====================
Singleton registry để load và cache model GBM/TFT khi API startup.

Usage
-----
  registry = ModelRegistry.get_instance()
  forecaster = registry.gbm_model  # GBMForecaster | None
  registry.reload_gbm("/path/to/model.joblib")

Behaviour
---------
- Khi startup, tự động tìm file model trong GBM_MODEL_PATH (env var).
- Nếu file không tồn tại, gbm_model = None (API vẫn start, nhưng /predict/gbm trả 503).
- Thread-safe: sử dụng threading.Lock để bảo vệ reload.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from loguru import logger


# Đường dẫn mặc định đến file model (có thể override bằng env var)
_DEFAULT_GBM_PATH = os.getenv("GBM_MODEL_PATH", "gbm_output/gbm_forecaster.joblib")
_DEFAULT_TFT_PATH = os.getenv("TFT_MODEL_PATH", "tft_output/tft_forecaster.joblib")


class ModelRegistry:
    """
    Singleton registry giữ các model đã load trong bộ nhớ.
    Tránh load lại model mỗi lần request.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._gbm_model = None          # GBMForecaster | None
        self._tft_model = None          # TFTForecaster | None
        self._gbm_path: Optional[str] = None
        self._tft_path: Optional[str] = None
        self._reload_lock = threading.Lock()

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def gbm_model(self):
        """GBMForecaster đã load, hoặc None nếu chưa load / load thất bại."""
        return self._gbm_model

    @property
    def tft_model(self):
        """TFTForecaster đã load, hoặc None nếu chưa load / load thất bại."""
        return self._tft_model

    @property
    def gbm_path(self) -> Optional[str]:
        return self._gbm_path

    @property
    def tft_path(self) -> Optional[str]:
        return self._tft_path

    # ── Load methods ──────────────────────────────────────────────────────────

    def load_gbm(self, path: Optional[str] = None) -> bool:
        """
        Load GBMForecaster từ file joblib.

        Parameters
        ----------
        path : str, optional
            Đường dẫn file. Nếu None dùng GBM_MODEL_PATH env var.

        Returns
        -------
        bool : True nếu load thành công, False nếu file không tồn tại.
        """
        model_path = Path(path or _DEFAULT_GBM_PATH)

        if not model_path.exists():
            logger.warning(
                f"GBM model file không tồn tại: {model_path}. "
                f"Hãy train model trước bằng `scripts/run_gbm_local.py` "
                f"rồi gọi `forecaster.save_model('{model_path}')`."
            )
            return False

        with self._reload_lock:
            try:
                from models.gbm_model import GBMForecaster
                self._gbm_model = GBMForecaster.load_model(str(model_path))
                self._gbm_path = str(model_path)
                logger.success(f"✅ GBM model loaded from {model_path}")
                return True
            except Exception as exc:
                logger.error(f"❌ Không thể load GBM model: {exc}")
                self._gbm_model = None
                return False

    def load_tft(self, path: Optional[str] = None) -> bool:
        """
        Load TFTForecaster từ file joblib.
        (TFT serialisation sẽ được bổ sung sau khi train xong trên Colab)
        """
        model_path = Path(path or _DEFAULT_TFT_PATH)

        if not model_path.exists():
            logger.warning(f"TFT model file không tồn tại: {model_path}.")
            return False

        with self._reload_lock:
            try:
                import joblib
                self._tft_model = joblib.load(str(model_path))
                self._tft_path = str(model_path)
                logger.success(f"✅ TFT model loaded from {model_path}")
                return True
            except Exception as exc:
                logger.error(f"❌ Không thể load TFT model: {exc}")
                self._tft_model = None
                return False

    def reload_gbm(self, path: Optional[str] = None) -> bool:
        """Hot-reload GBM model (dùng cho /models/reload endpoint)."""
        logger.info("🔄 Reloading GBM model...")
        return self.load_gbm(path)

    def reload_tft(self, path: Optional[str] = None) -> bool:
        """Hot-reload TFT model."""
        logger.info("🔄 Reloading TFT model...")
        return self.load_tft(path)

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Trả về trạng thái load của từng model."""
        gbm = self._gbm_model
        tft = self._tft_model

        gbm_info: dict = {"loaded": gbm is not None, "path": self._gbm_path}
        if gbm is not None:
            gbm_info.update({
                "target": gbm.cfg.target,
                "max_prediction_length": gbm.cfg.max_prediction_length,
                "features_count": len(gbm.feature_names_ or []),
                "conformal_q90": getattr(gbm, "_q90", None),
            })

        tft_info: dict = {"loaded": tft is not None, "path": self._tft_path}

        return {"GBM": gbm_info, "TFT": tft_info}
