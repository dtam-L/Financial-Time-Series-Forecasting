"""
config.py
=========
Centralised configuration loader.
All values come from the .env file (or environment variables).
"""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"[Config] Required environment variable '{key}' is missing.\n"
            f"Copy .env.example → .env and fill in the value."
        )
    return val

BINANCE_API_KEY: str = _require("BINANCE_API_KEY")
BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")

DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "financial_ts")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

DB_URL: str = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

SYMBOLS: list[str] = [
    s.strip()
    for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
    if s.strip()
]

TIMEFRAMES: list[str] = [
    tf.strip()
    for tf in os.getenv("TIMEFRAMES", "1d").split(",")
    if tf.strip()
]

START_DATE: datetime = datetime.fromisoformat(
    os.getenv("START_DATE", "2022-01-01")
)

BATCH_LIMIT: int = int(os.getenv("BATCH_LIMIT", "1000"))
LOG_DIR: Path = _PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
