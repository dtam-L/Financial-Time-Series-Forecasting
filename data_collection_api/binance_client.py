

from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
import requests
from loguru import logger

from . import config

# Base URL for Binance public REST API
_BINANCE_REST = "https://api.binance.com"


class BinanceClient:
    _TF_MS: dict[str, int] = {
        "1m":   60_000,
        "3m":   180_000,
        "5m":   300_000,
        "15m":  900_000,
        "30m":  1_800_000,
        "1h":   3_600_000,
        "2h":   7_200_000,
        "4h":   14_400_000,
        "6h":   21_600_000,
        "8h":   28_800_000,
        "12h":  43_200_000,
        "1d":   86_400_000,
        "3d":   259_200_000,
        "1w":   604_800_000,
        "1M":   2_592_000_000,
    }

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ) -> None:
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._api_key = api_key or config.BINANCE_API_KEY

        # HTTP session for direct public REST calls (OHLCV)
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self._api_key})

        # ccxt exchange for ticker / order-book (spot only, no portfolio margin)
        self._exchange = ccxt.binance(
            {
                "apiKey": self._api_key,
                "secret": api_secret or config.BINANCE_SECRET_KEY,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "fetchMarkets": ["spot"],   # skip margin/portfolio endpoints
                    "portfolioMargin": False,
                },
            }
        )
        logger.info("BinanceClient initialised (public REST + ccxt spot)")

    def _call_with_retry(self, fn, *args, **kwargs):
        """Execute a ccxt call; retry on transient errors with back-off."""
        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except ccxt.RateLimitExceeded as exc:
                wait = self._retry_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit exceeded (attempt {attempt}/{self._max_retries}). "
                    f"Sleeping {wait:.1f}s…"
                )
                time.sleep(wait)
                last_err = exc
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                wait = self._retry_delay * (2 ** attempt)
                logger.warning(
                    f"Network error '{exc}' (attempt {attempt}/{self._max_retries}). "
                    f"Sleeping {wait:.1f}s…"
                )
                time.sleep(wait)
                last_err = exc
            except ccxt.ExchangeError as exc:
                # Non-transient exchange error — fail immediately
                logger.error(f"Exchange error: {exc}")
                raise
        raise RuntimeError(
            f"All {self._max_retries} retries exhausted. Last error: {last_err}"
        )

    @staticmethod
    def _to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _request_with_retry(self, url: str, params: dict) -> requests.Response:
        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=10)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", self._retry_delay * (2 ** attempt)))
                    logger.warning(f"Rate limited (429). Sleeping {wait:.1f}s… (attempt {attempt})")
                    time.sleep(wait)
                    last_err = Exception("HTTP 429 Too Many Requests")
                    continue
                if resp.status_code >= 500:
                    wait = self._retry_delay * (2 ** attempt)
                    logger.warning(f"Server error {resp.status_code}. Sleeping {wait:.1f}s… (attempt {attempt})")
                    time.sleep(wait)
                    last_err = Exception(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp
            except requests.ConnectionError as exc:
                wait = self._retry_delay * (2 ** attempt)
                logger.warning(f"Connection error (attempt {attempt}). Sleeping {wait:.1f}s…")
                time.sleep(wait)
                last_err = exc
        raise RuntimeError(f"All {self._max_retries} retries exhausted. Last: {last_err}")


    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 1000,
    ):
        since = since or config.START_DATE
        until = until or datetime.now(timezone.utc)

        since_ms = self._to_ms(since)
        until_ms = self._to_ms(until)
        tf_ms = self._TF_MS.get(timeframe)
        if tf_ms is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'. "
                             f"Valid: {list(self._TF_MS)}")
        binance_symbol = symbol.replace("/", "")
        url = f"{_BINANCE_REST}/api/v3/klines"

        all_rows: list[list] = []
        current_ms = since_ms

        logger.info(
            f"Fetching {symbol} {timeframe} "
            f"from {since.date()} to {until.date()}"
        )

        while current_ms < until_ms:
            params = {
                "symbol": binance_symbol,
                "interval": timeframe,
                "startTime": current_ms,
                "endTime": until_ms - 1,
                "limit": min(limit, 1000),
            }
            resp = self._request_with_retry(url, params)
            batch = resp.json()

            if not batch:
                break

            all_rows.extend(batch)

            if len(batch) < limit:
                break

            current_ms = int(batch[-1][0]) + tf_ms
            logger.debug(
                f"  → fetched {len(batch)} rows, "
                f"cursor: {datetime.fromtimestamp(current_ms/1000, tz=timezone.utc).date()}"
            )

        if not all_rows:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return pd.DataFrame(
                columns=["open_time", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(all_rows).iloc[:, :6]
        df.columns = ["open_time", "open", "high", "low", "close", "volume"]  # type: ignore[assignment]
        df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
        df = df.reset_index(drop=True)

        logger.success(
            f"Fetched {len(df)} candles for {symbol} {timeframe} "
            f"({df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()})"
        )
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        ticker = self._call_with_retry(self._exchange.fetch_ticker, symbol)
        logger.debug(f"Ticker {symbol}: last={ticker.get('last')}")
        return ticker

    def fetch_order_book(self, symbol: str, depth: int = 20) -> dict:
        book = self._call_with_retry(
            self._exchange.fetch_order_book, symbol, limit=depth
        )
        logger.debug(
            f"Order book {symbol}: "
            f"best_bid={book['bids'][0][0] if book['bids'] else 'N/A'}, "
            f"best_ask={book['asks'][0][0] if book['asks'] else 'N/A'}"
        )
        return book

    def get_exchange_info(self) -> dict:
        markets = self._call_with_retry(self._exchange.load_markets)
        logger.info(f"Loaded {len(markets)} markets from Binance")
        return markets
