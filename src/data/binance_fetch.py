from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BASE = "https://fapi.binance.com"
_MAX_RETRIES = 3
_KLINE_LIMIT = 1500  # Binance max per request

_INTERVAL_MAP = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1w",
    # also accept native strings
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w",
}


def _get(url: str, params: dict) -> dict | list:
    """HTTP GET with exponential backoff on 429/5xx."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code in (429, 500, 502, 503):
                wait = 2 ** attempt
                logger.warning("Binance API %d, retry in %ds", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            logger.warning("Binance request failed (%s), retry %d", exc, attempt + 1)
            time.sleep(2 ** attempt)
    return {}


class BinanceFetcher:
    """Fetches public Binance Futures data — no API key required."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "H1",
        bars: int = 500,
    ) -> pd.DataFrame:
        """
        Fetches recent closed OHLCV bars from Binance Futures public klines endpoint.
        Automatically paginates when bars > 1500.
        Drops the still-forming candle (last open bar).

        Returns DataFrame with columns [open, high, low, close, volume],
        DatetimeIndex in UTC.
        """
        interval = _INTERVAL_MAP.get(timeframe, timeframe)
        all_rows: List[list] = []
        end_ms: Optional[int] = None

        remaining = bars
        while remaining > 0:
            limit = min(remaining, _KLINE_LIMIT)
            params: Dict = {"symbol": symbol, "interval": interval, "limit": limit}
            if end_ms is not None:
                params["endTime"] = end_ms

            try:
                raw = _get(f"{_BASE}/fapi/v1/klines", params)
            except Exception as exc:
                logger.error("fetch_ohlcv failed (%s)", exc)
                break

            if not raw:
                break

            all_rows = list(raw) + all_rows
            remaining -= len(raw)
            # paginate backwards: next batch ends just before earliest bar
            end_ms = int(raw[0][0]) - 1
            if len(raw) < limit:
                break

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "n_trades",
            "taker_base_vol", "taker_quote_vol", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"].astype(int), unit="ms", utc=True)

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)

        df = df.set_index("open_time").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]

        # drop the forming (still-open) candle: its close_time is in the future
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # rebuild close_time as Series aligned to df
        close_times = pd.to_datetime(
            [int(r[6]) for r in all_rows], unit="ms"
        )
        ct_series = pd.Series(close_times.values, index=df.index)
        df = df[ct_series < pd.Timestamp.now()]

        return df.iloc[-bars:]  # cap to requested count

    def get_funding_rates(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.Series:
        """GET /fapi/v1/fundingRate → pd.Series(funding_rate, index=datetime, freq≈8h)"""
        params: Dict = {"symbol": symbol, "limit": limit}
        if since:
            params["startTime"] = int(since.timestamp() * 1000)
        if until:
            params["endTime"] = int(until.timestamp() * 1000)

        try:
            data = _get(f"{_BASE}/fapi/v1/fundingRate", params)
        except Exception as exc:
            logger.error("Funding rate fetch failed: %s", exc)
            return pd.Series(dtype=float)

        if not data:
            return pd.Series(dtype=float)

        records = {
            pd.Timestamp(int(r["fundingTime"]), unit="ms", tz="UTC"): float(r["fundingRate"])
            for r in data
        }
        return pd.Series(records, name="funding_rate").sort_index()

    def get_open_interest_hist(
        self,
        symbol: str,
        period: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.Series:
        """GET /futures/data/openInterestHist → pd.Series(oi_value, index=datetime)"""
        params: Dict = {"symbol": symbol, "period": period, "limit": limit}
        if since:
            params["startTime"] = int(since.timestamp() * 1000)

        try:
            data = _get(f"{_BASE}/futures/data/openInterestHist", params)
        except Exception as exc:
            logger.error("OI history fetch failed: %s", exc)
            return pd.Series(dtype=float)

        if not data:
            return pd.Series(dtype=float)

        records = {
            pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC"): float(r["sumOpenInterest"])
            for r in data
        }
        return pd.Series(records, name="oi_value").sort_index()

    def get_order_book_snapshot(self, symbol: str, limit: int = 20) -> dict:
        """GET /fapi/v1/depth → {'bids': [(price,qty)...], 'asks': [(price,qty)...]}"""
        try:
            data = _get(f"{_BASE}/fapi/v1/depth", {"symbol": symbol, "limit": limit})
            return {
                "bids": [(float(p), float(q)) for p, q in data.get("bids", [])],
                "asks": [(float(p), float(q)) for p, q in data.get("asks", [])],
            }
        except Exception as exc:
            logger.error("Order book fetch failed: %s", exc)
            return {"bids": [], "asks": []}

    def compute_order_book_imbalance(self, book: dict, levels: int = 5) -> float:
        """OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty) over top-k levels."""
        bids = book.get("bids", [])[:levels]
        asks = book.get("asks", [])[:levels]
        if not bids or not asks:
            return 0.0
        bid_qty = sum(q for _, q in bids)
        ask_qty = sum(q for _, q in asks)
        total = bid_qty + ask_qty
        if total <= 0:
            return 0.0
        return (bid_qty - ask_qty) / total

    def enrich_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Adds three columns to df (forward-filled to bar index):
          funding_rate, oi_change, order_book_imbalance
        Only meaningful for Binance perpetuals. On error: columns = 0.
        """
        out = df.copy()
        out["funding_rate"] = 0.0
        out["oi_change"] = 0.0
        out["order_book_imbalance"] = 0.0

        try:
            since = df.index[0].to_pydatetime() if hasattr(df.index[0], "to_pydatetime") else None

            # Funding rates (8h frequency)
            funding = self.get_funding_rates(symbol, since=since)
            if not funding.empty:
                funding_reindexed = funding.reindex(df.index, method="ffill").fillna(0.0)
                out["funding_rate"] = funding_reindexed.values

            # Open Interest → compute pct_change as feature
            oi = self.get_open_interest_hist(symbol, period="1h", since=since)
            if not oi.empty:
                oi_reindexed = oi.reindex(df.index, method="ffill").fillna(method="bfill")
                oi_change = oi_reindexed.pct_change().fillna(0.0)
                out["oi_change"] = oi_change.values

            # Current order book snapshot (live only — static for backtest)
            book = self.get_order_book_snapshot(symbol)
            obi = self.compute_order_book_imbalance(book)
            out["order_book_imbalance"] = obi

        except Exception as exc:
            logger.warning("Binance enrichment failed, using zeros: %s", exc)

        return out
