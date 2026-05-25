from __future__ import annotations

"""
Binance Futures broker — real signed orders.

Requires environment variables:
  BINANCE_API_KEY     — your Futures API key
  BINANCE_API_SECRET  — your Futures API secret

Config must contain:
  live:
    use_testnet: true          # false → mainnet (REAL MONEY!)
    real_orders_confirmed: true

Testnet URL  : https://testnet.binancefuture.com
Mainnet URL  : https://fapi.binance.com
"""

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

_TESTNET_BASE = "https://testnet.binancefuture.com"
_MAINNET_BASE = "https://fapi.binance.com"
_RECV_WINDOW = 5000
_MAX_RETRIES = 3


class BinanceBroker:
    """
    Wrapper around the Binance Futures REST API for order management.
    Uses HMAC SHA256 signing for all private endpoints.
    Default: testnet — set use_testnet=False + real_orders_confirmed=True in config for mainnet.
    """

    def __init__(self, cfg: Dict[str, Any]):
        live_cfg = cfg.get("live", {})
        use_testnet = live_cfg.get("use_testnet", True)

        if not use_testnet and not live_cfg.get("real_orders_confirmed", False):
            raise RuntimeError(
                "Mainnet trading requires 'live.real_orders_confirmed: true' in config."
            )

        self._base = _TESTNET_BASE if use_testnet else _MAINNET_BASE
        self._api_key = os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = os.environ.get("BINANCE_API_SECRET", "")

        if not self._api_key or not self._api_secret:
            raise RuntimeError(
                "BINANCE_API_KEY and BINANCE_API_SECRET must be set as environment variables."
            )

        self._testnet = use_testnet
        logger.info("BinanceBroker init: %s", "TESTNET" if use_testnet else "MAINNET ⚠️")

    # ------------------------------------------------------------------
    # Public (unsigned) helpers
    # ------------------------------------------------------------------

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """Returns (bid, ask)."""
        data = self._get_public(f"{self._base}/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        return float(data["bidPrice"]), float(data["askPrice"])

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict[str, float]:
        data = self._signed_get("/fapi/v2/account", {})
        return {
            "balance": float(data.get("totalWalletBalance", 0)),
            "equity": float(data.get("totalMarginBalance", 0)),
            "margin": float(data.get("totalInitialMargin", 0)),
            "free_margin": float(data.get("availableBalance", 0)),
            "profit": float(data.get("totalUnrealizedProfit", 0)),
        }

    # ------------------------------------------------------------------
    # Order operations
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        direction: int,
        size: float,
        sl: float,
        tp: float,
        comment: str = "Bot",
    ) -> int:
        """Places a MARKET order and returns the Binance orderId."""
        side = "BUY" if direction == 1 else "SELL"
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{size:.6f}",
        }
        result = self._signed_post("/fapi/v1/order", params)
        order_id = int(result["orderId"])
        logger.info("[BINANCE] %s %s size=%.6f orderId=%d", side, symbol, size, order_id)

        # Place SL and TP as separate stop-market orders
        self._place_sl_tp(symbol, direction, size, sl, tp)
        return order_id

    def modify_stop(self, ticket: int, new_sl: float) -> bool:
        """Binance doesn't support modifying orders — cancel + replace."""
        logger.warning(
            "[BINANCE] modify_stop not supported inline — use cancel+replace (ticket=%d)", ticket
        )
        return False

    def close_position(self, ticket: int, symbol: str, size: float, direction: int) -> bool:
        """Closes an open position with a reduce-only market order."""
        close_side = "SELL" if direction == 1 else "BUY"
        params = {
            "symbol": symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": f"{size:.6f}",
            "reduceOnly": "true",
        }
        try:
            result = self._signed_post("/fapi/v1/order", params)
            logger.info("[BINANCE] CLOSE %s orderId=%d", symbol, result["orderId"])
            return True
        except Exception as exc:
            logger.error("[BINANCE] close_position failed: %s", exc)
            return False

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        data = self._signed_get("/fapi/v2/positionRisk", {})
        result = []
        for p in data:
            amt = float(p.get("positionAmt", 0))
            if amt == 0:
                continue
            sym = p["symbol"]
            if symbol and sym != symbol:
                continue
            result.append({
                "ticket": int(p.get("updateTime", 0)),
                "symbol": sym,
                "direction": 1 if amt > 0 else -1,
                "size": abs(amt),
                "entry_price": float(p.get("entryPrice", 0)),
                "stop": float(p.get("stopPrice", 0)),
                "take_profit": 0.0,
                "profit": float(p.get("unRealizedProfit", 0)),
                "open_time": int(p.get("updateTime", 0)) // 1000,
            })
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_sl_tp(
        self, symbol: str, direction: int, size: float, sl: float, tp: float
    ) -> None:
        close_side = "SELL" if direction == 1 else "BUY"
        try:
            self._signed_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": close_side,
                "type": "STOP_MARKET",
                "stopPrice": f"{sl:.8f}",
                "closePosition": "true",
            })
            self._signed_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": close_side,
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": f"{tp:.8f}",
                "closePosition": "true",
            })
        except Exception as exc:
            logger.warning("[BINANCE] SL/TP placement failed: %s", exc)

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = _RECV_WINDOW
        query = urlencode(params)
        sig = hmac.new(
            self._api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self._api_key}

    def _get_public(self, url: str, params: dict) -> dict:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _signed_get(self, path: str, params: dict) -> Any:
        for attempt in range(_MAX_RETRIES):
            try:
                p = self._sign(dict(params))
                resp = requests.get(
                    f"{self._base}{path}", params=p, headers=self._headers(), timeout=10
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

    def _signed_post(self, path: str, params: dict) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                p = self._sign(dict(params))
                resp = requests.post(
                    f"{self._base}{path}", params=p, headers=self._headers(), timeout=10
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}
