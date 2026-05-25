from __future__ import annotations

"""
Telegram notifications for the trading bot.

Requires environment variables:
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — your chat/channel id

If not configured (or notifications.enabled is false) every call is a silent
no-op, so the runner works fine without Telegram.
"""

import logging
import os
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org"


class Notifier:
    """Sends short status messages to Telegram. No-op when not configured."""

    def __init__(self, cfg: Dict[str, Any]):
        notif_cfg = cfg.get("notifications", {})
        self.enabled = bool(notif_cfg.get("enabled", False))
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

        if self.enabled and (not self._token or not self._chat_id):
            logger.warning(
                "notifications.enabled=true but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
                "are not set — notifications disabled."
            )
            self.enabled = False

        if self.enabled:
            logger.info("Telegram notifications active.")

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            requests.post(
                f"{_API}/bot{self._token}/sendMessage",
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)

    # ------------------------------------------------------------------
    # High-level events
    # ------------------------------------------------------------------

    def trade_opened(
        self, symbol: str, direction: int, size: float, price: float, prob: float | None
    ) -> None:
        arrow = "🟢 LONG" if direction == 1 else "🔴 SHORT"
        prob_str = f"  (prob {prob:.2f})" if prob is not None else ""
        self.send(
            f"{arrow}  <b>{symbol}</b>{prob_str}\n"
            f"size {size:.6f} @ {price:,.2f}"
        )

    def trade_closed(self, symbol: str, pnl: float, reason: str) -> None:
        icon = "✅" if pnl >= 0 else "❌"
        self.send(
            f"{icon}  <b>{symbol}</b> closed ({reason})\n"
            f"PnL: {pnl:+,.2f} USD"
        )

    def guard_tripped(self, name: str, detail: str) -> None:
        self.send(f"⚠️  <b>{name}</b> ausgelöst\n{detail}")

    def daily_summary(self, summary: Dict[str, Any]) -> None:
        self.send(
            "📊  <b>Tagesübersicht</b>\n"
            f"Equity:   {summary.get('equity', 0):,.2f} USD\n"
            f"PnL ges.: {summary.get('total_pnl', 0):+,.2f} USD\n"
            f"Trades:   {summary.get('n_trades', 0)}\n"
            f"Win-Rate: {summary.get('win_rate', 0):.1%}"
        )

    def error(self, message: str) -> None:
        self.send(f"🛑  <b>Fehler</b>\n{message}")

    def info(self, message: str) -> None:
        self.send(f"ℹ️  {message}")
