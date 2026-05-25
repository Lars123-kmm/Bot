from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _PaperPosition:
    ticket: int
    symbol: str
    direction: int       # 1=Long, -1=Short
    size: float
    entry_price: float
    sl: float
    tp: float
    fee_paid: float
    open_time: datetime
    # set when closed
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    pnl: Optional[float] = None
    close_reason: Optional[str] = None


class PaperBroker:
    """
    In-memory paper trading broker — same external interface as MT5Broker.

    Simulates market orders, SL/TP checks on bar H/L, fee deduction.
    Does NOT simulate partial fills, slippage beyond spread, or funding.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        fee_rate: float = 0.0004,      # 0.04% per side (Binance Futures taker)
        spread_pct: float = 0.0001,    # 0.01% half-spread
    ):
        self._balance = initial_balance
        self._fee_rate = fee_rate
        self._spread_pct = spread_pct

        self._next_ticket = 1
        self._open: Dict[int, _PaperPosition] = {}
        self._closed: List[_PaperPosition] = []

        # latest mid-prices per symbol (updated by update_bar / set_price)
        self._prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Price management
    # ------------------------------------------------------------------

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def update_bar(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> List[int]:
        """
        Called for each completed bar.  Checks every open position for that
        symbol against the bar's high/low.  Returns list of tickets that
        were closed (SL or TP hit).
        """
        self._prices[symbol] = close
        closed_tickets: List[int] = []

        for ticket, pos in list(self._open.items()):
            if pos.symbol != symbol:
                continue

            hit_sl = hit_tp = False
            if pos.direction == 1:          # Long
                if low <= pos.sl:
                    hit_sl = True
                    fill = pos.sl
                elif high >= pos.tp:
                    hit_tp = True
                    fill = pos.tp
            else:                           # Short
                if high >= pos.sl:
                    hit_sl = True
                    fill = pos.sl
                elif low <= pos.tp:
                    hit_tp = True
                    fill = pos.tp

            if hit_sl or hit_tp:
                reason = "sl" if hit_sl else "tp"
                self._close_position_internal(pos, fill, reason)
                closed_tickets.append(ticket)

        return closed_tickets

    # ------------------------------------------------------------------
    # MT5Broker-compatible interface
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict[str, float]:
        unrealized = sum(
            self._unrealized_pnl(p) for p in self._open.values()
        )
        equity = self._balance + unrealized
        return {
            "balance": round(self._balance, 2),
            "equity": round(equity, 2),
            "margin": 0.0,
            "free_margin": round(equity, 2),
            "profit": round(unrealized, 2),
        }

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        mid = self._prices.get(symbol, 0.0)
        half_spread = mid * self._spread_pct
        return round(mid - half_spread, 8), round(mid + half_spread, 8)  # bid, ask

    def place_market_order(
        self,
        symbol: str,
        direction: int,
        size: float,
        sl: float,
        tp: float,
        comment: str = "Paper",
    ) -> int:
        bid, ask = self.get_current_price(symbol)
        fill_price = ask if direction == 1 else bid
        notional = fill_price * size
        fee = notional * self._fee_rate

        if self._balance < fee:
            raise RuntimeError("Insufficient balance for fee")

        self._balance -= fee
        ticket = self._next_ticket
        self._next_ticket += 1

        pos = _PaperPosition(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=fill_price,
            sl=sl,
            tp=tp,
            fee_paid=fee,
            open_time=datetime.now(timezone.utc),
        )
        self._open[ticket] = pos

        logger.info(
            "[PAPER] BUY/SELL ticket=%d %s dir=%d size=%.6f @ %.5f sl=%.5f tp=%.5f",
            ticket, symbol, direction, size, fill_price, sl, tp,
        )
        return ticket

    def modify_stop(self, ticket: int, new_sl: float) -> bool:
        pos = self._open.get(ticket)
        if pos is None:
            return False
        pos.sl = new_sl
        logger.info("[PAPER] modify_stop ticket=%d new_sl=%.5f", ticket, new_sl)
        return True

    def close_position(self, ticket: int, symbol: str, size: float, direction: int) -> bool:
        pos = self._open.get(ticket)
        if pos is None:
            logger.warning("[PAPER] close_position: ticket %d not found", ticket)
            return False
        bid, ask = self.get_current_price(symbol)
        fill = bid if direction == 1 else ask
        self._close_position_internal(pos, fill, "manual")
        return True

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        for p in self._open.values():
            if symbol and p.symbol != symbol:
                continue
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "direction": p.direction,
                "size": p.size,
                "entry_price": p.entry_price,
                "stop": p.sl,
                "take_profit": p.tp,
                "profit": round(self._unrealized_pnl(p), 2),
                "open_time": int(p.open_time.timestamp()),
            })
        return result

    # ------------------------------------------------------------------
    # Additional helpers used by LiveRunner
    # ------------------------------------------------------------------

    def get_closed_trade(self, ticket: int) -> Optional[Dict[str, Any]]:
        for p in self._closed:
            if p.ticket == ticket:
                return {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "close_price": p.close_price,
                    "pnl": p.pnl,
                    "close_reason": p.close_reason,
                    "open_time": p.open_time,
                    "close_time": p.close_time,
                }
        return None

    def get_all_closed_trades(self) -> List[Dict[str, Any]]:
        return [self.get_closed_trade(p.ticket) for p in self._closed]  # type: ignore[misc]

    def summary(self) -> Dict[str, Any]:
        closed = self._closed
        pnls = [p.pnl for p in closed if p.pnl is not None]
        n_trades = len(pnls)
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        info = self.get_account_info()
        return {
            "balance": info["balance"],
            "equity": info["equity"],
            "total_pnl": round(total_pnl, 2),
            "n_trades": n_trades,
            "n_open": len(self._open),
            "win_rate": round(wins / n_trades, 4) if n_trades else 0.0,
            "avg_pnl": round(total_pnl / n_trades, 2) if n_trades else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unrealized_pnl(self, pos: _PaperPosition) -> float:
        mid = self._prices.get(pos.symbol, pos.entry_price)
        return (mid - pos.entry_price) * pos.direction * pos.size

    def _close_position_internal(
        self, pos: _PaperPosition, fill_price: float, reason: str
    ) -> None:
        notional = fill_price * pos.size
        fee = notional * self._fee_rate
        gross_pnl = (fill_price - pos.entry_price) * pos.direction * pos.size
        net_pnl = gross_pnl - fee

        self._balance += net_pnl
        pos.close_price = fill_price
        pos.close_time = datetime.now(timezone.utc)
        pos.pnl = round(net_pnl, 4)
        pos.close_reason = reason

        self._open.pop(pos.ticket, None)
        self._closed.append(pos)

        logger.info(
            "[PAPER] CLOSE ticket=%d %s reason=%s pnl=%.4f balance=%.2f",
            pos.ticket, pos.symbol, reason, net_pnl, self._balance,
        )
