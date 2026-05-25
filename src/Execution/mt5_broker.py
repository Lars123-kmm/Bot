from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5

from src.Execution.order_state import ActiveTrade

logger = logging.getLogger(__name__)


class MT5Broker:
    """
    Wrapper um die MT5-API für alle Order-Operationen.
    Alle Methoden werfen RuntimeError bei MT5-Fehlern.
    """

    def __init__(self, magic: int = 240101, deviation: int = 20):
        self.magic = magic
        self.deviation = deviation

    # ------------------------------------------------------------------
    # Konto-Informationen
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict[str, float]:
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
        }

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """Gibt (bid, ask) zurück."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")
        return tick.bid, tick.ask

    # ------------------------------------------------------------------
    # Order-Operationen
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        direction: int,
        size: float,
        sl: float,
        tp: float,
        comment: str = "EMA-Bot",
    ) -> int:
        """
        Platziert eine Market Order.
        Gibt das MT5-Ticket zurück.
        """
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Kein Preis für {symbol}: {mt5.last_error()}")

        price = tick.ask if direction == 1 else tick.bid

        request: Dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(size),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            raise RuntimeError(f"order_send fehlgeschlagen (retcode={code}): {mt5.last_error()}")

        logger.info("Order platziert: ticket=%d symbol=%s dir=%d size=%.6f sl=%.5f tp=%.5f",
                    result.order, symbol, direction, size, sl, tp)
        return result.order

    def modify_stop(self, ticket: int, new_sl: float) -> bool:
        """
        Modifiziert den Stop Loss einer offenen Position (für Trailing Stop).
        """
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning("Position mit Ticket %d nicht gefunden.", ticket)
            return False

        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(new_sl),
            "tp": pos.tp,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            logger.error("modify_stop fehlgeschlagen (ticket=%d, retcode=%s)", ticket, code)
            return False

        logger.info("Trailing Stop aktualisiert: ticket=%d neuer SL=%.5f", ticket, new_sl)
        return True

    def close_position(self, ticket: int, symbol: str, size: float, direction: int) -> bool:
        """Schließt eine offene Position per Market Order."""
        close_type = mt5.ORDER_TYPE_SELL if direction == 1 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Kein Preis für {symbol}")

        price = tick.bid if direction == 1 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(size),
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "EMA-Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            logger.error("close_position fehlgeschlagen (ticket=%d, retcode=%s)", ticket, code)
            return False

        logger.info("Position geschlossen: ticket=%d", ticket)
        return True

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Gibt alle offenen Positionen des Magic Numbers zurück."""
        if symbol:
            raw = mt5.positions_get(symbol=symbol)
        else:
            raw = mt5.positions_get()

        if raw is None:
            return []

        result = []
        for p in raw:
            if p.magic == self.magic:
                result.append({
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "direction": 1 if p.type == mt5.ORDER_TYPE_BUY else -1,
                    "size": p.volume,
                    "entry_price": p.price_open,
                    "stop": p.sl,
                    "take_profit": p.tp,
                    "profit": p.profit,
                    "open_time": p.time,
                })
        return result
