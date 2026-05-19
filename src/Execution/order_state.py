from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ActiveTrade:
    """
    Repräsentiert eine offene Position im Live-Trading.
    Wird vom LiveTrader verwaltet und an MT5Broker weitergegeben.
    """
    ticket: int             # MT5 Order-Ticket
    direction: int          # 1=Long, -1=Short
    symbol: str
    entry_price: float
    size: float
    stop: float
    take_profit: float
    atr_at_entry: float
    open_time: datetime
    highest_price: float    # für Trailing Stop (Long)
    lowest_price: float     # für Trailing Stop (Short)
    trailing_active: bool = False

    @classmethod
    def from_entry(
        cls,
        ticket: int,
        direction: int,
        symbol: str,
        entry_price: float,
        size: float,
        stop: float,
        take_profit: float,
        atr_at_entry: float,
    ) -> "ActiveTrade":
        now = datetime.utcnow()
        return cls(
            ticket=ticket,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            size=size,
            stop=stop,
            take_profit=take_profit,
            atr_at_entry=atr_at_entry,
            open_time=now,
            highest_price=entry_price,
            lowest_price=entry_price,
        )

    def update_trailing(self, current_high: float, current_low: float,
                        current_atr: float, activate_mult: float,
                        distance_mult: float) -> float | None:
        """
        Aktualisiert den Trailing Stop anhand des aktuellen Hochs/Tiefs.
        Gibt den neuen Stop zurück, falls dieser sich verändert hat, sonst None.
        """
        if self.direction == 1:  # Long
            self.highest_price = max(self.highest_price, current_high)
            if self.highest_price - self.entry_price >= activate_mult * current_atr:
                new_stop = self.highest_price - distance_mult * current_atr
                if new_stop > self.stop:
                    self.stop = new_stop
                    self.trailing_active = True
                    return new_stop
        else:  # Short
            self.lowest_price = min(self.lowest_price, current_low)
            if self.entry_price - self.lowest_price >= activate_mult * current_atr:
                new_stop = self.lowest_price + distance_mult * current_atr
                if new_stop < self.stop:
                    self.stop = new_stop
                    self.trailing_active = True
                    return new_stop
        return None
