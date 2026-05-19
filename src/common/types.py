from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


class PositionState(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass
class TradeSignal:
    direction: int        # 1=Long, -1=Short
    bar_index: int
    entry_price: float
    stop: float
    take_profit: float
    size: float
    atr: float


@dataclass
class ClosedTrade:
    direction: int        # 1=Long, -1=Short
    entry_price: float
    exit_price: float
    size: float
    pnl: float            # nach Fees/Slippage
    reason: str           # 'SL' | 'TP' | 'Trailing' | 'Signal' | 'Time'
    bars_held: int
    open_time: datetime = field(default_factory=datetime.utcnow)
    close_time: datetime = field(default_factory=datetime.utcnow)
