from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """
    Einfaches Kostenmodell für Backtests.
    Kombination aus Handelsgebühr und Slippage – wird pro Seite angewendet
    (einmal beim Entry, einmal beim Exit).
    """
    fee_rate: float = 0.001       # 0.1% Handelsgebühr (typisch Binance/Broker)
    slippage_rate: float = 0.001  # 0.1% Slippage pro Trade

    def apply(self, notional: float) -> float:
        """Gesamtkosten für eine Seite (Entry oder Exit)."""
        return abs(notional) * (self.fee_rate + self.slippage_rate)

    def round_trip(self, entry_notional: float, exit_notional: float) -> float:
        """Gesamtkosten für einen kompletten Trade (Entry + Exit)."""
        return self.apply(entry_notional) + self.apply(exit_notional)
