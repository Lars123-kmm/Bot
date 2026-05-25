from __future__ import annotations


class DrawdownGuard:
    """
    Circuit-Breaker: Stoppt den Bot, sobald der Gesamt-Drawdown
    das konfigurierte Maximum erreicht.

    Regel #2 aus dem Trading-Plan: Bei 10% Gesamt-Drawdown stoppt der Bot.
    """

    def __init__(self, max_dd: float = 0.10):
        self.max_dd = max_dd
        self.peak = 0.0

    def update(self, current_capital: float) -> tuple[bool, float]:
        """
        Aktualisiert den Peak und berechnet den aktuellen Drawdown.

        Returns:
            (trading_allowed, current_drawdown_as_fraction)
            trading_allowed=False bedeutet: Bot soll pausieren.
        """
        self.peak = max(self.peak, current_capital)
        if self.peak == 0.0:
            return True, 0.0
        dd = (self.peak - current_capital) / self.peak
        return dd < self.max_dd, dd

    def reset(self) -> None:
        self.peak = 0.0


class ConsecutiveLossGuard:
    """
    Stoppt den Bot nach N aufeinanderfolgenden Verlust-Trades.
    Schutz gegen Strategieversagen oder Marktregimewechsel.
    """

    def __init__(self, max_losses: int = 5):
        self.max_losses = max_losses
        self._consecutive = 0

    def record(self, pnl: float) -> None:
        if pnl < 0:
            self._consecutive += 1
        else:
            self._consecutive = 0

    def is_allowed(self) -> bool:
        return self._consecutive < self.max_losses

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive

    def reset(self) -> None:
        self._consecutive = 0
