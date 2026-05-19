from __future__ import annotations


def calculate_position(
    capital: float,
    entry_price: float,
    atr: float,
    risk_pct: float,
    stop_mult: float = 2.0,
    tp_mult: float = 3.0,
    direction: int = 1,
) -> dict:
    """
    ATR-basiertes Position Sizing.
    Verlust beim Erreichen des Stop Loss = exakt risk_pct * capital.

    Args:
        capital:     Aktuelles Kapital in Quote-Währung
        entry_price: Einstiegspreis
        atr:         ATR-Wert der aktuellen Bar
        risk_pct:    Risiko pro Trade (z.B. 0.01 = 1%)
        stop_mult:   ATR-Multiplikator für Stop Loss (Standard: 2.0)
        tp_mult:     ATR-Multiplikator für Take Profit (Standard: 3.0)
        direction:   1=Long, -1=Short

    Returns:
        dict mit size, size_quote, risk_quote, stop, take_profit
    """
    if atr <= 0:
        raise ValueError(f"ATR muss > 0 sein, erhalten: {atr}")
    if capital <= 0:
        raise ValueError(f"Kapital muss > 0 sein, erhalten: {capital}")

    stop_distance = stop_mult * atr
    risk_amount = capital * risk_pct
    size = risk_amount / stop_distance

    stop = entry_price - direction * stop_distance
    take_profit = entry_price + direction * tp_mult * atr

    return {
        "size": round(size, 6),
        "size_quote": round(size * entry_price, 2),
        "risk_quote": round(risk_amount, 2),
        "stop": round(stop, 8),
        "take_profit": round(take_profit, 8),
    }


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Half-Kelly Criterion für optionale Sizing-Überprüfung.
    Gibt die empfohlene Kapitalbindung als Anteil zurück (z.B. 0.092 = 9.2%).

    Beispiel: win_rate=0.55, avg_win=1.5, avg_loss=1.0 → 0.092
    """
    if avg_loss <= 0:
        raise ValueError("avg_loss muss > 0 sein.")
    b = avg_win / avg_loss
    kelly = (b * win_rate - (1.0 - win_rate)) / b
    return max(0.0, kelly * 0.5)
