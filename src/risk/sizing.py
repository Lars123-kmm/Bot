from __future__ import annotations

from scipy.stats import norm


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


def probability_scaled_position(
    capital: float,
    entry_price: float,
    atr: float,
    risk_pct: float,
    meta_prob: float,
    min_prob: float = 0.55,
    max_scale: float = 2.0,
    stop_mult: float = 2.0,
    tp_mult: float = 3.0,
    direction: int = 1,
    sizing_scale: float = 1.0,
) -> dict:
    """
    Wahrscheinlichkeits-skaliertes Position Sizing (Meta-Labeling).

    Die Positionsgröße wächst linear mit der Modell-Konfidenz:
      prob = min_prob  → scale = 0    (kein Trade, sollte schon gefiltert sein)
      prob = midpoint  → scale = 1.0  (normale Größe)
      prob = 1.0       → scale = max_scale

    Zusätzlicher sizing_scale-Faktor z.B. 0.5 bei VOLATILE-Regime.

    Args:
        meta_prob:    Modell-Wahrscheinlichkeit [0, 1]
        min_prob:     Untere Schwelle (unter der kein Trade)
        max_scale:    Maximaler Skalierungsfaktor (Standard: 2.0)
        sizing_scale: Externer Skalierungsfaktor (z.B. 0.5 bei VOLATILE)
    """
    prob_range = 1.0 - min_prob
    if prob_range <= 0:
        scale = 1.0
    else:
        scale = min(max_scale, (meta_prob - min_prob) / prob_range * max_scale)
        scale = max(0.0, scale)

    effective_risk = risk_pct * scale * sizing_scale
    if effective_risk <= 0:
        effective_risk = risk_pct * 0.1  # Mindest-Fallback

    return calculate_position(capital, entry_price, atr, effective_risk,
                               stop_mult=stop_mult, tp_mult=tp_mult, direction=direction)


def lopez_bet_size(
    prob: float,
    freq: float = 1.0,
) -> float:
    """
    Position size from calibrated probability (López de Prado Ch. 10).
    z = (prob - 0.5) / sqrt(prob*(1-prob)) * sqrt(freq)
    size = 2*Φ(z) - 1  ∈ [0, 1]
    Returns 0 for prob <= 0.5 (no edge).
    """
    if prob <= 0.5:
        return 0.0
    denom = max(prob * (1.0 - prob), 1e-10) ** 0.5
    z = (prob - 0.5) / denom * (freq ** 0.5)
    return float(max(0.0, min(1.0, 2.0 * norm.cdf(z) - 1.0)))


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
