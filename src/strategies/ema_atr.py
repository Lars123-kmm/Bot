from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def generate_signals(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Erzeugt Handelssignale basierend auf dem Multi-Filter EMA-Crossover-System.

    Bedingungen Long:
      1. EMA(fast) kreuzt EMA(slow) von unten (cross_up)
      2. ADX > adx_threshold
      3. RSI zwischen rsi_long_min und rsi_long_max
      4. Volumen > volume_ratio_min * MA(volume_ma_period)
      5. Preis < extended_max_pct% über EMA(fast)

    Bedingungen Short (spiegelbildlich, ohne extended-Filter):
      1. EMA(fast) kreuzt EMA(slow) von oben (cross_dn)
      2. ADX > adx_threshold
      3. RSI zwischen rsi_short_min und rsi_short_max
      4. Volumen > volume_ratio_min * MA(volume_ma_period)

    Signal wird um 1 Bar verschoben (kein Look-ahead-Bias).
    """
    out = df.copy()
    s = cfg["strategy"]

    required = ["cross_up", "cross_dn", "adx", "rsi", "vol_ratio", "extended"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Fehlende Indikator-Spalten: {missing}. compute_indicators() zuerst aufrufen.")

    long_signal = (
        out["cross_up"] &
        (out["adx"] > s["adx_threshold"]) &
        out["rsi"].between(s["rsi_long_min"], s["rsi_long_max"]) &
        (out["vol_ratio"] > s["volume_ratio_min"]) &
        (out["extended"] < s["extended_max_pct"])
    )

    short_signal = (
        out["cross_dn"] &
        (out["adx"] > s["adx_threshold"]) &
        out["rsi"].between(s["rsi_short_min"], s["rsi_short_max"]) &
        (out["vol_ratio"] > s["volume_ratio_min"])
    )

    out["signal"] = 0
    out.loc[long_signal, "signal"] = 1
    out.loc[short_signal, "signal"] = -1
    out["signal"] = out["signal"].shift(1).fillna(0).astype(int)

    return out
