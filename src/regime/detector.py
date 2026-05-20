from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

# Regime-Konstanten
RANGE = 0
TREND = 1
VOLATILE = 2

REGIME_NAMES = {RANGE: "Range", TREND: "Trend", VOLATILE: "Volatil"}


@dataclass
class RegimeStats:
    trend_pct: float
    range_pct: float
    volatile_pct: float
    dominant: str

    @classmethod
    def from_series(cls, regimes: pd.Series) -> "RegimeStats":
        n = max(len(regimes), 1)
        trend_pct = (regimes == TREND).sum() / n * 100
        range_pct = (regimes == RANGE).sum() / n * 100
        vol_pct = (regimes == VOLATILE).sum() / n * 100
        counts = {TREND: trend_pct, RANGE: range_pct, VOLATILE: vol_pct}
        dominant = REGIME_NAMES[max(counts, key=counts.get)]
        return cls(trend_pct=round(trend_pct, 1),
                   range_pct=round(range_pct, 1),
                   volatile_pct=round(vol_pct, 1),
                   dominant=dominant)


def detect_regime(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """
    Klassifiziert jeden Bar als RANGE (0), TREND (1) oder VOLATILE (2).

    Methodik (Kombination dreier unabhängiger Signale):
    1. Realisierte Volatilität  → VOLATILE wenn > vol_volatile_threshold
    2. Hurst-Exponent           → TREND wenn H > hurst_trend_threshold
    3. ADX-Durchschnitt         → Bestätigung Trend / Range

    Gibt pd.Series mit Werten 0/1/2 zurück, Index = df.index.
    """
    r = cfg.get("regime", {})
    window = int(r.get("window", 100))
    hurst_trend = float(r.get("hurst_trend_threshold", 0.55))
    vol_volatile = float(r.get("vol_volatile_threshold", 0.80))
    adx_min = float(r.get("adx_min_for_trend", 22))

    closes = df["close"].values
    adx_vals = df["adx"].values if "adx" in df.columns else np.zeros(len(df))

    regimes = np.zeros(len(df), dtype=int)

    for i in range(window, len(df)):
        win_closes = closes[i - window: i]
        win_adx = adx_vals[i - window: i]

        hurst = _hurst_exponent(win_closes)
        log_ret = np.log(win_closes[1:] / win_closes[:-1])
        realized_vol = log_ret.std() * np.sqrt(24 * 365)
        avg_adx = float(np.mean(win_adx))

        if realized_vol > vol_volatile:
            regimes[i] = VOLATILE
        elif hurst > hurst_trend and avg_adx > adx_min:
            regimes[i] = TREND
        else:
            regimes[i] = RANGE

    return pd.Series(regimes, index=df.index, dtype=int, name="regime")


def _hurst_exponent(ts: np.ndarray) -> float:
    """
    Hurst-Exponent via R/S-Analyse (Rescaled Range).
    H > 0.55 → Trending (persistente Zeitreihe)
    H < 0.45 → Mean-reverting
    0.45–0.55 → Random Walk (Range)
    """
    n = len(ts)
    if n < 20:
        return 0.5

    max_lag = min(n // 2, 20)
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        std = np.std(diffs)
        if std > 0:
            tau.append(std)
        else:
            tau.append(np.finfo(float).eps)

    if len(tau) < 2:
        return 0.5

    try:
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(poly[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.5
