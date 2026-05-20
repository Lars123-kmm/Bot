from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Kanonische Feature-Liste – muss bei Training und Inferenz identisch sein
ML_FEATURE_COLS: List[str] = [
    # Rendite-Features
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_16",
    # Volatilität
    "realized_vol_20", "vol_z",
    # Trend
    "ema_spread", "ema_slope_fast", "ema_slope_slow",
    # Momentum
    "rsi_change", "rsi_extreme",
    # Volumen
    "vol_z_score", "vol_trend",
    # ADX-Dynamik
    "adx_change", "adx_momentum",
    # Bestehende Indikatoren (direkt)
    "rsi", "adx", "vol_ratio", "extended", "atr",
    # Zeit
    "hour_sin", "hour_cos", "day_of_week",
    # Lag-Features
    "rsi_lag1", "rsi_lag2", "rsi_lag3",
    "adx_lag1", "adx_lag2", "adx_lag3",
    "vol_ratio_lag1", "vol_ratio_lag2", "vol_ratio_lag3",
    # EMA-Abstand normiert
    "ema_fast_norm", "ema_slow_norm",
]


def build_ml_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Baut den ML-Feature-Datensatz auf Basis des Indikator-DataFrames.
    Erwartet Ausgabe von compute_indicators() als Input.
    Gibt DataFrame mit ML_FEATURE_COLS zurück (NaN-Zeilen am Anfang entfernt).
    """
    _require_cols(df, ["open", "high", "low", "close", "volume",
                       "ema_fast", "ema_slow", "rsi", "adx", "atr",
                       "vol_ma", "vol_ratio", "extended"])

    out = df.copy()

    # --- Log-Returns (kein Look-ahead: shift(n) zeigt n Bars zurück) ---
    for n in [1, 2, 4, 8, 16]:
        out[f"ret_{n}"] = np.log(out["close"] / out["close"].shift(n))

    # --- Volatilitäts-Features ---
    out["realized_vol_20"] = out["ret_1"].rolling(20).std() * np.sqrt(24 * 365)
    atr_ma = out["atr"].rolling(50).mean()
    atr_std = out["atr"].rolling(50).std().replace(0, np.finfo(float).eps)
    out["vol_z"] = (out["atr"] - atr_ma) / atr_std

    # --- Trend-Features ---
    out["ema_spread"] = (out["ema_fast"] - out["ema_slow"]) / out["ema_slow"] * 100
    out["ema_slope_fast"] = out["ema_fast"].pct_change(3) * 100
    out["ema_slope_slow"] = out["ema_slow"].pct_change(8) * 100
    out["ema_fast_norm"] = out["close"] / out["ema_fast"] - 1.0
    out["ema_slow_norm"] = out["close"] / out["ema_slow"] - 1.0

    # --- Momentum-Features ---
    out["rsi_change"] = out["rsi"].diff(3)
    out["rsi_extreme"] = ((out["rsi"] > 70) | (out["rsi"] < 30)).astype(int)

    # --- Volumen-Features ---
    vol_std = out["volume"].rolling(50).std().replace(0, np.finfo(float).eps)
    out["vol_z_score"] = (out["volume"] - out["vol_ma"]) / vol_std
    out["vol_trend"] = out["vol_ma"].pct_change(5) * 100

    # --- ADX-Dynamik ---
    out["adx_change"] = out["adx"].diff(5)
    out["adx_momentum"] = out["adx"] - out["adx"].rolling(20).mean()

    # --- Zeit-Features (Marktzeiten, periodisch kodiert) ---
    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["day_of_week"] = out.index.dayofweek.astype(float)

    # --- Lag-Features (kein Look-ahead dank shift) ---
    for lag in [1, 2, 3]:
        out[f"rsi_lag{lag}"] = out["rsi"].shift(lag)
        out[f"adx_lag{lag}"] = out["adx"].shift(lag)
        out[f"vol_ratio_lag{lag}"] = out["vol_ratio"].shift(lag)

    out.dropna(subset=ML_FEATURE_COLS, inplace=True)
    return out


def _require_cols(df: pd.DataFrame, cols: list) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_ml_features() erwartet diese Spalten, die fehlen: {missing}. "
            "compute_indicators() zuerst aufrufen."
        )
