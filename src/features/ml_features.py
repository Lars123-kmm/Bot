from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.features.fracdiff import frac_diff_ffd, find_min_d

# Kanonische Feature-Liste – muss bei Training und Inferenz identisch sein
ML_FEATURE_COLS: List[str] = [
    # Rendite-Features
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_16",
    # Volatilität
    "realized_vol_20", "vol_z",
    # Präzisere Volatilitäts-Schätzer (Garman-Klass, Parkinson)
    "garman_klass_vol", "parkinson_vol",
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
    # Stationäre Preisdarstellung (fraktionale Differenzierung)
    "close_fracdiff",
    # Zeit
    "hour_sin", "hour_cos", "day_of_week",
    # Lag-Features
    "rsi_lag1", "rsi_lag2", "rsi_lag3",
    "adx_lag1", "adx_lag2", "adx_lag3",
    "vol_ratio_lag1", "vol_ratio_lag2", "vol_ratio_lag3",
    # EMA-Abstand normiert
    "ema_fast_norm", "ema_slow_norm",
    # Crypto-spezifische Signale (0 wenn Binance nicht verfügbar)
    "funding_rate_z", "oi_change_z", "order_book_imbalance",
    # HMM-Regime-Wahrscheinlichkeiten (0.5 wenn HMM nicht gelaufen)
    "hmm_prob_trend", "hmm_prob_volatile",
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

    # --- Garman-Klass und Parkinson Volatilität ---
    out["garman_klass_vol"] = _garman_klass_vol(out)
    out["parkinson_vol"] = _parkinson_vol(out)

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

    # --- Fraktionale Differenzierung (stationäre Preisdarstellung) ---
    out["close_fracdiff"] = _fracdiff_feature(out["close"], cfg)

    # --- Zeit-Features (Marktzeiten, periodisch kodiert) ---
    out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
    out["day_of_week"] = out.index.dayofweek.astype(float)

    # --- Lag-Features (kein Look-ahead dank shift) ---
    for lag in [1, 2, 3]:
        out[f"rsi_lag{lag}"] = out["rsi"].shift(lag)
        out[f"adx_lag{lag}"] = out["adx"].shift(lag)
        out[f"vol_ratio_lag{lag}"] = out["vol_ratio"].shift(lag)

    # --- Crypto-spezifische Binance-Features (0 wenn nicht vorhanden) ---
    for col in ("funding_rate", "oi_change"):
        if col not in out.columns:
            out[col] = 0.0
    if "order_book_imbalance" not in out.columns:
        out["order_book_imbalance"] = 0.0

    fr_std = out["funding_rate"].rolling(50).std().replace(0, np.finfo(float).eps)
    fr_mean = out["funding_rate"].rolling(50).mean()
    out["funding_rate_z"] = ((out["funding_rate"] - fr_mean) / fr_std).fillna(0.0)

    oi_std = out["oi_change"].rolling(50).std().replace(0, np.finfo(float).eps)
    oi_mean = out["oi_change"].rolling(50).mean()
    out["oi_change_z"] = ((out["oi_change"] - oi_mean) / oi_std).fillna(0.0)

    # --- HMM-Regime-Wahrscheinlichkeiten (0.5 Fallback wenn HMM nicht gelaufen) ---
    if "hmm_prob_trend" not in out.columns:
        out["hmm_prob_trend"] = 0.5
    if "hmm_prob_volatile" not in out.columns:
        out["hmm_prob_volatile"] = 0.5

    out.dropna(subset=ML_FEATURE_COLS, inplace=True)
    return out


def _garman_klass_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Garman-Klass annualized volatility (uses OHLC, more efficient than close-only)."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    gk = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window).mean()
    return np.sqrt(gk * 252).fillna(0.0)


def _parkinson_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Parkinson annualized volatility (uses high-low range only)."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    park = (log_hl / (4 * np.log(2))).rolling(window).mean()
    return np.sqrt(park * 252).fillna(0.0)


def _fracdiff_feature(series: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    """
    Fractionally differenced close price. d cached in cfg after first ADF search.
    Uses d=0.3 as fallback if series too short for ADF test.
    """
    d = cfg.get("fracdiff_d")
    if d is None:
        if len(series) >= 100:
            d = find_min_d(series)
        else:
            d = 0.3
        cfg["fracdiff_d"] = d
    return frac_diff_ffd(series, float(d)).fillna(0.0)


def _require_cols(df: pd.DataFrame, cols: list) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_ml_features() erwartet diese Spalten, die fehlen: {missing}. "
            "compute_indicators() zuerst aufrufen."
        )
