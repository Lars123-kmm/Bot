from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Berechnet alle Indikatoren für die Multi-Filter EMA-Crossover-Strategie.
    Implementiert mit reinem pandas/numpy (keine externe TA-Bibliothek nötig).

    Erwartet OHLCV-DataFrame mit UTC-DatetimeIndex.
    Gibt bereinigten DataFrame zurück (NaN-Zeilen am Anfang entfernt).
    """
    out = df.copy()
    s = cfg["strategy"]

    # --- EMA (Exponential Moving Average) ---
    # Wilder/pandas EWM: alpha = 2/(N+1), adjust=False = klassisches EMA
    out["ema_fast"] = out["close"].ewm(span=s["ema_fast"], adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=s["ema_slow"], adjust=False).mean()

    # --- ATR (Average True Range) ---
    # Wilder's smoothing: alpha = 1/period
    out["atr"] = _compute_atr(out, s["atr_period"])

    # --- RSI (Relative Strength Index) ---
    out["rsi"] = _compute_rsi(out["close"], s["rsi_period"])

    # --- ADX (Average Directional Index) ---
    out["adx"] = _compute_adx(out, s["adx_period"])

    # --- Volumen-Filter ---
    out["vol_ma"] = out["volume"].rolling(s["volume_ma_period"]).mean()
    out["vol_ratio"] = out["volume"] / out["vol_ma"]

    # --- Preis-Extension (wie weit ist close über EMA(fast)) ---
    out["extended"] = (out["close"] - out["ema_fast"]) / out["ema_fast"] * 100

    # --- Crossover-Erkennung ---
    # cross_up: EMA(fast) kreuzt EMA(slow) von unten
    out["cross_up"] = (
        (out["ema_fast"] > out["ema_slow"]) &
        (out["ema_fast"].shift(1) <= out["ema_slow"].shift(1))
    )
    # cross_dn: EMA(fast) kreuzt EMA(slow) von oben
    out["cross_dn"] = (
        (out["ema_fast"] < out["ema_slow"]) &
        (out["ema_fast"].shift(1) >= out["ema_slow"].shift(1))
    )

    out.dropna(inplace=True)
    return out


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR nach Wilder (alpha = 1/period)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI nach Wilder (EWM mit alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """ADX nach Wilder."""
    alpha = 1.0 / period

    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    # True Range
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = df["high"] - prev_high
    down_move = prev_low - df["low"]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index, dtype=float)
    minus_dm_s = pd.Series(minus_dm, index=df.index, dtype=float)

    smoothed_tr = tr.ewm(alpha=alpha, adjust=False).mean()
    smoothed_plus = plus_dm_s.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus = minus_dm_s.ewm(alpha=alpha, adjust=False).mean()

    eps = np.finfo(float).eps
    plus_di = 100.0 * smoothed_plus / smoothed_tr.replace(0, eps)
    minus_di = 100.0 * smoothed_minus / smoothed_tr.replace(0, eps)

    di_sum = (plus_di + minus_di).replace(0, eps)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum

    return dx.ewm(alpha=alpha, adjust=False).mean()
