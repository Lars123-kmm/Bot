# src/data/mt5_fetch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

import MetaTrader5 as mt5

from src.data.schema import to_utc_index, normalize_volume, validate_candles


@dataclass(frozen=True)
class MT5FetchConfig:
    """
    Minimale Fetch-Konfiguration.
    timeframe: mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, ...
    """
    symbol: str
    timeframe: int
    n_bars: int = 2000
    utc: bool = True  # MT5 liefert i. d. R. epoch seconds; wir konvertieren zu UTC


def mt5_initialize() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")


def mt5_shutdown() -> None:
    mt5.shutdown()


def fetch_rates(cfg: MT5FetchConfig) -> pd.DataFrame:
    """
    Holt die letzten n_bars Candles via MT5 und normalisiert auf:
    - UTC DatetimeIndex 'time'
    - Spalten: open/high/low/close + volume (kanonisch) + spread (falls vorhanden)
    - Validierung via schema.validate_candles
    """
    rates = mt5.copy_rates_from_pos(cfg.symbol, cfg.timeframe, 0, cfg.n_bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos returned None: {mt5.last_error()}")
    if len(rates) == 0:
        raise RuntimeError("Keine Daten zurückgegeben (0 bars).")

    df = pd.DataFrame(rates)

    # MT5: 'time' ist epoch seconds (UTC). pandas kann das sauber.
    # Spalten typischerweise: time, open, high, low, close, tick_volume, spread, real_volume
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Index setzen + UTC erzwingen
    df = to_utc_index(df, time_col="time")

    # Volume normalisieren
    df = normalize_volume(df)

    # Optional: Spalten-Order (nur kosmetisch)
    # open/high/low/close/volume/spread/real_volume (falls du behalten willst)
    # Standard: wir lassen zusätzliche Spalten stehen.

    # Validierung
    validate_candles(df)

    return df


