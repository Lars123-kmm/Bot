# src/data/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import pandas as pd


OHLC_COLS = ("open", "high", "low", "close")
VOLUME_ALIASES = ("volume", "tick_volume", "real_volume")

# Optional, aber empfohlen: normierter Spaltenname im System
CANONICAL_VOLUME_COL = "volume"


@dataclass(frozen=True)
class DataSchema:
    """
    Single Source of Truth für Candle-Daten.

    Konventionen:
    - Index: timezone-aware DatetimeIndex in UTC
    - Spalten: open/high/low/close float
    - volume: int/float (kanonisch: 'volume')
    - spread optional (Einheit muss im README dokumentiert sein)
    """
    ohlc_cols: tuple[str, str, str, str] = OHLC_COLS
    canonical_volume: str = CANONICAL_VOLUME_COL
    allow_spread: bool = True


def to_utc_datetime_index(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Erzwingt einen tz-aware UTC DatetimeIndex.
    - Wenn time_col gesetzt: nimmt df[time_col] als Zeitquelle.
    - Sonst: erwartet DatetimeIndex.
    """
    out = df.copy()

    if time_col is not None:
        if time_col not in out.columns:
            raise ValueError(f"time_col '{time_col}' fehlt in DataFrame.")
        idx = pd.to_datetime(out[time_col], utc=True, errors="raise")
        out = out.drop(columns=[time_col])
        out.index = idx
    else:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("DataFrame benötigt einen DatetimeIndex oder time_col.")
        # Wenn index naive ist: als UTC interpretieren (oder bewusst failen – siehe Kommentar unten)
        if out.index.tz is None:
            # Alternative streng: raise ValueError("Naiver DatetimeIndex; tz fehlt.")
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")

    out.index.name = "time"
    return out


def normalize_volume_column(df: pd.DataFrame, schema: DataSchema = DataSchema()) -> pd.DataFrame:
    """
    Vereinheitlicht volume-Spalte auf schema.canonical_volume.
    """
    out = df.copy()
    if schema.canonical_volume in out.columns:
        return out

    for c in VOLUME_ALIASES:
        if c in out.columns:
            out[schema.canonical_volume] = out[c]
            if c != schema.canonical_volume:
                out = out.drop(columns=[c])
            return out

    raise ValueError(f"Keine volume-Spalte gefunden. Erwartet eine von {VOLUME_ALIASES}.")


def validate_candles(
    df: pd.DataFrame,
    schema: DataSchema = DataSchema(),
    require_monotonic: bool = True,
    require_unique_index: bool = True,
) -> None:
    """
    Harter Validator für alle Eingänge (Fetch, CSV, Backtest, Live).
    Wirft ValueError bei Verstößen.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index muss DatetimeIndex sein.")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex muss timezone-aware sein (UTC empfohlen).")

    if require_monotonic and not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex ist nicht monoton steigend.")

    if require_unique_index and not df.index.is_unique:
        dup = df.index[df.index.duplicated()].unique()
        raise ValueError(f"Doppelte Zeitstempel im Index: {dup[:5]}{'...' if len(dup) > 5 else ''}")

    missing = [c for c in schema.ohlc_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende OHLC Spalten: {missing}")

    # Typ-/Wertechecks
    for c in schema.ohlc_cols:
        if df[c].isna().any():
            raise ValueError(f"NaNs in Spalte '{c}'.")
    if (df["high"] < df["low"]).any():
        raise ValueError("Ungültige Candles: high < low entdeckt.")
    if ((df["close"] > df["high"]) | (df["close"] < df["low"])).any():
        raise ValueError("Ungültige Candles: close außerhalb [low, high].")
    if ((df["open"] > df["high"]) | (df["open"] < df["low"])).any():
        raise ValueError("Ungültige Candles: open außerhalb [low, high].")

    if schema.canonical_volume not in df.columns:
        raise ValueError(f"Volume-Spalte fehlt: '{schema.canonical_volume}' (normalisieren via normalize_volume_column).")

    if schema.allow_spread and ("spread" in df.columns):
        if df["spread"].isna().any():
            raise ValueError("NaNs in 'spread'.")
        if (df["spread"] < 0).any():
            raise ValueError("Negative spread-Werte entdeckt.")
