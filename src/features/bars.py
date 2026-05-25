from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def dollar_bars(
    df: pd.DataFrame,
    target_dollar_vol: Optional[float] = None,
    bars_per_day: int = 50,
) -> pd.DataFrame:
    """
    Converts time bars to dollar bars (each bar represents equal $ amount).
    target_dollar_vol: auto-computed when None (30-day median / bars_per_day).
    Returns OHLCV DataFrame in same format as input.
    """
    if target_dollar_vol is None:
        # Estimate from data: median daily dollar volume / bars_per_day
        dollar_vol_series = df["close"] * df["volume"]
        # Group by date, sum, take median, divide
        daily_dvol = dollar_vol_series.resample("1D").sum()
        median_daily = daily_dvol.median()
        target_dollar_vol = max(median_daily / bars_per_day, 1.0)

    return _aggregate_bars(df, threshold=target_dollar_vol, bar_type="dollar")


def volume_bars(
    df: pd.DataFrame,
    target_volume: Optional[float] = None,
    bars_per_day: int = 50,
) -> pd.DataFrame:
    """Analogous to dollar_bars but volume-based."""
    if target_volume is None:
        daily_vol = df["volume"].resample("1D").sum()
        target_volume = max(daily_vol.median() / bars_per_day, 1.0)

    return _aggregate_bars(df, threshold=target_volume, bar_type="volume")


def _aggregate_bars(df: pd.DataFrame, threshold: float, bar_type: str) -> pd.DataFrame:
    """Core aggregation loop for dollar and volume bars."""
    opens, highs, lows, closes, volumes, timestamps = [], [], [], [], [], []

    cum = 0.0
    bar_open = bar_high = bar_low = bar_close = bar_vol = 0.0
    bar_ts = None
    first = True

    for ts, row in df.iterrows():
        o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]

        tick_val = (c * v) if bar_type == "dollar" else v

        if first:
            bar_open = o
            bar_high = h
            bar_low = l
            bar_close = c
            bar_vol = v
            bar_ts = ts
            first = False
        else:
            bar_high = max(bar_high, h)
            bar_low = min(bar_low, l)
            bar_close = c
            bar_vol += v

        cum += tick_val

        if cum >= threshold:
            opens.append(bar_open)
            highs.append(bar_high)
            lows.append(bar_low)
            closes.append(bar_close)
            volumes.append(bar_vol)
            timestamps.append(ts)

            cum = 0.0
            first = True

    if not opens:
        return df.copy()

    result = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=pd.DatetimeIndex(timestamps))

    if df.index.tz is not None and result.index.tz is None:
        result.index = result.index.tz_localize(df.index.tz)

    return result
