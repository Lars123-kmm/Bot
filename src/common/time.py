# src/common/time.py
from __future__ import annotations
import pandas as pd

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience-Wrapper: index -> tz-aware UTC, Name -> 'time'.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex.")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "time"
    return out
