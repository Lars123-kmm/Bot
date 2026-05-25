from __future__ import annotations

import numpy as np
import pandas as pd


def get_t1(
    signal_idx: pd.DatetimeIndex,
    close_idx: pd.DatetimeIndex,
    max_bars: int,
) -> pd.Series:
    """
    For each signal bar: end time of the label (t + max_bars or end of data).
    Returns pd.Series(end_time, index=signal_idx).
    """
    t1 = {}
    for t0 in signal_idx:
        pos = close_idx.get_loc(t0) if t0 in close_idx else None
        if pos is None:
            continue
        end_pos = min(pos + max_bars, len(close_idx) - 1)
        t1[t0] = close_idx[end_pos]
    return pd.Series(t1)


def num_concurrent_events(close_idx: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """
    For each close bar: number of concurrently active labels.
    Counts how many label intervals [t0_i, t1_i] overlap bar t.
    """
    counts = pd.Series(0, index=close_idx, dtype=float)
    for t0, end in t1.items():
        if t0 in counts.index and end in counts.index:
            counts.loc[t0:end] += 1
    return counts


def avg_uniqueness(t1: pd.Series, num_co: pd.Series) -> pd.Series:
    """
    Average uniqueness per label:
    uniqueness_i = mean(1 / num_concurrent[t0_i : t1_i])
    Higher = less overlap with other labels.
    """
    uniq = {}
    for t0, end in t1.items():
        try:
            co_slice = num_co.loc[t0:end]
            co_slice = co_slice[co_slice > 0]
            uniq[t0] = (1.0 / co_slice).mean() if len(co_slice) > 0 else 1.0
        except KeyError:
            uniq[t0] = 1.0
    return pd.Series(uniq)


def compute_sample_weights(
    labels_df: pd.DataFrame,
    close_idx: pd.DatetimeIndex,
    max_bars: int = 48,
) -> pd.Series:
    """
    Full pipeline → pd.Series(sample_weight, index=labels_df.index).
    sample_weight = uniqueness * abs(return_pct)  (López de Prado §4.4)
    Normalized to mean=1 for LightGBM compatibility.
    """
    t1 = get_t1(labels_df.index, close_idx, max_bars)
    num_co = num_concurrent_events(close_idx, t1)
    uniq = avg_uniqueness(t1, num_co)

    weights = uniq.copy()
    if "return_pct" in labels_df.columns:
        ret_abs = labels_df["return_pct"].abs().reindex(uniq.index).fillna(0.0)
        weights = uniq * (1.0 + ret_abs / max(float(ret_abs.max()), 1e-8))

    weights = weights.fillna(1.0).clip(lower=1e-4)
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights
