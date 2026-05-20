from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights_ffd(d: float, size: int, thres: float = 1e-5) -> np.ndarray:
    """Fixed-Width Window weights for fractional differentiation (López de Prado Ch. 5)."""
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
    return np.array(w[::-1])


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    thres: float = 1e-5,
    max_width: int = 100,
) -> pd.Series:
    """
    Fractionally differenced series using Fixed-Width Window method.
    d=0 → original, d=1 → first difference. Preserves more memory than d=1.

    max_width caps the convolution window so the first max_width-1 bars
    produce valid output (instead of size-1 bars with default thres stopping).
    For d < 0.5 the weights decay very slowly; without max_width the window
    would span nearly the full series and discard almost all rows.
    """
    w = get_weights_ffd(d, min(len(series), max_width), thres)
    width = len(w)
    output = {}
    for i in range(width - 1, len(series)):
        loc = series.index[i]
        window = series.iloc[i - width + 1: i + 1].values
        output[loc] = float(np.dot(w, window))
    # Reindex to the full series: the first width-1 bars become leading NaNs.
    return pd.Series(output, dtype=float, name=series.name).reindex(series.index)


def find_min_d(
    series: pd.Series,
    pvalue: float = 0.05,
    d_range: tuple = (0.0, 1.0),
    step: float = 0.05,
) -> float:
    """
    Finds minimum d such that ADF test confirms stationarity (p < pvalue).
    Starts search at d=0.1 — typically d≈0.1–0.4 is sufficient.
    """
    for d in np.arange(d_range[0] + step, d_range[1] + step, step):
        d = round(float(d), 4)
        fd = frac_diff_ffd(series, d)
        fd = fd.dropna()
        if len(fd) < 20:
            continue
        try:
            pval = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
            if pval < pvalue:
                return d
        except Exception:
            continue
    return round(d_range[1], 4)
