from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    atr_col: str = "atr",
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
    max_bars: int = 48,
    direction_col: str = "signal",
) -> pd.DataFrame:
    """
    Triple-Barrier-Labeling (de Prado: Advances in Financial Machine Learning).

    Für jeden aktiven Signal-Bar: Welche Barriere wird zuerst getroffen?
      - Obere Barriere (TP): Entry + upper_mult * ATR  → Label 1
      - Untere Barriere (SL): Entry - lower_mult * ATR → Label 0
      - Zeit-Barriere: nach max_bars ohne Treffer      → Label 0

    Args:
        df:            OHLCV + Indikatoren DataFrame
        signal_mask:   Boolean Series: True = Signal aktiv an diesem Bar
        atr_col:       Spaltenname des ATR
        upper_mult:    ATR-Multiplikator für TP-Barriere
        lower_mult:    ATR-Multiplikator für SL-Barriere
        max_bars:      Maximale Haltedauer (Zeit-Barriere)
        direction_col: Spalte mit 1/−1 (Long/Short) für Richtungs-Barrieren

    Returns:
        DataFrame mit Spalten: label (0/1), bars_to_exit, return_pct
        Index = Signal-Bar-Zeitstempel
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df[atr_col].values
    directions = df[direction_col].values if direction_col in df.columns else np.ones(len(df))

    results = []
    signal_indices = np.where(signal_mask.values)[0]

    for idx in signal_indices:
        entry = closes[idx]
        atr = atrs[idx]
        direction = int(directions[idx]) if directions[idx] != 0 else 1

        upper = entry + upper_mult * atr * direction
        lower = entry - lower_mult * atr * direction

        label = 0
        bars_to_exit = max_bars
        exit_price = entry

        for j in range(1, min(max_bars + 1, len(df) - idx)):
            future_idx = idx + j
            h = highs[future_idx]
            l = lows[future_idx]
            c = closes[future_idx]

            if direction == 1:  # Long
                if h >= upper:
                    label = 1
                    exit_price = upper
                    bars_to_exit = j
                    break
                if l <= lower:
                    label = 0
                    exit_price = lower
                    bars_to_exit = j
                    break
            else:  # Short
                if l <= upper:  # upper = entry - mult*atr für Short
                    label = 1
                    exit_price = upper
                    bars_to_exit = j
                    break
                if h >= lower:
                    label = 0
                    exit_price = lower
                    bars_to_exit = j
                    break

            if j == max_bars:
                exit_price = c
                bars_to_exit = j

        return_pct = (exit_price - entry) / entry * direction * 100

        results.append({
            "label": label,
            "bars_to_exit": bars_to_exit,
            "return_pct": round(return_pct, 4),
        })

    if not results:
        return pd.DataFrame(columns=["label", "bars_to_exit", "return_pct"])

    return pd.DataFrame(results, index=df.index[signal_indices])


def purge_overlap(
    labels: pd.DataFrame,
    feature_df: pd.DataFrame,
    max_bars: int = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entfernt überlappende Beobachtungen zwischen benachbarten Labels.
    Verhindert Label-Leakage in Walk-Forward-Cross-Validation.

    Returns: (features_purged, labels_purged) – synchronisierte DataFrames.
    """
    keep_idx = []
    last_exit_pos = -1

    label_positions = []
    valid_label_ts = []
    for ts in labels.index:
        if ts in feature_df.index:
            loc = feature_df.index.get_loc(ts)
            # get_loc returns a slice when the index has duplicate timestamps
            pos = loc.start if isinstance(loc, slice) else int(loc)
            label_positions.append(pos)
            valid_label_ts.append(ts)

    for i, pos in enumerate(label_positions):
        if pos > last_exit_pos:
            keep_idx.append(valid_label_ts[i])
            last_exit_pos = pos + max_bars

    labels_purged = labels.loc[keep_idx]
    features_purged = feature_df.loc[feature_df.index.isin(keep_idx)]
    return features_purged, labels_purged
