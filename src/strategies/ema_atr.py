from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.ml.meta_model import MetaModel


def generate_signals(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Erzeugt Handelssignale basierend auf dem Multi-Filter EMA-Crossover-System.

    Bedingungen Long:
      1. EMA(fast) kreuzt EMA(slow) von unten (cross_up)
      2. ADX > adx_threshold
      3. RSI zwischen rsi_long_min und rsi_long_max
      4. Volumen > volume_ratio_min * MA(volume_ma_period)
      5. Preis < extended_max_pct% über EMA(fast)

    Bedingungen Short (spiegelbildlich, ohne extended-Filter):
      1. EMA(fast) kreuzt EMA(slow) von oben (cross_dn)
      2. ADX > adx_threshold
      3. RSI zwischen rsi_short_min und rsi_short_max
      4. Volumen > volume_ratio_min * MA(volume_ma_period)

    Signal wird um 1 Bar verschoben (kein Look-ahead-Bias).
    """
    out = df.copy()
    s = cfg["strategy"]

    required = ["cross_up", "cross_dn", "adx", "rsi", "vol_ratio", "extended"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Fehlende Indikator-Spalten: {missing}. compute_indicators() zuerst aufrufen.")

    long_signal = (
        out["cross_up"] &
        (out["adx"] > s["adx_threshold"]) &
        out["rsi"].between(s["rsi_long_min"], s["rsi_long_max"]) &
        (out["vol_ratio"] > s["volume_ratio_min"]) &
        (out["extended"] < s["extended_max_pct"])
    )

    short_signal = (
        out["cross_dn"] &
        (out["adx"] > s["adx_threshold"]) &
        out["rsi"].between(s["rsi_short_min"], s["rsi_short_max"]) &
        (out["vol_ratio"] > s["volume_ratio_min"])
    )

    out["signal"] = 0
    out.loc[long_signal, "signal"] = 1
    out.loc[short_signal, "signal"] = -1
    out["signal"] = out["signal"].shift(1).fillna(0).astype(int)

    return out


def apply_ml_filter(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    meta_model: Optional["MetaModel"] = None,
    regime: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Schicht-2 + Schicht-3 Filter auf bestehende EMA-Signale.

    1. Regime-Gate:   signal = 0 wenn regime == RANGE (0)
                      sizing_scale = 0.5 wenn regime == VOLATILE (2)
    2. Meta-Modell:   signal = 0 wenn proba < threshold
    3. Sizing-Faktor: df['meta_prob'] für probability_scaled_position()

    Args:
        df:          DataFrame mit 'signal'-Spalte (aus generate_signals)
        cfg:         Konfigurations-Dict
        meta_model:  Trainiertes MetaModel (optional)
        regime:      pd.Series mit Regime-Klassifikation (0/1/2, optional)

    Returns:
        DataFrame mit ggf. gefilterten Signalen und neuer Spalte 'meta_prob'
    """
    from src.regime.detector import RANGE, VOLATILE

    out = df.copy()
    out["meta_prob"] = 0.65     # Default-Konfidenz wenn kein Modell
    out["sizing_scale"] = 1.0   # Standard-Sizing

    # --- Schicht 2: Regime-Gate ---
    if regime is not None:
        regime_aligned = regime.reindex(out.index, fill_value=RANGE)
        # RANGE → kein Trade
        out.loc[regime_aligned == RANGE, "signal"] = 0
        # VOLATILE → halbes Sizing
        half_size_in_volatile = cfg.get("regime", {}).get("half_size_in_volatile", True)
        if half_size_in_volatile:
            out.loc[regime_aligned == VOLATILE, "sizing_scale"] = 0.5

    # --- Schicht 3: Meta-Modell-Gate ---
    if meta_model is not None:
        from src.features.ml_features import ML_FEATURE_COLS, build_ml_features
        try:
            df_feat = build_ml_features(out, cfg)
            # Nur auf Bars mit tatsächlich vorhandenen Features predicten
            common_idx = df_feat.index.intersection(out.index)
            if len(common_idx) > 0:
                X = df_feat.loc[common_idx, ML_FEATURE_COLS]
                probas = meta_model.predict_proba(X)
                out.loc[common_idx, "meta_prob"] = probas

                # Signale unter dem Threshold deaktivieren
                signal_bars = out.loc[common_idx, "signal"] != 0
                low_confidence = out.loc[common_idx, "meta_prob"] < meta_model.threshold
                out.loc[common_idx[signal_bars & low_confidence], "signal"] = 0
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("ML-Filter übersprungen: %s", exc)

    return out
