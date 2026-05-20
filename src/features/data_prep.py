from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def prepare_market_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Optionale Vorverarbeitung der OHLCV-Daten vor der Indikator-Berechnung.
    Alle Schritte sind config-gesteuert; mit Default-Config (bars.type='time',
    binance.enabled=false, hmm.enabled=true) wird nur das HMM-Regime ergänzt.

    Reihenfolge:
      1. Bar-Typ-Konvertierung (Zeit → Dollar/Volume-Bars)
      2. Binance-Enrichment (Funding Rate, OI, Order-Book-Imbalance)
      3. HMM-Regime-Wahrscheinlichkeiten (hmm_prob_trend / hmm_prob_volatile)
    """
    out = df

    # --- 1. Bar-Typ ---
    bars_cfg = cfg.get("bars", {})
    bar_type = bars_cfg.get("type", "time")
    per_day = int(bars_cfg.get("target_bars_per_day", 50))
    if bar_type == "dollar":
        from src.features.bars import dollar_bars
        out = dollar_bars(out, bars_per_day=per_day)
        logger.info("Dollar-Bars: %d Zeit-Bars → %d Dollar-Bars", len(df), len(out))
    elif bar_type == "volume":
        from src.features.bars import volume_bars
        out = volume_bars(out, bars_per_day=per_day)
        logger.info("Volume-Bars: %d Zeit-Bars → %d Volume-Bars", len(df), len(out))

    # --- 2. Binance-Enrichment ---
    binance_cfg = cfg.get("binance", {})
    if binance_cfg.get("enabled", False):
        try:
            from src.data.binance_fetch import BinanceFetcher
            fetcher = BinanceFetcher()
            out = fetcher.enrich_ohlcv(out, binance_cfg.get("symbol", "BTCUSDT"))
            logger.info("Binance-Enrichment: funding_rate, oi_change, OBI ergänzt")
        except Exception as exc:
            logger.warning("Binance-Enrichment fehlgeschlagen (%s) — übersprungen.", exc)

    # --- 3. HMM-Regime ---
    hmm_cfg = cfg.get("hmm", {})
    if hmm_cfg.get("enabled", False):
        try:
            out = _add_hmm_features(out, hmm_cfg)
        except Exception as exc:
            logger.warning("HMM-Regime fehlgeschlagen (%s) — 0.5-Fallback bleibt.", exc)

    return out


def _add_hmm_features(df: pd.DataFrame, hmm_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Ergänzt hmm_prob_trend / hmm_prob_volatile Spalten.

    Das HMM wird nur auf den ersten `retrain_bars` Bars trainiert und dann
    vorwärts angewendet — so entsteht kein Look-ahead-Bias für spätere Bars
    (relevant im Walk-Forward-Kontext).
    """
    from src.regime.hmm_detector import HMMRegimeDetector

    retrain_bars = int(hmm_cfg.get("retrain_bars", 2000))
    fit_df = df.iloc[:retrain_bars] if len(df) > retrain_bars else df

    detector = HMMRegimeDetector(n_components=int(hmm_cfg.get("n_components", 2)))
    detector.fit(fit_df)
    proba = detector.predict_proba(df)

    out = df.copy()
    out["hmm_prob_trend"] = proba["hmm_prob_trend"].values
    out["hmm_prob_volatile"] = proba["hmm_prob_volatile"].values
    logger.info(
        "HMM-Regime: auf %d Bars trainiert, trend-prob mean=%.3f",
        len(fit_df), out["hmm_prob_trend"].mean(),
    )
    return out
