"""
bootstrap.py — Erstes Training mit Binance-Live-Daten.

Führt einmalig Walk-Forward-Training durch und speichert das Modell,
damit Paper Trading sofort mit ML-Filter startet.

Aufruf:
    python bootstrap.py
    python bootstrap.py --symbols BTCUSDT,ETHUSDT,BNBUSDT
    python bootstrap.py --bars 3000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("bootstrap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Erstes ML-Training mit Binance-Daten")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT",
                        help="Komma-getrennte Symbole (Standard: BTCUSDT,ETHUSDT)")
    parser.add_argument("--bars", type=int, default=5000,
                        help="Anzahl H1-Bars pro Symbol (Standard: 5000 ≈ 7 Monate)")
    parser.add_argument("--config", default="src/config/default.yaml",
                        help="Pfad zur Konfigurations-Datei")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    from src.common.config import load_config
    from src.data.binance_fetch import BinanceFetcher
    from src.features.data_prep import prepare_market_data
    from src.features.indicators import compute_indicators
    from src.ml.model_store import ModelStore
    from src.ml.walk_forward import walk_forward_train

    cfg = load_config(args.config)
    fetcher = BinanceFetcher()
    store = ModelStore(cfg.get("ml", {}).get("model_path", "models/"))

    # ── Schritt 1: Daten laden ────────────────────────────────────────
    logger.info("=== Schritt 1/2: Lade Binance-Daten %s ===", symbols)
    import pandas as pd
    frames = []
    for sym in symbols:
        logger.info("  %s: lade %d H1-Bars...", sym, args.bars)
        df = fetcher.fetch_ohlcv(sym, timeframe="H1", bars=args.bars)
        if df is None or len(df) < 200:
            logger.warning("  %s: zu wenig Daten — übersprungen.", sym)
            continue
        logger.info("  %s: %d Bars geladen.", sym, len(df))
        frames.append(df)

    if not frames:
        logger.error("Keine Daten geladen — Abbruch.")
        sys.exit(1)

    combined = pd.concat(frames).sort_index().drop_duplicates()
    logger.info("Gesamt: %d Bars von %d Symbolen.", len(combined), len(frames))

    # ── Schritt 2: Training ───────────────────────────────────────────
    logger.info("=== Schritt 2/2: Walk-Forward Training (5-15 Min) ===")
    logger.info("Indikatoren & Features werden berechnet...")

    try:
        best_model, results = walk_forward_train(combined, cfg)
    except RuntimeError as exc:
        logger.exception("Training fehlgeschlagen: %s", exc)
        logger.error(
            "Tipp: Zu wenige Signale pro Fold. Lösung:\n"
            "  1. Mehr Bars laden (--bars 8000)\n"
            "  2. Strategie-Filter in default.yaml lockern\n"
            "     adx_threshold: 18  volume_ratio_min: 0.8"
        )
        sys.exit(1)

    avg_auc = sum(r.test_metrics.get("auc", 0) for r in results) / max(len(results), 1)
    logger.info(
        "Training fertig: %d Folds, avg Test-AUC = %.4f",
        len(results), avg_auc,
    )

    store.save(best_model, metadata={
        "description": f"Bootstrap: {','.join(symbols)}, {args.bars} bars",
        "avg_test_auc": round(avg_auc, 4),
        "n_folds": len(results),
        "champion_status": "champion",
    })
    logger.info("Modell gespeichert → %s", cfg.get("ml", {}).get("model_path", "models/"))

    # ── Ergebnis-Zusammenfassung ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BOOTSTRAP ABGESCHLOSSEN")
    print("=" * 55)
    print(f"  Symbole:      {', '.join(symbols)}")
    print(f"  Bars/Symbol:  {args.bars}")
    print(f"  Folds:        {len(results)}")
    print(f"  Avg Test-AUC: {avg_auc:.4f}  (> 0.55 = gut)")
    print("=" * 55)
    if avg_auc >= 0.55:
        print("  ✓ Modell hat Vorhersagekraft — Paper Trading starten:")
        print("    python start.py  → Option 6")
    elif avg_auc >= 0.50:
        print("  ~ Modell knapp über Zufall — Paper Trading möglich,")
        print("    aber mehr Daten wären besser (--bars 8000).")
    else:
        print("  ✗ AUC unter 0.50 — Modell schlechter als Zufall.")
        print("    Mehr Daten laden oder Strategie-Filter anpassen.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
