"""
Entry-Point für den Multi-Filter EMA-Crossover-Bot.

Verwendung:
  python src/main.py --mode backtest --config src/config/default.yaml
  python src/main.py --mode live     --config src/config/default.yaml

Backtest-Modus:
  Lädt OHLCV-Daten, berechnet Indikatoren und Signale,
  führt den Backtest durch und gibt alle Performance-Metriken aus.
  Optionale Ausgabe der Equity-Kurve als CSV.

Live-Modus:
  Initialisiert MT5 und startet den LiveTrader-Loop.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.common.config import load_config
from src.common.logger import setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Filter EMA-Crossover Tradingbot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["backtest", "live", "train", "optimize"],
                        default="backtest")
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Startkapital für Backtest (in Quote-Währung)")
    parser.add_argument("--fee", type=float, default=0.001,
                        help="Handelsgebühr pro Seite (z.B. 0.001 = 0.1%%)")
    parser.add_argument("--slippage", type=float, default=0.001,
                        help="Slippage pro Seite (z.B. 0.001 = 0.1%%)")
    parser.add_argument("--csv", default=None,
                        help="Pfad zu einer lokalen OHLCV-CSV-Datei (für Backtest ohne MT5)")
    parser.add_argument("--equity-out", default=None,
                        help="CSV-Ausgabepfad für die Equity-Kurve (optional)")
    parser.add_argument("--report-dir", default="reports",
                        help="Verzeichnis für Daily Reports (Live-Modus)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-ml", action="store_true",
                        help="ML-Filter im Backtest aktivieren (Modell muss vorher trainiert sein)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Anzahl Optuna-Trials für --mode optimize")
    return parser.parse_args()


def run_backtest(cfg: dict, args: argparse.Namespace) -> None:
    import pandas as pd
    from src.backtest.costs import CostModel
    from src.backtest.engine import run_backtest as _run
    from src.features.indicators import compute_indicators
    from src.strategies.ema_atr import generate_signals

    logger = logging.getLogger(__name__)

    # Daten laden: CSV oder MT5
    if args.csv:
        logger.info("Lade Daten aus CSV: %s", args.csv)
        df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    else:
        logger.info("Lade Daten von MT5: symbol=%s tf=%s bars=%d",
                    cfg["symbol"], cfg["timeframe"], cfg["data"]["bars"])
        import MetaTrader5 as mt5
        from src.data.mt5._fetch import MT5FetchConfig, fetch_rates, mt5_initialize, mt5_shutdown

        _TIMEFRAME_MAP = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        mt5_initialize()
        try:
            fetch_cfg = MT5FetchConfig(
                symbol=cfg["symbol"],
                timeframe=_TIMEFRAME_MAP[cfg["timeframe"]],
                n_bars=cfg["data"]["bars"],
            )
            df = fetch_rates(fetch_cfg)
        finally:
            mt5_shutdown()

    logger.info("Berechne Indikatoren...")
    df = compute_indicators(df, cfg)

    logger.info("Generiere Signale...")
    df = generate_signals(df, cfg)
    n_raw = (df["signal"] != 0).sum()
    logger.info("%d Rohsignale auf %d Bars.", n_raw, len(df))

    # --- ML-Filter (optional) ---
    if getattr(args, "use_ml", False):
        from src.ml.model_store import ModelStore
        from src.regime.detector import RegimeStats, detect_regime
        from src.strategies.ema_atr import apply_ml_filter

        logger.info("Lade ML-Modell und berechne Regime...")
        store = ModelStore(cfg["ml"].get("model_path", "models/"))
        try:
            meta_model, meta_info = store.load_latest(cfg)
            logger.info("Modell geladen: v%d (AUC=%.3f)", meta_info["version"],
                        meta_info.get("train_metrics", {}).get("auc", 0))
        except FileNotFoundError as e:
            logger.error("%s\n→ Bitte zuerst: python src/main.py --mode train", e)
            return

        regime = detect_regime(df, cfg)
        df = apply_ml_filter(df, cfg, meta_model=meta_model, regime=regime)
        n_filtered = (df["signal"] != 0).sum()
        logger.info("Nach ML-Filter: %d Signale (%d gefiltert)", n_filtered, n_raw - n_filtered)

    cost_model = CostModel(fee_rate=args.fee, slippage_rate=args.slippage)

    logger.info("Starte Backtest (Kapital=%.2f)...", args.capital)
    df_result, trades, metrics = _run(df, cfg, initial_capital=args.capital, cost_model=cost_model)

    _print_metrics(metrics)

    if args.equity_out:
        out_path = Path(args.equity_out)
        df_result[["equity"]].to_csv(out_path)
        logger.info("Equity-Kurve gespeichert: %s", out_path)


def _print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print("  BACKTEST ERGEBNISSE")
    print("=" * 50)
    print(f"  Total Return:      {metrics['total_return_pct']:>8.1f}%")
    print(f"  Annual Return:     {metrics['annual_return_pct']:>8.1f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe']:>8.3f}   (Ziel: > 1.5)")
    print(f"  Sortino Ratio:     {metrics['sortino']:>8.3f}   (Ziel: > 2.0)")
    print(f"  Calmar Ratio:      {metrics['calmar']:>8.3f}   (Ziel: > 1.0)")
    print(f"  Max Drawdown:      {metrics['max_drawdown_pct']:>8.1f}%  (Ziel: < 20%)")
    print(f"  Win Rate:          {metrics['win_rate_pct']:>8.1f}%  (Ziel: > 45%)")
    print(f"  Profit Factor:     {metrics['profit_factor']:>8.3f}   (Ziel: > 1.5)")
    print(f"  Expectancy:        {metrics['expectancy']:>8.2f} USD")
    print(f"  Recovery Factor:   {metrics['recovery_factor']:>8.3f}   (Ziel: > 3)")
    print(f"  Trades gesamt:     {metrics['n_trades']:>8d}")
    print(f"  Avg. Haltedauer:   {metrics['avg_bars_held']:>8.1f} Bars")
    print("=" * 50)
    print()


def run_train(cfg: dict, args: argparse.Namespace) -> None:
    """Trainiert das Meta-Modell via Walk-Forward und speichert das beste Modell."""
    import pandas as pd
    from src.ml.model_store import ModelStore
    from src.ml.walk_forward import walk_forward_train

    logger = logging.getLogger(__name__)

    if args.csv:
        df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    else:
        logger.error("--mode train benötigt --csv <datei>. MT5-Live-Training folgt in der nächsten Version.")
        return

    logger.info("Starte Walk-Forward-Training auf %d Bars...", len(df))
    best_model, results = walk_forward_train(df, cfg)

    logger.info("\nWalk-Forward Ergebnisse:")
    for r in results:
        logger.info("  Fold %d: Train-AUC=%.3f | Test-AUC=%.3f | Labels=%d",
                    r.fold, r.train_metrics.get("auc", 0),
                    r.test_metrics.get("auc", 0), r.n_train_labels)

    store = ModelStore(cfg["ml"].get("model_path", "models/"))
    avg_test_auc = sum(r.test_metrics.get("auc", 0) for r in results) / max(len(results), 1)
    path = store.save(best_model, metadata={
        "description": f"Walk-Forward ({len(results)} Folds)",
        "avg_test_auc": round(avg_test_auc, 4),
        "n_folds": len(results),
    })
    logger.info("Bestes Modell gespeichert: %s", path)

    print("\n=== TOP-10 FEATURE IMPORTANCE ===")
    for feat, imp in best_model.top_features(10).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<25} {bar} {imp:.4f}")
    print()


def run_optimize(cfg: dict, args: argparse.Namespace) -> None:
    """Optuna-Hyperparameter-Optimierung und Ausgabe der besten Parameter."""
    import pandas as pd
    from src.ml.optimizer import optimize_strategy

    logger = logging.getLogger(__name__)

    if not args.csv:
        logger.error("--mode optimize benötigt --csv <datei>.")
        return

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    logger.info("Starte Optuna-Optimierung: %d Trials auf %d Bars...", args.trials, len(df))
    best_params = optimize_strategy(df, cfg, n_trials=args.trials)

    print("\n=== BESTE PARAMETER (Optuna) ===")
    for k, v in sorted(best_params.items()):
        print(f"  {k:<35} {v}")
    print("\n→ Diese Werte in src/config/default.yaml übernehmen.")
    print()


def run_live(cfg: dict, args: argparse.Namespace) -> None:
    from src.Execution.trader import LiveTrader

    report_dir = Path(args.report_dir)
    trader = LiveTrader(cfg, report_dir=report_dir)
    trader.run()


def main() -> int:
    args = _parse_args()
    setup_logger(level=args.log_level)

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config-Fehler: {exc}", file=sys.stderr)
        return 1

    mode_map = {
        "backtest": run_backtest,
        "live": run_live,
        "train": run_train,
        "optimize": run_optimize,
    }
    mode_map[args.mode](cfg, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
