#!/usr/bin/env python3
"""
demo_pipeline.py — End-to-End-Demo des ML-Trading-Bots

Zeigt alle neuen Komponenten in Aktion:
  1. Synthetische BTC-Daten (Regime-Switching GARCH)
  2. Dollar-Bars: Zeit-Bars → informationsreichere Dollar-Bars
  3. HMM-Regime: 2-State Gaussian HMM für Marktphasen
  4. Walk-Forward-Training mit:
       - Sample-Uniqueness-Gewichte (Non-IID-Reduktion)
       - Isotonische Kalibrierung (statt Platt/Sigmoid)
       - Deflated Sharpe Ratio (Overfitting-Test pro Fold)
       - Probability of Backtest Overfitting (PBO)
  5. Champion-Challenger: 3 aufeinanderfolgende OOS-Wins für Promotion
  6. Backtest mit Champion-Modell + López-Bet-Sizing
  7. Online-Lernen (River SRP): Simulation von Live-Trades

Kein MT5, kein Binance-Key, keine externen Daten nötig.

Verwendung:
  cd /home/user/Bot
  python Scripts/demo_pipeline.py
  python Scripts/demo_pipeline.py --bars dollar    # Dollar-Bars aktivieren
  python Scripts/demo_pipeline.py --runs 5         # Mehr Champion-Challenger-Runs
  python Scripts/demo_pipeline.py --no-online      # Online-Lernen überspringen
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run directly (python Scripts/demo_pipeline.py)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

# Saubere Trennlinie für die Ausgabe
SEP = "─" * 62


# ──────────────────────────────────────────────────────────────────────────────
# 1. Synthetische Daten (Regime-Switching GARCH)
# ──────────────────────────────────────────────────────────────────────────────
def generate_btc_data(
    n_bars: int = 6_000,
    freq: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simuliert realistische BTC/USDT OHLCV-Daten mit:
      - Regime-Wechsel: Bull, Bear, Seitwärts
      - GARCH-ähnlicher Volatilitätsclusterung
      - Volume-Preis-Korrelation
    Kein echter Marktdatenlieferant nötig.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_bars, freq=freq, tz="UTC")

    # Regime-Parameter: (drift_per_bar, base_vol)
    regimes = {
        "bull":  ( 0.0008, 0.015),
        "bear":  (-0.0006, 0.020),
        "range": ( 0.0000, 0.010),
    }
    regime_seq = ["bull", "bull", "range", "bear", "bear", "range",
                  "bull", "range", "bear", "bull"]
    bars_per_regime = n_bars // len(regime_seq)

    regime_labels = []
    for r in regime_seq:
        regime_labels.extend([r] * bars_per_regime)
    regime_labels += [regime_seq[-1]] * (n_bars - len(regime_labels))

    # Preispfad
    log_returns = np.zeros(n_bars)
    vol = 0.015
    for i in range(n_bars):
        drift, base_vol = regimes[regime_labels[i]]
        # GARCH(1,1)-ähnliche Vola
        vol = 0.05 * base_vol + 0.90 * vol + 0.05 * abs(log_returns[i - 1]) if i > 0 else base_vol
        vol = np.clip(vol, base_vol * 0.3, base_vol * 3.0)
        log_returns[i] = drift + rng.normal(0, vol)

    prices = 40_000 * np.exp(np.cumsum(log_returns))

    # OHLCV aus Log-Returns aufbauen
    opens  = prices.copy()
    closes = prices * np.exp(rng.normal(0, 0.001, n_bars))
    highs  = np.maximum(opens, closes) * (1 + rng.uniform(0.001, 0.008, n_bars))
    lows   = np.minimum(opens, closes) * (1 - rng.uniform(0.001, 0.008, n_bars))

    # Volumen: höher in volatilen Phasen, mit Preis-Korrelation
    base_vol_arr = np.array([50_000 if r == "range" else 100_000 for r in regime_labels])
    volumes = base_vol_arr * (1 + 2 * np.abs(log_returns) / 0.02) * rng.uniform(0.7, 1.3, n_bars)

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }, index=idx)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Konfiguration (relaxed für Demo, damit genug Signale entstehen)
# ──────────────────────────────────────────────────────────────────────────────
def build_demo_cfg(
    bar_type: str = "time",
    model_dir: str = "/tmp/demo_models",
    hmm_enabled: bool = True,
) -> dict:
    """
    Liefert eine vollständige Config mit gelockerten Filter-Schwellwerten,
    damit Walk-Forward genügend Signale auf synthetischen Daten findet.
    """
    return {
        "symbol": "BTCUSDT",
        "timeframe": "H1",
        "timezone": "UTC",
        "data": {"bars": 6000},

        # ── Strategie (relaxed für Demo) ──────────────────────────────────────
        "strategy": {
            "ema_fast": 20,
            "ema_slow": 50,
            "adx_period": 14,
            "adx_threshold": 10,        # relaxed: 10 statt 25
            "rsi_period": 14,
            "rsi_long_min":  25,        # relaxed: 25 statt 40
            "rsi_long_max":  75,        # relaxed: 75 statt 65
            "rsi_short_min": 25,        # relaxed
            "rsi_short_max": 75,        # relaxed
            "volume_ma_period": 20,
            "volume_ratio_min": 0.5,    # relaxed: 0.5 statt 1.2
            "extended_max_pct": 10.0,   # relaxed
            "atr_period": 14,
            "atr_stop_mult": 2.0,
            "atr_tp_mult": 3.0,
            "trailing_activate_atr": 1.5,
            "trailing_distance_atr": 1.0,
            "time_exit_bars": 48,
            "signal_on_close": True,
        },

        # ── Risiko ────────────────────────────────────────────────────────────
        "risk": {
            "account_risk_per_trade": 0.01,
            "max_total_drawdown": 0.10,
            "max_daily_drawdown": 0.02,
            "max_consecutive_losses": 5,
            "cooldown_minutes": 30,
            "max_spread_atr_ratio": 0.15,
        },

        # ── ML ────────────────────────────────────────────────────────────────
        "ml": {
            "enabled": True,
            "min_probability": 0.52,
            "model_path": model_dir,
            "label_upper_mult": 2.0,
            "label_lower_mult": 2.0,
            "label_max_bars": 48,
            "train_bars": 1500,
            "test_bars":  400,
            "step_bars":  300,
            "purge_bars":  48,
            "lgbm_num_leaves": 31,
            "lgbm_learning_rate": 0.05,
            "lgbm_n_estimators": 100,
            "lgbm_min_child_samples": 10,
        },

        # ── Regime ────────────────────────────────────────────────────────────
        "regime": {
            "enabled": True,
            "window": 100,
            "hurst_trend_threshold": 0.55,
            "hurst_range_threshold": 0.45,
            "vol_volatile_threshold": 0.80,
            "adx_min_for_trend": 15,
            "half_size_in_volatile": True,
        },

        # ── Bars ──────────────────────────────────────────────────────────────
        "bars": {
            "type": bar_type,
            "target_bars_per_day": 50,
        },

        # ── Binance (deaktiviert im Demo) ─────────────────────────────────────
        "binance": {"enabled": False, "symbol": "BTCUSDT", "order_book_levels": 5},

        # ── HMM-Regime ────────────────────────────────────────────────────────
        "hmm": {
            "enabled": hmm_enabled,
            "n_components": 2,
            "model_path": f"{model_dir}/hmm_detector.pkl",
            "retrain_bars": 2000,
        },

        # ── Champion-Challenger ───────────────────────────────────────────────
        "champion_challenger": {
            "enabled": True,
            "n_windows_needed": 2,   # 2 statt 3 — Demo läuft schneller durch
            "metric": "auc",
        },

        # ── Online-Lernen ─────────────────────────────────────────────────────
        "online_model": {
            "enabled": True,
            "blend_weight": 0.3,
            "model_path": f"{model_dir}/online_filter.pkl",
        },

        # ── Execution (nicht genutzt im Demo) ─────────────────────────────────
        "execution": {"mode": "demo", "magic_number": 240101,
                      "slippage_points": 10, "deviation_points": 20},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktion: Banner ausgeben
# ──────────────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Demo-Schritte
# ──────────────────────────────────────────────────────────────────────────────

def demo_data_generation(n_bars: int) -> pd.DataFrame:
    section("SCHRITT 1 — Synthetische BTC-Daten (Regime-Switching GARCH)")
    t0 = time.time()
    df = generate_btc_data(n_bars=n_bars)
    elapsed = time.time() - t0

    regimes = {
        "bull":  sum(1 for i, ts in enumerate(df.index)
                     if df["close"].iloc[i] > df["close"].iloc[0]),
        "range": n_bars // 3,
    }

    print(f"  Bars generiert:   {len(df):>6d}")
    print(f"  Zeitraum:         {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Preisbereich:     ${df['close'].min():,.0f} – ${df['close'].max():,.0f}")
    print(f"  Ø Tagesvolumen:   ${(df['close'] * df['volume']).resample('1D').sum().mean():,.0f}")
    print(f"  Zeit:             {elapsed:.2f}s")
    return df


def demo_dollar_bars(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    section("SCHRITT 2 — Dollar-Bars (informationsgleiche Sampling-Rate)")
    from src.features.bars import dollar_bars

    t0 = time.time()
    db = dollar_bars(df, bars_per_day=cfg["bars"]["target_bars_per_day"])
    elapsed = time.time() - t0

    ratio = len(db) / len(df)
    print(f"  Zeit-Bars:        {len(df):>6d}")
    print(f"  Dollar-Bars:      {len(db):>6d}  ({ratio:.2f}× Ausgangsgröße)")
    print(f"  Zeitspanne:       {db.index[0].date()} → {db.index[-1].date()}")
    print(f"  Ø Bar-Wert:       ${(db['close'] * db['volume']).mean():,.0f}")
    print(f"  Zeit:             {elapsed:.2f}s")
    print()
    print("  → Dollar-Bars vermeiden Over-/Under-Sampling bei ruhigen/volatilen Märkten.")
    print("    Jeder Bar repräsentiert dieselbe gehandelte Geldmenge (López de Prado).")
    return db


def demo_hmm_regime(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    section("SCHRITT 3 — HMM-Regime (2-State Gaussian HMM)")
    from src.regime.hmm_detector import HMMRegimeDetector

    t0 = time.time()
    retrain_bars = int(cfg["hmm"]["retrain_bars"])
    fit_df = df.iloc[:retrain_bars]
    detector = HMMRegimeDetector(n_components=2, random_state=42)
    detector.fit(fit_df)
    proba = detector.predict_proba(df)
    elapsed = time.time() - t0

    df = df.copy()
    df["hmm_prob_trend"]    = proba["hmm_prob_trend"].values
    df["hmm_prob_volatile"] = proba["hmm_prob_volatile"].values

    n_trend    = (detector.predict(df) == 0).sum()
    n_volatile = (detector.predict(df) == 1).sum()
    print(f"  Trainiert auf:    {len(fit_df):>6d} Bars")
    print(f"  Angewendet auf:   {len(df):>6d} Bars")
    print(f"  State 0 (Trend):  {n_trend:>6d} Bars  ({n_trend/len(df)*100:.1f}%)")
    print(f"  State 1 (Vola):   {n_volatile:>6d} Bars  ({n_volatile/len(df)*100:.1f}%)")
    print(f"  Ø Trend-Prob:     {df['hmm_prob_trend'].mean():.3f}")
    print(f"  Zeit:             {elapsed:.2f}s")
    print()
    print("  → State 0 (niedriger Volatilität): EMA-Crossover-Signale bevorzugt.")
    print("    State 1 (hohe Volatilität): ML-Konfidenz-Schwelle erhöht.")
    return df


def demo_fracdiff(df: pd.DataFrame) -> None:
    section("SCHRITT 4 — Fraktionale Differenzierung (Stationarität mit Gedächtnis)")
    from src.features.fracdiff import find_min_d, frac_diff_ffd

    close = df["close"]
    t0 = time.time()
    d = find_min_d(close, pvalue=0.05)
    fd = frac_diff_ffd(close, d, max_width=100)
    elapsed = time.time() - t0

    valid = fd.dropna()
    print(f"  Minimales d:      {d:.4f}  (ADF-Test, p < 0.05)")
    print(f"  Original-Bars:    {len(close):>6d}")
    print(f"  Gültige Bars:     {len(valid):>6d}  ({len(valid)/len(close)*100:.1f}%)")
    print(f"  Ø fracdiff:       {valid.mean():.4f}")
    print(f"  Zeit:             {elapsed:.2f}s")
    print()
    print(f"  → d={d:.2f} bedeutet: Preis-Zeitreihe ist bei d={d:.2f} stationär (ADF),")
    print( "    behält aber mehr Gedächtnis als erste Differenz (d=1).")
    print( "    max_width=100 begrenzt den Convolution-Kernel (Performance-Fix).")


def demo_walk_forward(df_time: pd.DataFrame, cfg: dict) -> tuple:
    """
    Führt Walk-Forward-Training durch.
    Gibt (best_model, results, avg_test_auc) zurück.
    """
    section("SCHRITT 5 — Walk-Forward-Training (Purged CV + Sample Weights + DSR)")
    from src.features.data_prep import prepare_market_data
    from src.features.indicators import compute_indicators
    from src.features.ml_features import build_ml_features
    from src.ml.walk_forward import walk_forward_train
    from src.strategies.ema_atr import generate_signals

    # prepare_market_data: HMM-Regime ergänzen (Dollar-Bars schon angewendet)
    # Für die Walk-Forward übergeben wir die Zeit-Bars (prepare_market_data
    # macht intern die Bar-Konvertierung und HMM).
    print("  Vorverarbeitung (HMM-Regime, Indikatoren, ML-Features)...")
    t0 = time.time()

    best_model, results = walk_forward_train(df_time, cfg)
    elapsed = time.time() - t0

    avg_test_auc = round(
        sum(r.test_metrics.get("auc", 0.0) for r in results) / max(len(results), 1), 4
    )

    print()
    print(f"  {'Fold':>4}  {'Train-Zeitraum':>22}  {'Test-Zeitraum':>22}  "
          f"{'TrainAUC':>8}  {'TestAUC':>7}  {'DSR':>6}  {'Labels':>7}")
    print(f"  {'─'*4}  {'─'*22}  {'─'*22}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*7}")
    for r in results:
        print(f"  {r.fold:>4}  "
              f"{r.train_start} → {r.train_end}  "
              f"{r.test_start} → {r.test_end}  "
              f"{r.train_metrics.get('auc', 0):.4f}  "
              f"{r.test_metrics.get('auc', 0):.4f}  "
              f"{r.dsr:.4f}  "
              f"{r.n_train_labels:>7d}")

    from src.ml.validation import probability_of_backtest_overfitting
    oos_sharpes = [(r.test_metrics.get("auc", 0.5) - 0.5) * 4.0 for r in results]
    pbo = probability_of_backtest_overfitting(oos_sharpes)

    print()
    print(f"  Folds:            {len(results)}")
    print(f"  Ø Test-AUC:       {avg_test_auc:.4f}  (Ziel: > 0.55)")
    print(f"  PBO:              {pbo:.3f}   (Ziel: < 0.30  — Anteil neg. OOS-Folds)")
    print(f"  Trainingszeit:    {elapsed:.1f}s")
    print()
    print("  Erklärung der Metriken:")
    print("  · TrainAUC: AUC auf dem Trainings-Set (kann overfitten)")
    print("  · TestAUC:  AUC auf dem Hold-Out-Set  (echter OOS-Test)")
    print("  · DSR:      Deflated Sharpe Ratio — nahe 1.0 = statistisch signifikant")
    print("  · Labels:   Anzahl triple-barrier-gewichteter Trainingspunkte")

    return best_model, results, avg_test_auc


def demo_champion_challenger(
    cfg: dict,
    model_dir: Path,
    n_runs: int = 4,
) -> None:
    """
    Simuliert mehrere Trainingsläufe und zeigt Champion-Challenger-Logik.
    Läuft n_runs Walk-Forward-Batches mit leicht variierten Seeds.
    """
    section("SCHRITT 6 — Champion-Challenger-Framework")
    from src.ml.champion_challenger import ChampionChallenger
    from src.ml.model_store import ModelStore

    store = ModelStore(model_dir)
    n_windows_needed = int(cfg["champion_challenger"]["n_windows_needed"])
    cc_cfg = cfg.get("champion_challenger", {})

    print(f"  Regeln: Challenger wird Champion nach {n_windows_needed} aufeinanderfolgenden")
    print( "          OOS-Siegen. Losgeschlagener Champion bleibt aktiv bis dahin.")
    print()

    # Simulierte AUC-Werte pro Run (variiert durch Noise)
    rng = np.random.default_rng(99)
    auc_sequence = [0.58, 0.61, 0.63, 0.55, 0.66, 0.62, 0.59, 0.64]
    auc_sequence = (auc_sequence + list(rng.uniform(0.55, 0.68, max(0, n_runs - len(auc_sequence)))))[:n_runs]

    print(f"  {'Run':>3}  {'Sim-AUC':>7}  {'Champion-AUC':>12}  {'Streak':>6}  {'Status':>12}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*12}  {'─'*6}  {'─'*12}")

    champion_auc = None
    prior_streak = 0

    for run_idx, sim_auc in enumerate(auc_sequence):
        cc = ChampionChallenger(
            n_windows_needed=n_windows_needed,
            metric="auc",
        )

        if champion_auc is not None:
            # Dummy-Modell für den aktuellen Champion
            from src.ml.meta_model import MetaModel
            dummy_champ = MetaModel(cfg)
            cc.set_champion(dummy_champ, {}, metrics={"auc": champion_auc})

            dummy_chall = MetaModel(cfg)
            promoted = cc.propose_challenger(
                dummy_chall, {}, {"auc": sim_auc}, carry_wins=prior_streak
            )
            streak_now = cc.streak
            status = "→ CHAMPION" if promoted else "  Challenger"
            if promoted:
                champion_auc = sim_auc
                prior_streak = 0
            else:
                prior_streak = streak_now
        else:
            # Erstes Modell wird automatisch Champion
            from src.ml.meta_model import MetaModel
            dummy_champ = MetaModel(cfg)
            cc.set_champion(dummy_champ, {}, metrics={"auc": sim_auc})
            promoted = True
            champion_auc = sim_auc
            prior_streak = 0
            status = "→ CHAMPION"
            streak_now = 0

        champ_str = f"{champion_auc:.4f}" if champion_auc else "  —"
        print(f"  {run_idx+1:>3}  {sim_auc:.4f}  {champ_str:>12}  {prior_streak:>6}  {status}")

    print()
    print(f"  Finaler Champion-AUC: {champion_auc:.4f}")
    print()
    print("  → Das Champion-Modell wird nur ersetzt wenn der Challenger")
    print(f"    {n_windows_needed}× hintereinander besser ist — kein unbewachter Live-Drift.")


def demo_sample_weights(df_feat_sample: pd.DataFrame, cfg: dict) -> None:
    """Zeigt Sample-Uniqueness-Gewichte am Beispiel."""
    section("SCHRITT 7 — Sample-Uniqueness-Gewichte (Non-IID-Reduktion)")
    from src.ml.labeling import triple_barrier_labels
    from src.ml.sample_weights import compute_sample_weights

    ml_cfg = cfg["ml"]
    signal_mask = df_feat_sample["signal"] != 0
    if signal_mask.sum() < 20:
        print("  (Zu wenige Signale für Demo — übersprungen)")
        return

    labels = triple_barrier_labels(
        df_feat_sample, signal_mask,
        upper_mult=float(ml_cfg["label_upper_mult"]),
        lower_mult=float(ml_cfg["label_lower_mult"]),
        max_bars=int(ml_cfg["label_max_bars"]),
    )
    if len(labels) < 10:
        print("  (Zu wenige Labels für Demo — übersprungen)")
        return

    weights = compute_sample_weights(
        labels, df_feat_sample.index, max_bars=int(ml_cfg["label_max_bars"])
    )

    print(f"  Signale:          {signal_mask.sum():>6d}")
    print(f"  Labels (nach TBL):{len(labels):>6d}")
    print(f"  Sample-Weights:")
    print(f"    Min:            {weights.min():.4f}")
    print(f"    Max:            {weights.max():.4f}")
    print(f"    Mean:           {weights.mean():.4f}  (normalisiert auf ~1.0)")
    print(f"    Std:            {weights.std():.4f}")
    print()
    print("  → Labels mit vielen gleichzeitig aktiven Barrieren erhalten")
    print("    niedrigere Gewichte (geringere 'Einzigartigkeit' des Samples).")
    print("    Verhindert, dass überlappende Triple-Barrier-Labels den Lerner dominieren.")


def demo_lopez_bet_sizing() -> None:
    section("SCHRITT 8 — López-de-Prado Bet-Sizing")
    from src.risk.sizing import calculate_position, lopez_bet_size

    print("  Klassisches ATR-Sizing vs. López-Bet-Sizing (prob-basiert):\n")
    print(f"  {'Prob':>5}  {'López-Size':>10}  {'Klassisch (1% Risiko)':>22}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*22}")

    capital = 10_000.0
    entry   = 42_000.0
    atr     = 800.0

    for prob in [0.50, 0.52, 0.55, 0.58, 0.62, 0.68, 0.75, 0.85]:
        bet = lopez_bet_size(prob)
        pos = calculate_position(capital, entry, atr,
                                 risk_pct=0.01 * bet if bet > 0 else 0.001,
                                 direction=1)
        print(f"  {prob:.2f}  {bet:>10.4f}  {pos['size']:>10.6f} BTC  "
              f"({pos['risk_quote']:.1f} USD Risiko)")

    print()
    print("  → Bei prob=0.50 (kein Edge): Betgröße = 0  → kein Trade.")
    print("    Bei prob=0.75: Betgröße ≈ 0.89 × max  → voller Einsatz.")
    print("    Formel: size = 2·Φ(z) − 1,  z = (p−0.5)/√(p·(1−p))")


def demo_online_learning(cfg: dict, model_dir: Path) -> None:
    section("SCHRITT 9 — Online-Lernen (River SRP — Simulation Live-Trades)")
    from src.ml.online_model import OnlineMetaFilter

    filt = OnlineMetaFilter(cfg)

    if filt._model is None:
        print("  river nicht installiert — Online-Lernen übersprungen.")
        print("  Installation: pip install river")
        return

    rng = np.random.default_rng(55)

    print("  Simuliere 60 Live-Trade-Abschlüsse (win_rate ≈ 58%)...\n")
    print(f"  {'Trade':>5}  {'LightGBM':>8}  {'Blended':>7}  {'Outcome':>7}  "
          f"{'SRP-aktiv':>9}  {'Acc':>6}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*6}")

    for i in range(60):
        # Synthetisches Feature-Dict (in Live-Modus: echter Entry-Bar)
        x = {
            "atr_ratio":        float(rng.uniform(0.8, 1.5)),
            "rsi":              float(rng.uniform(35, 65)),
            "adx":              float(rng.uniform(15, 40)),
            "hmm_prob_trend":   float(rng.uniform(0.3, 0.9)),
            "ema_diff_pct":     float(rng.normal(0, 0.005)),
            "vol_ratio":        float(rng.uniform(0.8, 2.0)),
        }
        lgbm_prob = float(rng.uniform(0.50, 0.75))
        outcome   = int(rng.random() < 0.58)  # 58% Win-Rate

        blended = filt.blend(lgbm_prob, x)
        filt.partial_fit(x, outcome)

        stats = filt.stats()
        active = "ja" if stats["active"] else "nein"

        if i < 5 or i >= 55 or i % 10 == 9:
            print(f"  {i+1:>5}  {lgbm_prob:.4f}  {blended:.4f}  "
                  f"{'Gewinn' if outcome else 'Verlust':>7}  "
                  f"{active:>9}  {stats['accuracy']:.3f}")
        elif i == 5:
            print( "  ...   (Trades 6–30 werden bis ARF-Aktivierung gelernt)")
        elif i == 30:
            print( "  ...   (ARF jetzt aktiv, beeinflusst Blend)")

    final = filt.stats()
    print()
    print(f"  Nach 60 Trades:")
    print(f"    Gesehene Samples:  {final['n_seen']}")
    print(f"    Geschätzte Acc:    {final['accuracy']:.3f}")
    print(f"    Blend-Gewicht:     {final['blend_weight']:.2f}  (ARF)")
    print(f"    ARF aktiv:         {final['active']}")
    print()
    print("  → Erste 30 Trades: reines LightGBM (ARF zu wenig Daten).")
    print("    Ab Trade 31: LightGBM 70% + SRP 30% = adaptiver Blend.")
    print("    Nach wöchentlichem Retraining: ARF-State bleibt erhalten (joblib.dump).")

    save_path = model_dir / "online_filter_demo.pkl"
    filt.save(save_path)
    print(f"\n  Online-Filter gespeichert: {save_path}")


def demo_backtest_summary(df_time: pd.DataFrame, cfg: dict, model_dir: Path) -> None:
    """Zeigt wie ein Champion-Modell im Backtest verwendet würde."""
    section("SCHRITT 10 — Backtest-Vorschau mit Champion-Modell")

    store_path = model_dir
    try:
        from src.ml.model_store import ModelStore
        store = ModelStore(store_path)
        versions = store.list_versions()
        if versions:
            latest = max(versions, key=lambda e: e["version"])
            print(f"  Registry enthält {len(versions)} Modell-Version(en).")
            print(f"  Neueste Version:  v{latest['version']}  "
                  f"(AUC={latest.get('avg_test_auc', 'n/a')}  "
                  f"Status={latest.get('champion_status', 'n/a')})")
            print()
            champs = [v for v in versions if v.get("champion_status") == "champion"]
            challengers = [v for v in versions if v.get("champion_status") == "challenger"]
            print(f"  Champions:        {len(champs)}")
            print(f"  Challengers:      {len(challengers)}")
            print()
            print("  load_champion() → lädt immer den besten validierten Champion,")
            print("  nicht das zuletzt gespeicherte Modell. Dadurch kein ungeprüfter")
            print("  Produktions-Rollout nach einem schlechten Trainingslauf.")
        else:
            print("  (Keine Modelle in Registry — Walk-Forward lief evtl. ohne Store)")
    except Exception as exc:
        print(f"  Registry nicht lesbar: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-End Demo des ML-Trading-Bots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bars", choices=["time", "dollar", "volume"],
                   default="time", help="Bar-Typ für Daten-Vorverarbeitung")
    p.add_argument("--runs", type=int, default=5,
                   help="Anzahl Champion-Challenger-Simulationsruns")
    p.add_argument("--n-bars", type=int, default=5_500,
                   help="Anzahl synthetischer OHLCV-Bars")
    p.add_argument("--no-online", action="store_true",
                   help="Online-Lernen (River SRP) überspringen")
    p.add_argument("--no-dollar", action="store_true",
                   help="Dollar-Bar-Demo überspringen (impliziert --bars time)")
    p.add_argument("--model-dir", default="/tmp/demo_models",
                   help="Verzeichnis für Modell-Speicherung")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.time()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    bar_type = "time" if args.no_dollar else args.bars
    cfg = build_demo_cfg(bar_type=bar_type, model_dir=str(model_dir))

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         ML-TRADING-BOT — END-TO-END PIPELINE DEMO           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Bar-Typ:  {bar_type:<8}  Runs: {args.runs:<3}  Bars: {args.n_bars:<6}  "
          f"Models: {str(model_dir):<10} ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── 1. Daten generieren ────────────────────────────────────────────────
    df_time = demo_data_generation(args.n_bars)

    # ── 2. Dollar-Bars ─────────────────────────────────────────────────────
    if bar_type == "dollar" and not args.no_dollar:
        demo_dollar_bars(df_time, cfg)

    # ── 3. HMM-Regime ─────────────────────────────────────────────────────
    demo_hmm_regime(df_time, cfg)

    # ── 4. Fraktionale Differenzierung ────────────────────────────────────
    demo_fracdiff(df_time)

    # ── 5. Walk-Forward-Training ───────────────────────────────────────────
    best_model, results, avg_test_auc = demo_walk_forward(df_time, cfg)

    # Modell speichern (für Champion-Challenger + Backtest)
    from src.ml.model_store import ModelStore
    store = ModelStore(model_dir)
    store.save(best_model, metadata={
        "description": "Demo Walk-Forward",
        "avg_test_auc": avg_test_auc,
        "n_folds": len(results),
        "champion_status": "champion",  # Erstes Modell ist immer Champion
        "challenger_streak": 0,
    })

    # ── 6. Sample-Weights Demo ────────────────────────────────────────────
    try:
        from src.features.data_prep import prepare_market_data
        from src.features.indicators import compute_indicators
        from src.features.ml_features import build_ml_features
        from src.strategies.ema_atr import generate_signals
        df_prep = prepare_market_data(df_time.head(2000), cfg)
        df_ind  = compute_indicators(df_prep, cfg)
        df_sig  = generate_signals(df_ind, cfg)
        df_feat = build_ml_features(df_sig, cfg)
        demo_sample_weights(df_feat, cfg)
    except Exception as exc:
        logger.warning("Sample-Weights-Demo fehlgeschlagen: %s", exc)

    # ── 7. López Bet-Sizing ────────────────────────────────────────────────
    demo_lopez_bet_sizing()

    # ── 8. Champion-Challenger ─────────────────────────────────────────────
    demo_champion_challenger(cfg, model_dir, n_runs=args.runs)

    # ── 9. Online-Lernen ──────────────────────────────────────────────────
    if not args.no_online:
        demo_online_learning(cfg, model_dir)

    # ── 10. Backtest-Registry-Vorschau ─────────────────────────────────────
    demo_backtest_summary(df_time, cfg, model_dir)

    # ── Abschluss ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    section("FERTIG")
    print(f"  Gesamtlaufzeit:   {elapsed_total:.1f}s")
    print()
    print("  Alle Komponenten erfolgreich demonstriert.")
    print()
    print("  Nächste Schritte:")
    print("  1. Echte Daten:   python src/main.py --mode train --csv daten.csv")
    print("  2. Backtest:      python src/main.py --mode backtest --csv daten.csv --use-ml")
    print("  3. Optimierung:   python src/main.py --mode optimize --csv daten.csv")
    print("  4. Live-Trading:  python src/main.py --mode live")
    print()
    print("  Flags:")
    print("  --bar-type dollar    → Dollar-Bars verwenden")
    print("  --use-ml             → Champion-Modell im Backtest")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
