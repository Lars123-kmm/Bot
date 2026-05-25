from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd

from src.backtest.engine import run_backtest
from src.features.indicators import compute_indicators
from src.features.ml_features import build_ml_features
from src.ml.labeling import purge_overlap, triple_barrier_labels
from src.ml.meta_model import MetaModel
from src.strategies.ema_atr import apply_ml_filter, generate_signals

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_strategy(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    n_trials: int = 50,
    holdout_ratio: float = 0.25,
    objective: str = "sharpe",
) -> Dict[str, Any]:
    """
    Optuna-basierte Hyperparameter-Optimierung.
    Optimiert gleichzeitig Strategie-Parameter UND LightGBM-Hyperparameter.

    Suchraum Strategie:
        adx_threshold:     [18, 35]
        rsi_long_min:      [30, 50]
        rsi_long_max:      [55, 75]
        volume_ratio_min:  [1.0, 1.8]
        atr_stop_mult:     [1.5, 3.5]
        atr_tp_mult:       [1.5, 4.5]
        trailing_activate: [1.0, 2.5]

    Suchraum LightGBM:
        num_leaves:        [15, 127]
        learning_rate:     [0.01, 0.15]
        min_child_samples: [5, 60]
        feature_fraction:  [0.5, 1.0]

    Args:
        df:             OHLCV-DataFrame (ohne Indikatoren)
        cfg:            Basis-Konfiguration
        n_trials:       Anzahl Optuna-Trials
        holdout_ratio:  Anteil der Daten als Hold-out (nicht für Training)
        objective:      "sharpe" | "calmar" | "profit_factor"

    Returns:
        Dict mit besten Parametern (kann direkt in cfg['strategy'] / cfg['ml'] gemergt werden)
    """
    n = len(df)
    holdout_start = int(n * (1 - holdout_ratio))
    train_df = df.iloc[:holdout_start].copy()
    holdout_df = df.iloc[holdout_start:].copy()

    logger.info("Optuna-Optimierung: %d Trials, Train=%d Bars, Hold-out=%d Bars",
                n_trials, len(train_df), len(holdout_df))

    def objective_fn(trial: optuna.Trial) -> float:
        trial_cfg = copy.deepcopy(cfg)

        # --- Strategie-Parameter ---
        s = trial_cfg["strategy"]
        s["adx_threshold"] = trial.suggest_int("adx_threshold", 18, 35)
        s["rsi_long_min"] = trial.suggest_int("rsi_long_min", 30, 50)
        s["rsi_long_max"] = trial.suggest_int("rsi_long_max", 55, 75)
        s["rsi_short_min"] = trial.suggest_int("rsi_short_min", 25, 45)
        s["rsi_short_max"] = trial.suggest_int("rsi_short_max", 50, 70)
        s["volume_ratio_min"] = trial.suggest_float("volume_ratio_min", 1.0, 1.8)
        s["atr_stop_mult"] = trial.suggest_float("atr_stop_mult", 1.5, 3.5)
        s["atr_tp_mult"] = trial.suggest_float("atr_tp_mult", 1.5, 4.5)
        s["trailing_activate_atr"] = trial.suggest_float("trailing_activate_atr", 1.0, 2.5)

        if s["rsi_long_min"] >= s["rsi_long_max"]:
            return -999.0
        if s["rsi_short_min"] >= s["rsi_short_max"]:
            return -999.0

        # --- LightGBM-Parameter ---
        ml_cfg = trial_cfg.setdefault("ml", {})
        ml_cfg["lgbm_num_leaves"] = trial.suggest_int("lgbm_num_leaves", 15, 127)
        ml_cfg["lgbm_learning_rate"] = trial.suggest_float("lgbm_learning_rate", 0.01, 0.15, log=True)
        ml_cfg["lgbm_min_child_samples"] = trial.suggest_int("lgbm_min_child_samples", 5, 60)
        ml_cfg["lgbm_n_estimators"] = trial.suggest_int("lgbm_n_estimators", 100, 400)

        try:
            # Indikatoren + Features + Signale berechnen
            df_ind = compute_indicators(train_df, trial_cfg)
            df_sig = generate_signals(df_ind, trial_cfg)
            df_feat = build_ml_features(df_sig, trial_cfg)

            signal_mask = df_feat["signal"] != 0
            if signal_mask.sum() < 15:
                return -999.0

            labels = triple_barrier_labels(
                df_feat, signal_mask,
                upper_mult=float(ml_cfg.get("label_upper_mult", 2.0)),
                lower_mult=float(ml_cfg.get("label_lower_mult", 2.0)),
                max_bars=int(ml_cfg.get("label_max_bars", 48)),
            )
            if len(labels) < 15 or labels["label"].nunique() < 2:
                return -999.0

            from src.features.ml_features import ML_FEATURE_COLS
            X_raw, y_raw = purge_overlap(labels, df_feat)
            X = df_feat.loc[X_raw.index, ML_FEATURE_COLS]
            y = y_raw["label"]

            model = MetaModel(trial_cfg)
            model.fit(X, y)

            # Hold-out Backtest MIT ML-Filter
            df_h_ind = compute_indicators(holdout_df, trial_cfg)
            df_h_sig = generate_signals(df_h_ind, trial_cfg)
            df_h_feat = build_ml_features(df_h_sig, trial_cfg)
            df_h_filt = apply_ml_filter(df_h_feat, trial_cfg, meta_model=model)

            _, _, metrics = run_backtest(df_h_filt, trial_cfg)

            score_map = {
                "sharpe": metrics.get("sharpe", 0),
                "calmar": metrics.get("calmar", 0),
                "profit_factor": min(metrics.get("profit_factor", 0), 10),
            }
            return float(score_map.get(objective, metrics.get("sharpe", 0)))

        except Exception as exc:
            logger.debug("Trial %d fehlgeschlagen: %s", trial.number, exc)
            return -999.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info("Optimierung abgeschlossen: bester Score=%.4f", study.best_value)
    logger.info("Beste Parameter: %s", best)

    return best
