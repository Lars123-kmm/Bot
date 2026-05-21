"""Fast Optuna HPO for LightGBM — called automatically during live retraining.

Unlike optimizer.py (which also tunes strategy params and runs a full backtest per trial),
this module focuses purely on LightGBM hyperparameters and uses PurgedKFold CV.
It runs in 2–5 minutes on typical trade history sizes (200–2000 samples).

Each retrain cycle: Optuna finds better LightGBM params → model improves → better
trades → more/better training data → even better params (compound feedback loop).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.ml.purged_kfold import PurgedKFold

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def autotune_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    n_trials: int = 40,
    n_splits: int = 3,
    t1: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Optuna search over LightGBM hyperparameters using PurgedKFold CV.

    Args:
        X:             Feature matrix (ML_FEATURE_COLS or subset).
        y:             Binary labels (1=win, 0=loss).
        sample_weight: Optional López-de-Prado uniqueness weights.
        n_trials:      Optuna trials (40 is fast; 100 for overnight runs).
        n_splits:      PurgedKFold CV folds.
        t1:            Label end-times for purging (optional, uses simple split if None).

    Returns:
        Dict of best params, keyed like cfg['ml'] (e.g. "lgbm_num_leaves").
        Safe to call merge_params(cfg, autotune_lgbm(...)).
    """
    if len(X) < 50 or y.nunique() < 2:
        logger.warning(
            "autotune_lgbm: only %d samples / %d classes — returning defaults.",
            len(X), int(y.nunique()),
        )
        return _defaults()

    sw = sample_weight.values if sample_weight is not None else None
    cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=0.01)
    X_arr = X.values
    y_arr = y.values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        aucs = []
        try:
            for train_idx, test_idx in cv.split(X):
                if len(test_idx) < 5:
                    continue
                X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
                y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
                sw_tr = sw[train_idx] if sw is not None else None
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                clf = LGBMClassifier(**params)
                clf.fit(X_tr, y_tr, sample_weight=sw_tr)
                prob = clf.predict_proba(X_te)[:, 1]
                aucs.append(roc_auc_score(y_te, prob))
        except Exception as exc:
            logger.debug("autotune trial %d failed: %s", trial.number, exc)
            return 0.0
        return float(np.mean(aucs)) if aucs else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    p = study.best_params
    result = {
        "lgbm_num_leaves":        p.get("num_leaves", 50),
        "lgbm_learning_rate":     p.get("learning_rate", 0.05),
        "lgbm_n_estimators":      p.get("n_estimators", 200),
        "lgbm_min_child_samples": p.get("min_child_samples", 20),
    }
    logger.info(
        "autotune_lgbm: %d trials done — best AUC=%.4f | params=%s",
        n_trials, study.best_value, result,
    )
    return result


def merge_params(cfg: Dict[str, Any], best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Inject autotune results into cfg['ml'] (non-destructive copy)."""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("ml", {}).update(best_params)
    return cfg


def _defaults() -> Dict[str, Any]:
    return {
        "lgbm_num_leaves": 50,
        "lgbm_learning_rate": 0.05,
        "lgbm_n_estimators": 200,
        "lgbm_min_child_samples": 20,
    }
