from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.features.data_prep import prepare_market_data
from src.features.indicators import compute_indicators
from src.features.ml_features import ML_FEATURE_COLS, build_ml_features
from src.ml.labeling import purge_overlap, triple_barrier_labels
from src.ml.meta_model import MetaModel
from src.ml.sample_weights import compute_sample_weights
from src.ml.validation import deflated_sharpe_ratio, probability_of_backtest_overfitting
from src.strategies.ema_atr import generate_signals

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    model: MetaModel
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]     # AUC, Accuracy auf Test-Set
    n_train_labels: int
    n_test_labels: int
    dsr: float = 0.0                   # Deflated Sharpe Ratio für diesen Fold
    feature_importance: Dict[str, float] = field(default_factory=dict)


def walk_forward_train(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    train_bars: Optional[int] = None,
    test_bars: Optional[int] = None,
    step_bars: Optional[int] = None,
    purge_bars: Optional[int] = None,
) -> Tuple[MetaModel, List[WalkForwardResult]]:
    """
    Purged Walk-Forward-Cross-Validation für das Meta-Modell.

    Rollendes Fenster-Schema:
      Fold 1: Train [0..T]        → Gap [purge]  → Test [T+purge..T+purge+test]
      Fold 2: Train [step..T+step] → Gap [purge] → Test [T+step+purge..]
      ...

    Args:
        df:          OHLCV-DataFrame (noch OHNE Indikatoren)
        cfg:         Konfigurations-Dict
        train_bars:  Überschreibt cfg['ml']['train_bars']
        test_bars:   Überschreibt cfg['ml']['test_bars']
        step_bars:   Überschreibt cfg['ml']['step_bars']
        purge_bars:  Überschreibt cfg['ml']['purge_bars']

    Returns:
        (bestes_modell, alle_fold_ergebnisse)
        Bestes Modell = höchste AUC auf Test-Set.
    """
    ml_cfg = cfg.get("ml", {})
    train_bars = train_bars or int(ml_cfg.get("train_bars", 2000))
    test_bars = test_bars or int(ml_cfg.get("test_bars", 500))
    step_bars = step_bars or int(ml_cfg.get("step_bars", 250))
    purge_bars = purge_bars or int(ml_cfg.get("purge_bars", 48))

    # Daten-Vorverarbeitung (Bar-Typ, Binance, HMM) + Indikatoren + Features
    logger.info("Berechne Indikatoren und ML-Features für Walk-Forward...")
    df = prepare_market_data(df, cfg)
    df_ind = compute_indicators(df, cfg)
    df_sig = generate_signals(df_ind, cfg)
    df_feat = build_ml_features(df_sig, cfg)

    results: List[WalkForwardResult] = []
    best_model: Optional[MetaModel] = None
    best_auc = -1.0
    oos_sharpes: List[float] = []

    fold = 0
    start = 0

    while start + train_bars + purge_bars + test_bars <= len(df_feat):
        train_end = start + train_bars
        test_start = train_end + purge_bars
        test_end = test_start + test_bars

        train_df = df_feat.iloc[start:train_end]
        test_df = df_feat.iloc[test_start:test_end]

        if len(train_df) < 100 or len(test_df) < 20:
            break

        # Labels für Train-Set
        train_signal_mask = (train_df["signal"] != 0)
        if train_signal_mask.sum() < 10:
            logger.warning("Fold %d: zu wenige Signale im Train-Set (%d). Übersprungen.",
                           fold, train_signal_mask.sum())
            start += step_bars
            fold += 1
            continue

        train_labels = triple_barrier_labels(
            train_df, train_signal_mask,
            upper_mult=float(ml_cfg.get("label_upper_mult", 2.0)),
            lower_mult=float(ml_cfg.get("label_lower_mult", 2.0)),
            max_bars=int(ml_cfg.get("label_max_bars", 48)),
        )

        X_train_raw, y_train_labels = purge_overlap(train_labels, train_df,
                                                     max_bars=int(ml_cfg.get("label_max_bars", 48)))

        if len(y_train_labels) < 10:
            logger.warning("Fold %d: nach Purging zu wenige Labels (%d). Übersprungen.",
                           fold, len(y_train_labels))
            start += step_bars
            fold += 1
            continue

        X_train = train_df.loc[X_train_raw.index, ML_FEATURE_COLS]
        y_train = y_train_labels["label"]

        # Sample-Uniqueness-Gewichte (López de Prado §4.4) — reduziert Non-IID-Bias
        sample_weight = compute_sample_weights(
            y_train_labels,
            train_df.index,
            max_bars=int(ml_cfg.get("label_max_bars", 48)),
        )

        # Modell trainieren
        model = MetaModel(cfg)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        # Labels für Test-Set
        test_signal_mask = (test_df["signal"] != 0)
        test_metrics = {"auc": 0.0, "n_signals": int(test_signal_mask.sum())}

        if test_signal_mask.sum() >= 5:
            test_labels = triple_barrier_labels(
                test_df, test_signal_mask,
                upper_mult=float(ml_cfg.get("label_upper_mult", 2.0)),
                lower_mult=float(ml_cfg.get("label_lower_mult", 2.0)),
                max_bars=int(ml_cfg.get("label_max_bars", 48)),
            )
            if len(test_labels) >= 5:
                X_test = test_df.loc[test_labels.index, ML_FEATURE_COLS].dropna()
                y_test = test_labels.loc[X_test.index, "label"]
                if len(y_test) >= 5 and y_test.nunique() > 1:
                    probas = model.predict_proba(X_test)
                    from sklearn.metrics import roc_auc_score
                    auc = float(roc_auc_score(y_test, probas))
                    accuracy = float(((probas >= model.threshold) == y_test).mean())
                    test_metrics = {"auc": round(auc, 4), "accuracy": round(accuracy, 4),
                                    "n_signals": len(y_test)}

        # DSR: proxy-SR aus OOS-AUC, angepasst für Anzahl Trials (Folds)
        oos_auc = test_metrics.get("auc", 0.5)
        pseudo_sr = (oos_auc - 0.5) * 4.0  # AUC=0.7 → SR≈0.8
        oos_sharpes.append(pseudo_sr)
        fold_dsr = deflated_sharpe_ratio(
            sr=pseudo_sr,
            t=max(test_metrics.get("n_signals", 30), 10),
            n_trials=fold + 1,
        )

        result = WalkForwardResult(
            fold=fold,
            train_start=str(train_df.index[0].date()),
            train_end=str(train_df.index[-1].date()),
            test_start=str(test_df.index[0].date()),
            test_end=str(test_df.index[-1].date()),
            model=model,
            train_metrics=model.train_metrics_,
            test_metrics=test_metrics,
            n_train_labels=len(y_train),
            n_test_labels=test_metrics.get("n_signals", 0),
            dsr=round(fold_dsr, 4),
            feature_importance=model.top_features(10),
        )
        results.append(result)

        logger.info(
            "Fold %d: Train=%s→%s | Test=%s→%s | Train-AUC=%.3f | Test-AUC=%.3f | DSR=%.3f | Labels=%d",
            fold, result.train_start, result.train_end,
            result.test_start, result.test_end,
            result.train_metrics.get("auc", 0),
            result.test_metrics.get("auc", 0),
            result.dsr,
            result.n_train_labels,
        )

        if test_metrics.get("auc", 0) > best_auc:
            best_auc = test_metrics.get("auc", 0)
            best_model = model

        start += step_bars
        fold += 1

    if not results:
        raise RuntimeError("Walk-Forward: kein einziger Fold konnte trainiert werden. "
                           "Mehr Daten oder kleinere train_bars/test_bars verwenden.")

    if best_model is None:
        best_model = results[-1].model

    pbo = probability_of_backtest_overfitting(oos_sharpes)
    logger.info(
        "Walk-Forward abgeschlossen: %d Folds | bestes Test-AUC=%.3f | PBO=%.2f",
        fold, best_auc, pbo,
    )
    return best_model, results
