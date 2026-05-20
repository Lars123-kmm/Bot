from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features.ml_features import ML_FEATURE_COLS

logger = logging.getLogger(__name__)


class MetaModel:
    """
    Zwei-Stufen-Entscheidungsmodell (Meta-Labeling):

    Schicht 1: EMA-Crossover bestimmt die Richtung (Long/Short)
    Schicht 2: MetaModel bewertet ob der Trade tatsächlich profitabel wird

    Modell: LightGBM + Platt-Kalibrierung → echte Wahrscheinlichkeiten [0, 1]
    """

    def __init__(self, cfg: Dict[str, Any]):
        ml_cfg = cfg.get("ml", {})
        self.threshold = float(ml_cfg.get("min_probability", 0.55))
        self._cfg = cfg
        self._model: Optional[CalibratedClassifierCV] = None
        self._scaler = StandardScaler()
        self.feature_importances_: Optional[Dict[str, float]] = None
        self.train_metrics_: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MetaModel":
        """
        Trainiert LightGBM auf Triple-Barrier-Labels und kalibriert die Ausgaben.

        X: Feature-Matrix (aus build_ml_features(), Spalten = ML_FEATURE_COLS)
        y: Binäre Labels (0/1, aus triple_barrier_labels())
        """
        X_feat = self._select_features(X)
        X_scaled = self._scaler.fit_transform(X_feat)

        ml_cfg = self._cfg.get("ml", {})
        base = LGBMClassifier(
            num_leaves=int(ml_cfg.get("lgbm_num_leaves", 50)),
            learning_rate=float(ml_cfg.get("lgbm_learning_rate", 0.05)),
            n_estimators=int(ml_cfg.get("lgbm_n_estimators", 200)),
            min_child_samples=int(ml_cfg.get("lgbm_min_child_samples", 20)),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        # Kalibrierung: Platt Scaling → kalibrierte Wahrscheinlichkeiten
        self._model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self._model.fit(X_scaled, y.values)

        # Feature-Importance aus dem Base-Modell extrahieren
        if hasattr(self._model.calibrated_classifiers_[0].estimator, "feature_importances_"):
            importances = np.mean([
                c.estimator.feature_importances_
                for c in self._model.calibrated_classifiers_
            ], axis=0)
            self.feature_importances_ = dict(zip(ML_FEATURE_COLS, importances.tolist()))

        # Trainings-Metriken
        probas = self._model.predict_proba(X_scaled)[:, 1]
        self.train_metrics_ = {
            "auc": round(float(roc_auc_score(y, probas)), 4),
            "brier_score": round(float(brier_score_loss(y, probas)), 4),
            "n_samples": len(y),
            "positive_rate": round(float(y.mean()), 4),
        }
        logger.info("MetaModel trainiert: AUC=%.3f, Brier=%.3f, n=%d",
                    self.train_metrics_["auc"],
                    self.train_metrics_["brier_score"],
                    self.train_metrics_["n_samples"])
        return self

    # ------------------------------------------------------------------
    # Inferenz
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Kalibrierte Erfolgswahrscheinlichkeit für jeden Bar.
        Returns: 1D-Array mit Werten in [0, 1].
        """
        self._require_fitted()
        X_feat = self._select_features(X)
        X_scaled = self._scaler.transform(X_feat)
        return self._model.predict_proba(X_scaled)[:, 1]

    def is_tradeable(self, X_row: pd.DataFrame) -> Tuple[bool, float]:
        """
        Gibt (freigegeben, wahrscheinlichkeit) für eine einzelne Bar zurück.
        Freigegeben wenn proba >= self.threshold.
        """
        proba = float(self.predict_proba(X_row)[0])
        return proba >= self.threshold, proba

    def top_features(self, n: int = 10) -> Dict[str, float]:
        """Gibt die n wichtigsten Features sortiert zurück."""
        if self.feature_importances_ is None:
            return {}
        sorted_items = sorted(self.feature_importances_.items(),
                               key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:n])

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        joblib.dump({
            "model": self._model,
            "scaler": self._scaler,
            "threshold": self.threshold,
            "feature_importances": self.feature_importances_,
            "train_metrics": self.train_metrics_,
        }, path)
        logger.info("MetaModel gespeichert: %s", path)

    def load(self, path: Path) -> "MetaModel":
        data = joblib.load(path)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self.threshold = data.get("threshold", self.threshold)
        self.feature_importances_ = data.get("feature_importances")
        self.train_metrics_ = data.get("train_metrics", {})
        logger.info("MetaModel geladen: %s (AUC=%.3f)", path,
                    self.train_metrics_.get("auc", 0))
        return self

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in ML_FEATURE_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"Fehlende ML-Feature-Spalten: {missing}")
        return X[ML_FEATURE_COLS].copy()

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("MetaModel wurde noch nicht trainiert. fit() zuerst aufrufen.")
