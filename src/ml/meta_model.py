from __future__ import annotations

import logging
import warnings
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
        # Active feature list — may be a subset of ML_FEATURE_COLS when
        # AdaptiveFeatureSelector has dropped low-importance features.
        self.active_features_: list = list(
            ml_cfg.get("active_features", None) or ML_FEATURE_COLS
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "MetaModel":
        """
        Trainiert LightGBM auf Triple-Barrier-Labels und kalibriert die Ausgaben.

        X: Feature-Matrix (aus build_ml_features(), Spalten = ML_FEATURE_COLS)
        y: Binäre Labels (0/1, aus triple_barrier_labels())
        sample_weight: Optional López-de-Prado Sample-Uniqueness-Gewichte
        """
        X_feat = self._select_features(X)
        # np.asarray ensures numpy (not DataFrame) so LightGBM doesn't set feature_names_in_
        X_scaled = np.asarray(self._scaler.fit_transform(X_feat))

        sw = None
        if sample_weight is not None:
            sw = sample_weight.reindex(X.index).fillna(1.0).values

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

        # Isotonische Kalibrierung — Fallback-Kette für kleine Datensätze
        # CalibratedClassifierCV.fit löst intern sklearn-Warnings aus (numpy vs. DataFrame
        # in CV-Folds). Nur kosmetisch, kein Einfluss auf Ergebnisse.
        min_class = int(np.bincount(y.values.astype(int)).min())
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names",
                category=UserWarning,
            )
            if min_class >= 3:
                self._model = CalibratedClassifierCV(base, method="isotonic", cv=3)
                self._model.fit(X_scaled, y.values, sample_weight=sw)
            elif min_class >= 2:
                self._model = CalibratedClassifierCV(base, method="sigmoid", cv=2)
                self._model.fit(X_scaled, y.values, sample_weight=sw)
            else:
                # Zu wenige Samples für CV-Kalibrierung → unkalibrierter LightGBM
                base.fit(X_scaled, y.values, sample_weight=sw)
                self._model = base

        # Feature-Importance aus dem Base-Modell extrahieren
        cols = self.active_features_
        if hasattr(self._model, "calibrated_classifiers_"):
            importances = np.mean([
                c.estimator.feature_importances_
                for c in self._model.calibrated_classifiers_
                if hasattr(c.estimator, "feature_importances_")
            ], axis=0)
            self.feature_importances_ = dict(zip(cols, importances.tolist()))
        elif hasattr(self._model, "feature_importances_"):
            self.feature_importances_ = dict(
                zip(cols, self._model.feature_importances_.tolist())
            )

        # Trainings-Metriken
        probas = self._model.predict_proba(X_scaled)[:, 1]
        auc = float(roc_auc_score(y, probas)) if y.nunique() > 1 else 0.5
        self.train_metrics_ = {
            "auc": round(auc, 4),
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
        X_scaled = np.asarray(self._scaler.transform(X_feat))
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
            "active_features": self.active_features_,
        }, path)
        logger.info("MetaModel gespeichert: %s", path)

    def load(self, path: Path) -> "MetaModel":
        data = joblib.load(path)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self.threshold = data.get("threshold", self.threshold)
        self.feature_importances_ = data.get("feature_importances")
        self.train_metrics_ = data.get("train_metrics", {})
        self.active_features_ = data.get("active_features", list(ML_FEATURE_COLS))
        logger.info("MetaModel geladen: %s (AUC=%.3f, features=%d)", path,
                    self.train_metrics_.get("auc", 0), len(self.active_features_))
        return self

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.active_features_
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Fehlende ML-Feature-Spalten: {missing}")
        return X[cols].copy()

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("MetaModel wurde noch nicht trainiert. fit() zuerst aufrufen.")
