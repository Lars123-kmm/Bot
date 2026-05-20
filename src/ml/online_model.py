from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

logger = logging.getLogger(__name__)


class OnlineMetaFilter:
    """
    Complements the LightGBM batch model with an incremental online learner (river ARF).
    Goal: adaptation between weekly retraining cycles without full refit.

    Blends ARF probability with LightGBM probability.
    ARF only activates after 30+ seen samples to avoid cold-start noise.
    No influence on backtest — online learning is live-mode only.
    """

    def __init__(self, cfg: Dict[str, Any], blend_weight: float = 0.3):
        online_cfg = cfg.get("online_model", {})
        self._blend_weight = float(online_cfg.get("blend_weight", blend_weight))
        self._n_seen = 0
        self._n_correct = 0
        self._model = self._build_arf()

    def _build_arf(self):
        try:
            # SRPClassifier (Streaming Random Patches) is river's adaptive ensemble
            from river.ensemble import SRPClassifier
            from river.tree import HoeffdingAdaptiveTreeClassifier
            return SRPClassifier(
                model=HoeffdingAdaptiveTreeClassifier(seed=42),
                n_models=10,
                seed=42,
            )
        except ImportError:
            logger.warning("river nicht installiert — OnlineMetaFilter deaktiviert (blend_weight=0).")
            self._blend_weight = 0.0
            return None

    def partial_fit(self, x: Dict[str, float], y: int) -> None:
        """
        Update ARF after a trade closes.
        x: feature dict of the entry bar, y: 1=win, 0=loss
        Called in LiveTrader after a position is detected as closed.
        """
        if self._model is None:
            return
        proba_before = self._model.predict_proba_one(x)
        pred_label = 1 if proba_before.get(1, 0.5) >= 0.5 else 0
        self._model.learn_one(x, y)
        self._n_seen += 1
        if pred_label == y:
            self._n_correct += 1

    def predict_proba_online(self, x: Dict[str, float]) -> float:
        """ARF probability estimate for a single bar (streaming)."""
        if self._model is None:
            return 0.5
        proba = self._model.predict_proba_one(x)
        return float(proba.get(1, 0.5))

    def blend(self, lgbm_prob: float, x: Dict[str, float]) -> float:
        """
        Final blend: (1-w)*lgbm_prob + w*arf_prob, clipped to [0.3, 0.95].
        ARF weight applies only after 30 samples to avoid cold-start distortion.
        """
        if self._blend_weight <= 0 or self._n_seen < 30:
            return float(max(0.3, min(0.95, lgbm_prob)))
        arf_prob = self.predict_proba_online(x)
        blended = (1.0 - self._blend_weight) * lgbm_prob + self._blend_weight * arf_prob
        return float(max(0.3, min(0.95, blended)))

    def save(self, path: Path) -> None:
        joblib.dump({
            "model": self._model,
            "blend_weight": self._blend_weight,
            "n_seen": self._n_seen,
            "n_correct": self._n_correct,
        }, path)
        logger.info("OnlineMetaFilter gespeichert: %s", path)

    def load(self, path: Path) -> "OnlineMetaFilter":
        data = joblib.load(path)
        self._model = data["model"]
        self._blend_weight = data["blend_weight"]
        self._n_seen = data["n_seen"]
        self._n_correct = data["n_correct"]
        return self

    def stats(self) -> Dict[str, Any]:
        """Returns sample count, accuracy, and whether ARF is actively blending."""
        accuracy = self._n_correct / self._n_seen if self._n_seen > 0 else 0.0
        return {
            "n_seen": self._n_seen,
            "accuracy": round(accuracy, 4),
            "blend_weight": self._blend_weight,
            "active": self._model is not None and self._n_seen >= 30,
        }
