from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.ml.meta_model import MetaModel
from src.ml.model_store import ModelStore

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    model: MetaModel
    metadata: Dict[str, Any]
    metrics: Dict[str, float]
    n_wins: int = 0


class ChampionChallenger:
    """
    Keeps the champion model frozen while a challenger trains in the background.
    Promotion only when challenger dominates for N consecutive OOS windows.
    """

    def __init__(
        self,
        model_store: ModelStore,
        n_windows_needed: int = 3,
        metric: str = "sharpe",
    ):
        self._store = model_store
        self.n_windows_needed = n_windows_needed
        self.metric = metric
        self._champion: Optional[ModelCandidate] = None
        self._challenger: Optional[ModelCandidate] = None

    def set_champion(self, model: MetaModel, metadata: Dict[str, Any]) -> None:
        """Sets the initial champion model."""
        self._champion = ModelCandidate(
            model=model,
            metadata=metadata,
            metrics=model.train_metrics_,
        )
        logger.info("Champion gesetzt: AUC=%.3f", model.train_metrics_.get("auc", 0))

    def propose_challenger(
        self,
        model: MetaModel,
        metadata: Dict[str, Any],
        oos_metrics: Dict[str, float],
    ) -> bool:
        """
        Compares challenger to champion on current OOS window.
        Returns True if challenger is promoted to champion.
        """
        if self._champion is None:
            self.set_champion(model, metadata)
            return False

        champ_score = self._champion.metrics.get(self.metric, 0.0)
        chall_score = oos_metrics.get(self.metric, 0.0)

        if self._challenger is None or id(self._challenger.model) != id(model):
            self._challenger = ModelCandidate(
                model=model, metadata=metadata, metrics=oos_metrics
            )

        if chall_score > champ_score:
            self._challenger.n_wins += 1
            logger.info(
                "Challenger gewinnt OOS-Window: %s=%.4f vs Champion=%.4f (wins=%d/%d)",
                self.metric, chall_score, champ_score,
                self._challenger.n_wins, self.n_windows_needed,
            )
        else:
            self._challenger.n_wins = 0
            logger.info(
                "Champion hält: %s=%.4f vs Challenger=%.4f",
                self.metric, champ_score, chall_score,
            )

        if self._challenger.n_wins >= self.n_windows_needed:
            logger.info(
                "PROMOTION: Challenger wird Champion (%s=%.4f nach %d Wins).",
                self.metric, chall_score, self._challenger.n_wins,
            )
            self._champion = ModelCandidate(
                model=model, metadata=metadata, metrics=oos_metrics, n_wins=0
            )
            self._challenger = None
            self._store.save(model, {**metadata, "promoted_by": "champion_challenger"})
            return True

        return False

    def get_active_model(self) -> MetaModel:
        """Returns champion as long as challenger hasn't reached n_windows_needed wins."""
        if self._champion is None:
            raise RuntimeError("Kein Champion-Modell gesetzt. set_champion() zuerst aufrufen.")
        return self._champion.model

    def status(self) -> Dict[str, Any]:
        """Returns current champion/challenger status with metrics."""
        return {
            "champion_auc": self._champion.metrics.get("auc", 0) if self._champion else None,
            "champion_metric": self._champion.metrics.get(self.metric, 0) if self._champion else None,
            "challenger_wins": self._challenger.n_wins if self._challenger else 0,
            "wins_needed": self.n_windows_needed,
            "metric": self.metric,
        }
