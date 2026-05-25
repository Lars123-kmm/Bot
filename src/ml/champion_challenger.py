from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.ml.meta_model import MetaModel

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    model: MetaModel
    metadata: Dict[str, Any]
    metrics: Dict[str, float]
    n_wins: int = 0


class ChampionChallenger:
    """
    Reine Entscheidungs-Engine für Champion-Challenger-Modell-Promotion.

    Ein Challenger wird erst zum Champion befördert, wenn er den Champion in
    n_windows_needed aufeinanderfolgenden Out-of-Sample-Vergleichen schlägt.

    Die Persistenz (welche Version ist Champion, der Win-Streak) liegt beim
    Aufrufer — diese Klasse entscheidet nur. carry_wins seedet den Streak aus
    vorherigen Trainingsläufen.
    """

    def __init__(self, n_windows_needed: int = 3, metric: str = "auc"):
        self.n_windows_needed = n_windows_needed
        self.metric = metric
        self._champion: Optional[ModelCandidate] = None
        self._challenger: Optional[ModelCandidate] = None

    def set_champion(
        self,
        model: MetaModel,
        metadata: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Setzt das aktuelle Champion-Modell. metrics default = train_metrics_."""
        self._champion = ModelCandidate(
            model=model,
            metadata=metadata,
            metrics=metrics if metrics is not None else model.train_metrics_,
        )
        logger.info("Champion gesetzt: %s=%.4f",
                    self.metric, self._champion.metrics.get(self.metric, 0.0))

    def propose_challenger(
        self,
        model: MetaModel,
        metadata: Dict[str, Any],
        oos_metrics: Dict[str, float],
        carry_wins: int = 0,
    ) -> bool:
        """
        Vergleicht einen Challenger mit dem Champion auf einem OOS-Window.
        carry_wins seedet den Streak aus vorherigen Läufen (Persistenz).
        Gibt True zurück, wenn der Challenger zum Champion befördert wird.
        """
        if self._champion is None:
            self.set_champion(model, metadata, metrics=oos_metrics)
            return True

        champ_score = self._champion.metrics.get(self.metric, 0.0)
        chall_score = oos_metrics.get(self.metric, 0.0)
        self._challenger = ModelCandidate(
            model=model, metadata=metadata, metrics=oos_metrics, n_wins=carry_wins,
        )

        if chall_score > champ_score:
            self._challenger.n_wins += 1
            logger.info(
                "Challenger gewinnt: %s=%.4f vs Champion=%.4f (Streak %d/%d)",
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
            self._champion = self._challenger
            self._challenger = None
            return True

        return False

    @property
    def streak(self) -> int:
        """Aktueller Consecutive-Win-Streak (0 direkt nach Promotion oder Niederlage)."""
        return self._challenger.n_wins if self._challenger else 0

    def get_active_model(self) -> MetaModel:
        """Gibt den Champion zurück (der Challenger ist erst nach Promotion aktiv)."""
        if self._champion is None:
            raise RuntimeError("Kein Champion-Modell gesetzt. set_champion() zuerst aufrufen.")
        return self._champion.model

    def status(self) -> Dict[str, Any]:
        """Aktueller Champion/Challenger-Status mit Metriken."""
        return {
            "champion_metric": self._champion.metrics.get(self.metric) if self._champion else None,
            "challenger_streak": self.streak,
            "wins_needed": self.n_windows_needed,
            "metric": self.metric,
        }
