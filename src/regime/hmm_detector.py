from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    2-state Gaussian HMM on (log_return, realized_vol).
    State 0 = low-vol / trend-friendly
    State 1 = high-vol / ranging

    Used as a SOFT filter (probabilities as features), not a hard switch,
    to reduce false-positive rate.
    """

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._model: Optional[GaussianHMM] = None
        self._trend_state: int = 0

    def _build_features(self, df: pd.DataFrame, window: int = 20) -> np.ndarray:
        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
        realized_vol = log_ret.rolling(window).std().fillna(0.0) * np.sqrt(24)
        X = np.column_stack([log_ret.values, realized_vol.values])
        return X

    def fit(self, df: pd.DataFrame, window: int = 20) -> "HMMRegimeDetector":
        """
        Trains HMM on log_return and realized_vol.
        State assignment: lower-vol state → State 0 (TREND-friendly).
        """
        X = self._build_features(df, window)
        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
        )
        model.fit(X)
        self._model = model

        # Identify which state is "trend-friendly" (lower mean volatility)
        mean_vols = [model.means_[s, 1] for s in range(self.n_components)]
        self._trend_state = int(np.argmin(mean_vols))

        logger.info(
            "HMMRegimeDetector trainiert: trend_state=%d, means=%s",
            self._trend_state,
            [f"{m[1]:.4f}" for m in model.means_],
        )
        return self

    def predict(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Returns hard state classification (0=trend-friendly, 1=volatile)."""
        self._require_fitted()
        X = self._build_features(df, window)
        raw_states = self._model.predict(X)
        # Map to 0=trend, 1=volatile regardless of HMM internal state numbering
        mapped = np.where(raw_states == self._trend_state, 0, 1)
        return pd.Series(mapped, index=df.index, dtype=int, name="hmm_state")

    def predict_proba(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Returns state probabilities:
        Columns: hmm_prob_trend, hmm_prob_volatile
        """
        self._require_fitted()
        X = self._build_features(df, window)
        _, posteriors = self._model.score_samples(X)

        trend_col = posteriors[:, self._trend_state]
        volatile_col = 1.0 - trend_col

        return pd.DataFrame(
            {"hmm_prob_trend": trend_col, "hmm_prob_volatile": volatile_col},
            index=df.index,
        )

    def save(self, path: Path) -> None:
        joblib.dump({"model": self._model, "trend_state": self._trend_state}, path)
        logger.info("HMMRegimeDetector gespeichert: %s", path)

    def load(self, path: Path) -> "HMMRegimeDetector":
        data = joblib.load(path)
        self._model = data["model"]
        self._trend_state = data["trend_state"]
        return self

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("HMMRegimeDetector wurde noch nicht trainiert. fit() zuerst aufrufen.")
