"""Feature importance tracker with automatic dropout over retrain cycles.

After each retrain, LightGBM feature importances are persisted to SQLite.
Features that consistently rank in the bottom percentile for N consecutive
cycles are silently dropped from the active feature set.
If a dropped feature recovers (importance rises above the threshold in the
next M cycles), it is automatically reinstated.

This creates exponential improvement: the model focuses resources on proven
signals, while weak/noisy features stop polluting the decision boundary.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.features.ml_features import ML_FEATURE_COLS

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_cycles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle       INTEGER NOT NULL,
    feature     TEXT    NOT NULL,
    importance  REAL    NOT NULL,
    recorded_at TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_fc_cycle ON feature_cycles(cycle);
"""


class AdaptiveFeatureSelector:
    """
    Tracks feature importances across retrain cycles and auto-drops weak features.

    Args:
        path:               SQLite file path.
        drop_percentile:    Features below this quantile in each cycle count as "weak".
        min_cycles_to_drop: Feature must be weak for this many consecutive cycles to get dropped.
        recovery_cycles:    After dropping, feature is reinstated if it rises above the
                            percentile threshold for this many consecutive cycles.
    """

    def __init__(
        self,
        path: str = "data/feature_importance.db",
        drop_percentile: float = 0.10,
        min_cycles_to_drop: int = 3,
        recovery_cycles: int = 2,
    ):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._drop_pct = drop_percentile
        self._min_cycles = min_cycles_to_drop
        self._recovery = recovery_cycles
        self._dropped: set = set()
        self._current_cycle: int = self._max_cycle()
        # Restore dropped state from existing history
        if self._current_cycle >= self._min_cycles:
            self._recompute_dropped()
        logger.info(
            "AdaptiveFeatureSelector: %s — cycle=%d, %d/%d features active",
            self._path, self._current_cycle,
            len(self.get_active_features()), len(ML_FEATURE_COLS),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, model: object) -> None:
        """Record feature importances from a freshly trained MetaModel."""
        importances: Optional[Dict] = getattr(model, "feature_importances_", None)
        if not importances:
            logger.warning("AdaptiveFeatureSelector.update: model has no feature_importances_")
            return

        self._current_cycle += 1
        rows = [
            (self._current_cycle, feat, float(imp))
            for feat, imp in importances.items()
            if feat in ML_FEATURE_COLS
        ]
        self._conn.executemany(
            "INSERT INTO feature_cycles (cycle, feature, importance) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._recompute_dropped()
        active = len(self.get_active_features())
        logger.info(
            "Cycle %d: %d/%d features active (dropped: %s)",
            self._current_cycle, active, len(ML_FEATURE_COLS),
            sorted(self._dropped) if self._dropped else "none",
        )

    def get_active_features(self) -> List[str]:
        """Return ML_FEATURE_COLS minus persistently low-importance features."""
        return [f for f in ML_FEATURE_COLS if f not in self._dropped]

    def dropped_features(self) -> List[str]:
        return sorted(self._dropped)

    def report(self) -> pd.DataFrame:
        """Feature × cycle importance matrix (useful for dashboards/reports)."""
        rows = self._conn.execute(
            "SELECT cycle, feature, importance FROM feature_cycles ORDER BY cycle, feature"
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["cycle", "feature", "importance"])
        df = pd.DataFrame(rows, columns=["cycle", "feature", "importance"])
        try:
            return df.pivot(index="feature", columns="cycle", values="importance")
        except Exception:
            return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute_dropped(self) -> None:
        if self._current_cycle < self._min_cycles:
            return

        start = self._current_cycle - self._min_cycles + 1
        rows = self._conn.execute(
            "SELECT cycle, feature, importance FROM feature_cycles WHERE cycle >= ?",
            (start,),
        ).fetchall()
        if not rows:
            return

        df = pd.DataFrame(rows, columns=["cycle", "feature", "importance"])
        if df["cycle"].nunique() < self._min_cycles:
            return

        # Per-cycle bottom-percentile threshold
        thresholds = df.groupby("cycle")["importance"].quantile(self._drop_pct)

        new_dropped: set = set()
        for feat in ML_FEATURE_COLS:
            feat_rows = df[df["feature"] == feat]
            if len(feat_rows) < self._min_cycles:
                continue
            n_below = sum(
                1 for _, r in feat_rows.iterrows()
                if r["importance"] <= thresholds.get(r["cycle"], 0.0)
            )
            if n_below >= self._min_cycles:
                new_dropped.add(feat)

        # Recovery: dropped feature rose above threshold in recent cycles
        if self._dropped and self._current_cycle >= self._recovery:
            rec_start = self._current_cycle - self._recovery + 1
            rec_rows = self._conn.execute(
                "SELECT feature, importance FROM feature_cycles WHERE cycle >= ?",
                (rec_start,),
            ).fetchall()
            if rec_rows:
                rec_df = pd.DataFrame(rec_rows, columns=["feature", "importance"])
                rec_threshold = rec_df["importance"].quantile(self._drop_pct)
                reinstated = set()
                for feat in self._dropped:
                    feat_recent = rec_df[rec_df["feature"] == feat]
                    # Reinstate if above threshold in ALL recent cycles
                    if (
                        len(feat_recent) >= self._recovery
                        and (feat_recent["importance"] > rec_threshold).all()
                    ):
                        reinstated.add(feat)
                if reinstated:
                    logger.info("Reinstating features: %s", sorted(reinstated))
                new_dropped -= reinstated

        newly_dropped = new_dropped - self._dropped
        if newly_dropped:
            logger.warning(
                "Dropping %d persistently weak features: %s", len(newly_dropped), sorted(newly_dropped)
            )
        self._dropped = new_dropped

    def _max_cycle(self) -> int:
        row = self._conn.execute("SELECT MAX(cycle) FROM feature_cycles").fetchone()
        return int(row[0]) if row[0] is not None else 0

    def close(self) -> None:
        self._conn.close()
