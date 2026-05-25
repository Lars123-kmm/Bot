"""Persistent SQLite store for bar features + trade outcomes.

Every opened trade gets its ML features logged at entry time.
When the trade closes, the actual outcome (win/loss) is written back.
This creates a self-labeling dataset that grows indefinitely across sessions
and serves as ground-truth for autonomous hyperparameter tuning.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.features.ml_features import ML_FEATURE_COLS

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_records (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket     INTEGER UNIQUE,
    symbol     TEXT    NOT NULL,
    bar_time   TEXT    NOT NULL,
    direction  INTEGER NOT NULL,
    features   TEXT    NOT NULL,
    outcome    INTEGER,
    pnl        REAL,
    created_at TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_outcome ON trade_records(outcome);
CREATE INDEX IF NOT EXISTS idx_symbol  ON trade_records(symbol);
"""


class TradeDataStore:
    """
    Grows across bot sessions: every trade entry → features + eventual outcome.

    After ~50–200 completed trades the dataset becomes useful for HPO.
    After 500+ trades it provides a strong real-market signal for retraining.
    """

    def __init__(self, path: str = "data/trade_history.db"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        total = self._conn.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
        logger.info("TradeDataStore: %s (%d trades on disk)", self._path, total)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_entry(
        self,
        symbol: str,
        bar_time: object,
        direction: int,
        features: Dict[str, float],
        ticket: int,
    ) -> None:
        """Record features at trade entry (outcome filled in later)."""
        feat_json = json.dumps({k: _safe_float(v) for k, v in features.items()})
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO trade_records"
                " (ticket, symbol, bar_time, direction, features)"
                " VALUES (?, ?, ?, ?, ?)",
                (ticket, symbol, str(bar_time), direction, feat_json),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("DataStore.log_entry failed: %s", exc)

    def update_outcome(self, ticket: int, outcome: int, pnl: float) -> None:
        """Mark a pending trade as completed with its realized outcome."""
        try:
            self._conn.execute(
                "UPDATE trade_records SET outcome=?, pnl=? WHERE ticket=?",
                (int(outcome), float(pnl), ticket),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("DataStore.update_outcome failed: %s", exc)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_training_data(
        self,
        symbol: Optional[str] = None,
        min_samples: int = 50,
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Returns (X, y) for all completed trades, or None if not enough data yet.

        X columns = ML_FEATURE_COLS (missing features filled with 0).
        y = 1 (win) / 0 (loss).
        """
        query = (
            "SELECT features, outcome FROM trade_records"
            " WHERE outcome IS NOT NULL"
        )
        params: List = []
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
        query += " ORDER BY bar_time ASC"

        rows = self._conn.execute(query, params).fetchall()
        if len(rows) < min_samples:
            logger.info(
                "DataStore: %d completed trades (need %d for HPO)",
                len(rows), min_samples,
            )
            return None

        records, labels = [], []
        for feat_json, outcome in rows:
            try:
                feat = json.loads(feat_json)
                records.append({col: feat.get(col, 0.0) for col in ML_FEATURE_COLS})
                labels.append(int(outcome))
            except Exception:
                continue

        if not records:
            return None

        X = pd.DataFrame(records, columns=ML_FEATURE_COLS).fillna(0.0)
        y = pd.Series(labels, name="label", dtype=int)
        logger.info("DataStore: %d real-trade samples loaded for HPO", len(X))
        return X, y

    def stats(self) -> Dict[str, object]:
        row = self._conn.execute(
            "SELECT"
            " COUNT(*),"
            " SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END)"
            " FROM trade_records"
        ).fetchone()
        total, wins, losses, pending = (r or 0 for r in row)
        completed = wins + losses
        return {
            "total_trades": int(total),
            "wins": int(wins),
            "losses": int(losses),
            "pending": int(pending),
            "win_rate": round(wins / max(completed, 1), 3),
        }

    def close(self) -> None:
        self._conn.close()


def _safe_float(v: object) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if f == f else 0.0  # NaN check
    except Exception:
        return 0.0
