"""Live bar store — persists every processed bar to SQLite.

After each tick, the full processed DataFrame (OHLCV + all indicators +
ML features + signal + regime + efficiency) is appended here.
Queryable with Python/Jupyter at any time while the bot runs.

Usage:
    import sqlite3, pandas as pd
    conn = sqlite3.connect("data/bars.db")
    df = pd.read_sql("SELECT * FROM bars WHERE symbol='BTCUSDT' ORDER BY ts DESC LIMIT 500", conn)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT    NOT NULL,
    ts        TEXT    NOT NULL,
    open      REAL, high REAL, low REAL, close REAL, volume REAL,
    signal    INTEGER,
    regime    INTEGER,
    atr       REAL,
    rsi       REAL,
    adx       REAL,
    ema_fast  REAL,
    ema_slow  REAL,
    efficiency_score REAL,
    features  TEXT,
    saved_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol ON bars(symbol);
CREATE INDEX IF NOT EXISTS idx_bars_ts     ON bars(ts);
"""

_MAX_ROWS_PER_SYMBOL = 10_000


class BarStore:
    """Append-only store for processed bars. Auto-prunes to _MAX_ROWS_PER_SYMBOL."""

    def __init__(self, path: str = "data/bars.db"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("BarStore: %s", self._path)

    def append(
        self,
        symbol: str,
        df: pd.DataFrame,
        efficiency_score: Optional[float] = None,
        n_latest: int = 1,
    ) -> None:
        """Persist the last n_latest rows of df to the database."""
        rows = df.iloc[-n_latest:]
        for ts, row in rows.iterrows():
            features = {
                col: float(row[col])
                for col in df.columns
                if col not in ("open","high","low","close","volume","signal","regime",
                               "atr","rsi","adx","ema_fast","ema_slow")
                and pd.api.types.is_numeric_dtype(type(row[col]))
            }
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO bars"
                    " (symbol,ts,open,high,low,close,volume,signal,regime,"
                    "  atr,rsi,adx,ema_fast,ema_slow,efficiency_score,features)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        symbol, str(ts),
                        _f(row, "open"), _f(row, "high"), _f(row, "low"),
                        _f(row, "close"), _f(row, "volume"),
                        int(row["signal"]) if "signal" in row.index else None,
                        int(row.get("regime", 0)),
                        _f(row, "atr"), _f(row, "rsi"), _f(row, "adx"),
                        _f(row, "ema_fast"), _f(row, "ema_slow"),
                        round(efficiency_score, 4) if efficiency_score is not None else None,
                        json.dumps({k: round(v, 6) for k, v in features.items() if v == v}),
                    ),
                )
            except Exception as exc:
                logger.debug("BarStore.append failed for %s: %s", symbol, exc)
        self._conn.commit()
        self._prune(symbol)

    def query(
        self,
        symbol: Optional[str] = None,
        n: int = 500,
        since: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return recent bars as a DataFrame."""
        q = "SELECT * FROM bars"
        params: List = []
        filters = []
        if symbol:
            filters.append("symbol=?")
            params.append(symbol)
        if since:
            filters.append("ts >= ?")
            params.append(since)
        if filters:
            q += " WHERE " + " AND ".join(filters)
        q += f" ORDER BY ts DESC LIMIT {n}"
        df = pd.read_sql_query(q, self._conn, params=params)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.sort_values("ts").reset_index(drop=True)
        return df

    def symbols(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT symbol FROM bars ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in rows]

    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT symbol, COUNT(*) as n, MAX(ts) as latest FROM bars GROUP BY symbol"
        ).fetchall()
        return {r[0]: {"n_bars": r[1], "latest": r[2]} for r in rows}

    def _prune(self, symbol: str) -> None:
        self._conn.execute(
            "DELETE FROM bars WHERE symbol=? AND id NOT IN"
            " (SELECT id FROM bars WHERE symbol=? ORDER BY id DESC LIMIT ?)",
            (symbol, symbol, _MAX_ROWS_PER_SYMBOL),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _f(row: pd.Series, col: str) -> Optional[float]:
    if col not in row.index:
        return None
    v = row[col]
    try:
        f = float(v)
        return f if f == f else None
    except Exception:
        return None
