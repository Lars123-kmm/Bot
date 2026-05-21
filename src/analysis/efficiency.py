"""Market efficiency tracker — continuous statistical monitoring.

Four complementary tests measure whether the market currently shows
exploitable patterns (inefficient) or behaves like a random walk (efficient).

    Score 0.0 = highly inefficient → patterns exist → bot has EDGE
    Score 1.0 = highly efficient   → random walk   → no statistical edge

Tests:
  1. Variance Ratio (Lo-MacKinlay 1988) — deviation from RW at q=2,4,8,16
  2. Ljung-Box autocorrelation — serial correlation in returns
  3. Approximate Entropy — predictability / complexity of the return series
  4. Hurst Exponent — long-memory / persistence (reused from regime detector)

All four are stored in SQLite so you can see how efficiency evolves over time.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.regime.detector import _hurst_exponent

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS efficiency_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT    NOT NULL,
    ts         TEXT    NOT NULL,
    window     INTEGER,
    vr_score   REAL,
    lb_score   REAL,
    apen_score REAL,
    hurst_score REAL,
    composite  REAL,
    has_edge   INTEGER,
    saved_at   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_eff_symbol ON efficiency_history(symbol);
CREATE INDEX IF NOT EXISTS idx_eff_ts     ON efficiency_history(ts);
"""

# Keep at most this many rows per symbol (auto-pruned on update)
_MAX_ROWS_PER_SYMBOL = 5_000


class EfficiencyTracker:
    """
    Continuously monitors market efficiency across symbols.

    Usage inside LiveRunner:
        tracker.update(symbol, df["close"])   # called every bar
        score  = tracker.current_score(symbol)
        history = tracker.get_history(symbol, n=100)
    """

    def __init__(
        self,
        path: str = "data/efficiency_history.db",
        window: int = 252,
        vr_lags: tuple = (2, 4, 8, 16),
        lb_lags: int = 10,
    ):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._window = window
        self._vr_lags = vr_lags
        self._lb_lags = lb_lags
        # In-memory cache of the most recent score per symbol
        self._latest: Dict[str, Dict[str, Any]] = {}
        logger.info("EfficiencyTracker: %s (window=%d)", self._path, window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, symbol: str, prices: pd.Series) -> Dict[str, Any]:
        """
        Compute efficiency metrics from the last `window` prices and persist them.
        Returns the metrics dict (also accessible via current_score).
        """
        metrics = self.compute(prices)
        ts = str(prices.index[-1]) if hasattr(prices.index, "__getitem__") else ""
        self._save(symbol, ts, metrics)
        self._latest[symbol] = metrics
        logger.debug(
            "%s efficiency: composite=%.3f has_edge=%s",
            symbol, metrics["composite"], metrics["has_edge"],
        )
        return metrics

    def compute(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Compute all efficiency metrics without persisting.
        Safe to call directly for one-off analysis.
        """
        log_ret = np.log(prices / prices.shift(1)).dropna().values
        tail = log_ret[-self._window:]

        vr_score  = self._vr_efficiency(tail)
        lb_score  = self._lb_efficiency(tail)
        apen_score = self._apen_efficiency(tail)
        hurst_score = self._hurst_efficiency(prices.values[-self._window:])

        composite = round(
            0.30 * vr_score +
            0.30 * lb_score +
            0.20 * apen_score +
            0.20 * hurst_score,
            4,
        )

        return {
            "vr_score":    round(vr_score, 4),
            "lb_score":    round(lb_score, 4),
            "apen_score":  round(apen_score, 4),
            "hurst_score": round(hurst_score, 4),
            "composite":   composite,
            # Edge exists when market is measurably inefficient
            "has_edge": bool(composite < 0.60),
            "interpretation": _interpret(composite),
        }

    def current_score(self, symbol: str) -> Optional[float]:
        """Most recent composite efficiency score for a symbol (None if not yet computed)."""
        m = self._latest.get(symbol)
        return m["composite"] if m else None

    def has_edge(self, symbol: str) -> bool:
        """True if the market currently shows exploitable inefficiencies."""
        m = self._latest.get(symbol)
        return bool(m["has_edge"]) if m else True  # default: assume edge (don't block)

    def get_history(
        self,
        symbol: Optional[str] = None,
        n: int = 200,
    ) -> pd.DataFrame:
        """Recent efficiency history as a DataFrame (newest last)."""
        query = (
            "SELECT ts, symbol, composite, vr_score, lb_score, apen_score,"
            " hurst_score, has_edge"
            " FROM efficiency_history"
        )
        params: List = []
        if symbol:
            query += " WHERE symbol=?"
            params.append(symbol)
        query += f" ORDER BY id DESC LIMIT {n}"

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame()

        cols = ["ts", "symbol", "composite", "vr_score", "lb_score",
                "apen_score", "hurst_score", "has_edge"]
        df = pd.DataFrame(rows[::-1], columns=cols)
        try:
            df["ts"] = pd.to_datetime(df["ts"])
        except Exception:
            pass
        return df

    def summary(self) -> Dict[str, Any]:
        """Summary of latest scores for all tracked symbols."""
        return {
            sym: {
                "composite": m["composite"],
                "has_edge": m["has_edge"],
                "interpretation": m["interpretation"],
            }
            for sym, m in self._latest.items()
        }

    # ------------------------------------------------------------------
    # Statistical tests (each → [0, 1], 1 = efficient)
    # ------------------------------------------------------------------

    def _vr_efficiency(self, returns: np.ndarray) -> float:
        """Average VR p-value across multiple lags. High p → efficient."""
        pvalues = []
        for q in self._vr_lags:
            _, pval = _variance_ratio(returns, q)
            pvalues.append(pval)
        return float(np.mean(pvalues)) if pvalues else 0.5

    def _lb_efficiency(self, returns: np.ndarray) -> float:
        """Min p-value from Ljung-Box test (conservative). High p → efficient."""
        if len(returns) < self._lb_lags + 5:
            return 0.5
        try:
            result = acorr_ljungbox(returns, lags=self._lb_lags, return_df=True)
            min_pval = float(result["lb_pvalue"].min())
            # Normalize: p > 0.10 → score 1.0; p = 0 → score 0
            return min(min_pval / 0.10, 1.0)
        except Exception:
            return 0.5

    def _apen_efficiency(self, returns: np.ndarray) -> float:
        """Approximate entropy, normalized. Higher ApEn → more random → efficient."""
        apen = _approx_entropy(returns)
        # ApEn ≥ 1.0 → fully random; ApEn = 0 → completely predictable
        return float(min(max(apen / 1.0, 0.0), 1.0))

    def _hurst_efficiency(self, prices: np.ndarray) -> float:
        """
        Hurst exponent H → efficiency score.
        H ≈ 0.5 = random walk = 1.0 (efficient)
        H near 0 or 1 = strong pattern = 0.0 (inefficient)
        """
        h = _hurst_exponent(prices[-self._window:])
        # Map |H - 0.5| ∈ [0, 0.5] to efficiency [1, 0]
        return float(max(1.0 - 2.0 * abs(h - 0.5), 0.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, symbol: str, ts: str, metrics: Dict[str, Any]) -> None:
        try:
            self._conn.execute(
                "INSERT INTO efficiency_history"
                " (symbol, ts, window, vr_score, lb_score, apen_score, hurst_score,"
                "  composite, has_edge)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol, ts, self._window,
                    metrics["vr_score"], metrics["lb_score"],
                    metrics["apen_score"], metrics["hurst_score"],
                    metrics["composite"], int(metrics["has_edge"]),
                ),
            )
            self._conn.commit()
            self._prune(symbol)
        except Exception as exc:
            logger.warning("EfficiencyTracker._save failed: %s", exc)

    def _prune(self, symbol: str) -> None:
        """Keep only the last _MAX_ROWS_PER_SYMBOL rows for each symbol."""
        self._conn.execute(
            "DELETE FROM efficiency_history WHERE symbol=? AND id NOT IN"
            " (SELECT id FROM efficiency_history WHERE symbol=?"
            "  ORDER BY id DESC LIMIT ?)",
            (symbol, symbol, _MAX_ROWS_PER_SYMBOL),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ------------------------------------------------------------------
# Statistical helper functions
# ------------------------------------------------------------------

def _variance_ratio(returns: np.ndarray, q: int) -> tuple[float, float]:
    """
    Lo-MacKinlay (1988) variance ratio test.
    Returns (VR, p-value). p close to 1.0 = random walk = efficient.
    """
    n = len(returns)
    if n < 4 * q:
        return 1.0, 1.0

    mu = returns.mean()

    # 1-period variance
    var1 = ((returns - mu) ** 2).sum() / (n - 1)
    if var1 <= 0:
        return 1.0, 1.0

    # q-period overlapping returns via cumulative sum
    cum = np.concatenate([[0.0], np.cumsum(returns)])
    rq = cum[q:] - cum[:-q]
    nq = len(rq)

    varq = ((rq - q * mu) ** 2).sum() / (nq * q)
    vr = float(varq / var1)

    # Homoskedastic Z-statistic (Lo-MacKinlay Eq. 13)
    phi = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n)
    z = (vr - 1.0) / np.sqrt(phi) if phi > 0 else 0.0
    pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return vr, pvalue


def _approx_entropy(series: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """
    Approximate Entropy (Pincus 1991).
    Higher = more random = more efficient. Typical financial returns: 0.3–1.5.
    Capped at 300 observations for performance (O(N²) per template length).
    """
    x = series[-300:].copy()
    n = len(x)
    if n < 10:
        return 0.5

    r = r_ratio * np.std(x, ddof=1)
    if r <= 0:
        return 0.0

    def _phi(m_len: int) -> float:
        templates = np.lib.stride_tricks.sliding_window_view(x, m_len)  # (n-m+1, m)
        dist = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        cnt = (dist <= r).sum(axis=1).astype(float)
        cnt = np.maximum(cnt, 1.0)  # avoid log(0)
        return float(np.mean(np.log(cnt / (n - m_len + 1))))

    try:
        result = _phi(m) - _phi(m + 1)
        return float(max(result, 0.0))
    except Exception:
        return 0.5


def _interpret(score: float) -> str:
    if score < 0.35:
        return "Stark ineffizient — klarer statistischer Edge"
    if score < 0.55:
        return "Mäßig ineffizient — Edge vorhanden"
    if score < 0.70:
        return "Grenzwertig — Edge unsicher"
    if score < 0.85:
        return "Weitgehend effizient — wenig Edge"
    return "Nahezu effizient — kein messbarer Edge"
