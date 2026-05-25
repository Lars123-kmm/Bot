"""Tests for src/analysis/efficiency.py — Market Efficiency Tracker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.efficiency import (
    EfficiencyTracker,
    _approx_entropy,
    _interpret,
    _variance_ratio,
)


def _price_series(n: int = 400, seed: int = 0, drift: float = 0.0001) -> pd.Series:
    """Synthetic GBM price series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    log_ret = rng.normal(drift, 0.012, n)
    prices = 40_000 * np.exp(np.cumsum(log_ret))
    return pd.Series(prices, index=idx)


def _trending_prices(n: int = 400) -> pd.Series:
    """Strongly trending (persistent) series."""
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    prices = np.linspace(30_000, 60_000, n) + np.random.default_rng(99).normal(0, 200, n)
    return pd.Series(prices, index=idx)


# ── Variance Ratio ────────────────────────────────────────────────────────────

class TestVarianceRatio:
    def test_returns_tuple(self):
        returns = np.random.default_rng(0).normal(0, 0.01, 200)
        vr, pval = _variance_ratio(returns, q=4)
        assert isinstance(vr, float)
        assert 0.0 <= pval <= 1.0

    def test_pure_random_walk_high_pvalue(self):
        """iid returns → VR ≈ 1, high p-value (fail to reject RW)."""
        rng = np.random.default_rng(42)
        pvalues = []
        for _ in range(10):
            r = rng.normal(0, 0.01, 500)
            _, pval = _variance_ratio(r, q=4)
            pvalues.append(pval)
        assert np.mean(pvalues) > 0.05  # on average fail to reject RW

    def test_autocorrelated_series_low_pvalue(self):
        """Strongly autocorrelated returns → reject RW → low p-value."""
        # AR(1) with high coefficient
        n = 500
        r = np.zeros(n)
        rng = np.random.default_rng(7)
        for i in range(1, n):
            r[i] = 0.6 * r[i-1] + rng.normal(0, 0.005)
        _, pval = _variance_ratio(r, q=4)
        assert pval < 0.10

    def test_insufficient_data_returns_efficient(self):
        r = np.array([0.01, -0.01, 0.02])
        vr, pval = _variance_ratio(r, q=4)
        assert vr == 1.0 and pval == 1.0


# ── Approximate Entropy ───────────────────────────────────────────────────────

class TestApproxEntropy:
    def test_random_series_higher_apen(self):
        """iid noise has higher ApEn than deterministic pattern."""
        rng = np.random.default_rng(0)
        random_r = rng.normal(0, 0.01, 200)
        pattern_r = np.tile([0.01, -0.01, 0.02, -0.02], 50)
        assert _approx_entropy(random_r) > _approx_entropy(pattern_r)

    def test_constant_series_low_apen(self):
        """Constant series → perfectly predictable → ApEn ≈ 0."""
        constant = np.ones(100) * 0.001
        assert _approx_entropy(constant) < 0.1

    def test_returns_non_negative(self):
        rng = np.random.default_rng(5)
        r = rng.standard_normal(150)
        assert _approx_entropy(r) >= 0.0


# ── EfficiencyTracker ─────────────────────────────────────────────────────────

class TestEfficiencyTracker:
    def test_compute_returns_all_keys(self, tmp_path):
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"))
        prices = _price_series(400)
        result = tracker.compute(prices)
        for key in ("vr_score", "lb_score", "apen_score", "hurst_score",
                    "composite", "has_edge", "interpretation"):
            assert key in result

    def test_composite_in_unit_interval(self, tmp_path):
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"))
        for seed in range(5):
            prices = _price_series(400, seed=seed)
            result = tracker.compute(prices)
            assert 0.0 <= result["composite"] <= 1.0

    def test_update_persists_to_db(self, tmp_path):
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"))
        prices = _price_series(400)
        tracker.update("BTCUSDT", prices)
        history = tracker.get_history("BTCUSDT")
        assert len(history) == 1

    def test_current_score_none_before_update(self, tmp_path):
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"))
        assert tracker.current_score("BTCUSDT") is None

    def test_current_score_after_update(self, tmp_path):
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"))
        prices = _price_series(400)
        tracker.update("BTCUSDT", prices)
        score = tracker.current_score("BTCUSDT")
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_has_edge_default_true_before_update(self, tmp_path):
        """Before any data: assume edge exists (don't block trades)."""
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"))
        assert tracker.has_edge("BTCUSDT") is True

    def test_trending_series_lower_score_than_random(self, tmp_path):
        """Strongly trending market should score lower (more inefficient)."""
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"), window=200)
        random_prices  = _price_series(400, seed=0)
        trending_prices = _trending_prices(400)
        random_score   = tracker.compute(random_prices)["composite"]
        trending_score = tracker.compute(trending_prices)["composite"]
        assert trending_score < random_score

    def test_history_multiple_updates(self, tmp_path):
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"))
        prices = _price_series(600)
        for i in range(5):
            tracker.update("ETHUSDT", prices.iloc[:300 + i * 50])
        history = tracker.get_history("ETHUSDT")
        assert len(history) == 5

    def test_summary_contains_symbol(self, tmp_path):
        tracker = EfficiencyTracker(path=str(tmp_path / "eff.db"))
        tracker.update("BTCUSDT", _price_series(400))
        s = tracker.summary()
        assert "BTCUSDT" in s
        assert "composite" in s["BTCUSDT"]


# ── Interpret ─────────────────────────────────────────────────────────────────

class TestInterpret:
    def test_low_score_strong_edge(self):
        assert "stark" in _interpret(0.20).lower() or "edge" in _interpret(0.20).lower()

    def test_high_score_no_edge(self):
        assert "effizient" in _interpret(0.90).lower()

    def test_all_thresholds_return_string(self):
        for score in [0.0, 0.34, 0.54, 0.69, 0.84, 1.0]:
            assert isinstance(_interpret(score), str)
