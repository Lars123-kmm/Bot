"""Tests for src/reporting/equity_report.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.reporting.equity_report import (
    _drawdown_series,
    _equity_curve,
    _pnl_histogram,
    _per_symbol_stats,
    _rolling_winrate,
    _summary_stats,
    generate_report,
)


def _trades(n: int = 20, win_rate: float = 0.6, seed: int = 0):
    import random
    random.seed(seed)
    trades = []
    for i in range(n):
        win = random.random() < win_rate
        pnl = random.uniform(50, 200) if win else -random.uniform(30, 100)
        trades.append({
            "symbol": "BTCUSDT" if i % 2 == 0 else "ETHUSDT",
            "direction": 1,
            "size": 0.01,
            "entry_price": 40_000.0,
            "close_price": 40_000.0 + pnl * 100,
            "pnl": round(pnl, 2),
            "close_time": f"2024-01-{(i % 28) + 1:02d} 12:00",
            "open_time": f"2024-01-{(i % 28) + 1:02d} 10:00",
            "close_reason": "tp" if win else "sl",
        })
    return trades


class TestEquityCurve:
    def test_starts_at_initial_capital(self):
        labels, equity = _equity_curve(_trades(10), 10_000)
        assert equity[0] == 10_000

    def test_length_is_trades_plus_one(self):
        trades = _trades(15)
        labels, equity = _equity_curve(trades, 10_000)
        assert len(equity) == 16  # start + 15 trades

    def test_equity_reflects_pnl(self):
        trades = [{"pnl": 100.0, "close_time": "2024-01-01", "symbol": "BTC"}]
        _, equity = _equity_curve(trades, 10_000)
        assert equity[-1] == pytest.approx(10_100.0)


class TestDrawdownSeries:
    def test_all_positive_equity_no_drawdown(self):
        equity = [10_000, 10_100, 10_200, 10_300]
        dd = _drawdown_series(equity)
        assert all(d == 0.0 for d in dd)

    def test_drawdown_after_peak(self):
        equity = [10_000, 11_000, 10_500]
        dd = _drawdown_series(equity)
        assert dd[-1] < 0.0
        assert abs(dd[-1] - (-500 / 11_000)) < 1e-6


class TestRollingWinrate:
    def test_empty_trades_returns_empty(self):
        labels, rates = _rolling_winrate([], window=20)
        assert labels == [] and rates == []

    def test_rate_between_zero_and_one(self):
        _, rates = _rolling_winrate(_trades(50), window=10)
        assert all(0.0 <= r <= 1.0 for r in rates)


class TestPnlHistogram:
    def test_empty_returns_empty(self):
        assert _pnl_histogram([]) == ([], [])

    def test_bin_counts_sum_to_n_trades(self):
        trades = _trades(30)
        _, counts = _pnl_histogram(trades, n_bins=10)
        assert sum(counts) == 30


class TestPerSymbolStats:
    def test_both_symbols_present(self):
        stats = _per_symbol_stats(_trades(20))
        syms = {s["symbol"] for s in stats}
        assert "BTCUSDT" in syms and "ETHUSDT" in syms

    def test_win_rate_in_valid_range(self):
        stats = _per_symbol_stats(_trades(20))
        for s in stats:
            assert 0.0 <= s["win_rate"] <= 100.0


class TestSummaryStats:
    def test_contains_required_keys(self):
        trades = _trades(20)
        _, equity = _equity_curve(trades, 10_000)
        s = _summary_stats(trades, 10_000, equity)
        for k in ("n_trades", "win_rate", "total_pnl", "sharpe", "max_drawdown_pct"):
            assert k in s

    def test_n_trades_correct(self):
        trades = _trades(15)
        _, equity = _equity_curve(trades, 10_000)
        s = _summary_stats(trades, 10_000, equity)
        assert s["n_trades"] == 15


class TestGenerateReport:
    def test_creates_html_file(self, tmp_path):
        out = str(tmp_path / "report.html")
        generate_report(_trades(10), 10_000, output_path=out)
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "<canvas" in content
        assert "Chart" in content

    def test_empty_trades_does_not_crash(self, tmp_path):
        out = str(tmp_path / "empty.html")
        generate_report([], 10_000, output_path=out)
        assert Path(out).exists()

    def test_with_feature_importances(self, tmp_path):
        from src.features.ml_features import ML_FEATURE_COLS
        fi = {col: float(i) for i, col in enumerate(ML_FEATURE_COLS)}
        out = str(tmp_path / "fi.html")
        generate_report(_trades(10), 10_000, output_path=out, feature_importances=fi)
        content = Path(out).read_text()
        assert ML_FEATURE_COLS[0] in content or "feat" in content.lower()
