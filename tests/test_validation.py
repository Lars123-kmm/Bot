"""Tests for src/ml/validation.py — DSR, TRL, PBO."""
from __future__ import annotations

import math

import pytest

from src.ml.validation import (
    deflated_sharpe_ratio,
    min_track_record_length,
    probability_of_backtest_overfitting,
)


class TestDeflatedSharpeRatio:
    def test_single_trial_is_psr(self):
        """With n_trials=1, DSR == PSR (probability SR > 0)."""
        dsr = deflated_sharpe_ratio(sr=1.0, t=252, n_trials=1)
        assert 0.0 < dsr < 1.0

    def test_positive_sr_above_half(self):
        dsr = deflated_sharpe_ratio(sr=1.5, t=500, n_trials=1)
        assert dsr > 0.5

    def test_negative_sr_below_half(self):
        dsr = deflated_sharpe_ratio(sr=-0.5, t=252, n_trials=1)
        assert dsr < 0.5

    def test_more_trials_lowers_dsr(self):
        """More independent trials → higher expected SR* → lower DSR for same SR."""
        dsr_1  = deflated_sharpe_ratio(sr=1.0, t=252, n_trials=1)
        dsr_50 = deflated_sharpe_ratio(sr=1.0, t=252, n_trials=50)
        assert dsr_50 < dsr_1

    def test_larger_t_raises_dsr(self):
        """Longer track record → more confident → higher DSR for same SR."""
        dsr_short = deflated_sharpe_ratio(sr=0.8, t=50,   n_trials=5)
        dsr_long  = deflated_sharpe_ratio(sr=0.8, t=1000, n_trials=5)
        assert dsr_long > dsr_short

    def test_returns_zero_for_t_leq_one(self):
        assert deflated_sharpe_ratio(sr=2.0, t=1) == 0.0

    def test_output_in_unit_interval(self):
        for sr in [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            dsr = deflated_sharpe_ratio(sr=sr, t=200, n_trials=10)
            assert 0.0 <= dsr <= 1.0


class TestMinTrackRecordLength:
    def test_positive_sr_gives_finite_value(self):
        trl = min_track_record_length(sr=1.0)
        assert math.isfinite(trl)
        assert trl > 0

    def test_zero_sr_gives_inf(self):
        trl = min_track_record_length(sr=0.0)
        assert math.isinf(trl)

    def test_higher_sr_needs_fewer_observations(self):
        trl_low  = min_track_record_length(sr=0.5)
        trl_high = min_track_record_length(sr=2.0)
        assert trl_low > trl_high


class TestProbabilityOfBacktestOverfitting:
    def test_all_positive_sharpes_returns_zero(self):
        pbo = probability_of_backtest_overfitting([1.0, 0.5, 0.8, 2.0])
        assert pbo == 0.0

    def test_all_negative_sharpes_returns_one(self):
        pbo = probability_of_backtest_overfitting([-0.1, -0.5, -1.0])
        assert pbo == 1.0

    def test_half_negative_returns_half(self):
        pbo = probability_of_backtest_overfitting([1.0, -1.0, 0.5, -0.5])
        assert abs(pbo - 0.5) < 1e-9

    def test_empty_list_returns_one(self):
        """No OOS data → worst-case assumption."""
        assert probability_of_backtest_overfitting([]) == 1.0

    def test_output_in_unit_interval(self):
        pbo = probability_of_backtest_overfitting([0.1, -0.2, 0.3, 0.4])
        assert 0.0 <= pbo <= 1.0
