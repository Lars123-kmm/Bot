"""Tests for src/ml/sample_weights.py — Sample-Uniqueness-Gewichte."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.sample_weights import (
    avg_uniqueness,
    compute_sample_weights,
    get_t1,
    num_concurrent_events,
)


@pytest.fixture
def simple_index():
    return pd.date_range("2022-01-01", periods=200, freq="1h", tz="UTC")


@pytest.fixture
def signal_times(simple_index):
    """Every 10th bar is a signal."""
    return simple_index[::10]


class TestGetT1:
    def test_returns_series_of_same_length_as_signal_idx(self, signal_times, simple_index):
        t1 = get_t1(signal_times, simple_index, max_bars=20)
        assert len(t1) == len(signal_times)

    def test_end_times_are_after_start_times(self, signal_times, simple_index):
        t1 = get_t1(signal_times, simple_index, max_bars=20)
        for t0, end in t1.items():
            assert end >= t0

    def test_end_time_does_not_exceed_data_end(self, signal_times, simple_index):
        t1 = get_t1(signal_times, simple_index, max_bars=20)
        last = simple_index[-1]
        for _, end in t1.items():
            assert end <= last

    def test_max_bars_caps_lookahead(self, signal_times, simple_index):
        t1_short = get_t1(signal_times, simple_index, max_bars=5)
        t1_long  = get_t1(signal_times, simple_index, max_bars=50)
        for t0 in signal_times:
            if t0 in t1_short.index and t0 in t1_long.index:
                assert t1_short[t0] <= t1_long[t0]


class TestNumConcurrentEvents:
    def test_counts_are_non_negative(self, signal_times, simple_index):
        t1 = get_t1(signal_times, simple_index, max_bars=20)
        nc = num_concurrent_events(simple_index, t1)
        assert (nc >= 0).all()

    def test_max_count_bounded_by_n_signals(self, signal_times, simple_index):
        t1 = get_t1(signal_times, simple_index, max_bars=20)
        nc = num_concurrent_events(simple_index, t1)
        assert nc.max() <= len(signal_times)


class TestAvgUniqueness:
    def test_values_between_zero_and_one(self, signal_times, simple_index):
        t1 = get_t1(signal_times, simple_index, max_bars=20)
        nc = num_concurrent_events(simple_index, t1)
        uniq = avg_uniqueness(t1, nc)
        assert (uniq > 0).all()
        assert (uniq <= 1.0 + 1e-9).all()

    def test_higher_overlap_lower_uniqueness(self, simple_index):
        """Signals clustered closely should have lower uniqueness than spread-out ones."""
        close_signals = simple_index[:5]   # every bar → lots of overlap
        spread_signals = simple_index[::40]  # every 40th bar → little overlap
        t1_close  = get_t1(close_signals,  simple_index, max_bars=10)
        t1_spread = get_t1(spread_signals, simple_index, max_bars=10)
        nc_close  = num_concurrent_events(simple_index, t1_close)
        nc_spread = num_concurrent_events(simple_index, t1_spread)
        u_close  = avg_uniqueness(t1_close,  nc_close).mean()
        u_spread = avg_uniqueness(t1_spread, nc_spread).mean()
        assert u_spread > u_close


class TestComputeSampleWeights:
    def test_returns_series_with_correct_index(self, signal_times, simple_index):
        labels = pd.DataFrame({"label": np.ones(len(signal_times))}, index=signal_times)
        w = compute_sample_weights(labels, simple_index, max_bars=20)
        assert set(w.index).issubset(set(signal_times))

    def test_all_weights_positive(self, signal_times, simple_index):
        labels = pd.DataFrame({"label": np.ones(len(signal_times))}, index=signal_times)
        w = compute_sample_weights(labels, simple_index, max_bars=20)
        assert (w > 0).all()

    def test_mean_approximately_one(self, signal_times, simple_index):
        labels = pd.DataFrame({"label": np.ones(len(signal_times))}, index=signal_times)
        w = compute_sample_weights(labels, simple_index, max_bars=20)
        assert abs(w.mean() - 1.0) < 0.1

    def test_return_pct_column_influences_weights(self, signal_times, simple_index):
        """Labels with return_pct column should produce different weights than without."""
        labels_plain = pd.DataFrame(
            {"label": np.ones(len(signal_times))}, index=signal_times)
        labels_ret = pd.DataFrame({
            "label": np.ones(len(signal_times)),
            "return_pct": np.linspace(0.001, 0.05, len(signal_times)),
        }, index=signal_times)
        w_plain = compute_sample_weights(labels_plain, simple_index, max_bars=20)
        w_ret   = compute_sample_weights(labels_ret,   simple_index, max_bars=20)
        # Weights should differ when return_pct column is present
        assert not np.allclose(w_plain.values, w_ret.values, atol=1e-6)
