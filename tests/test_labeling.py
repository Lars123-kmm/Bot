"""Tests for src/ml/labeling.py — triple-barrier labeling and overlap purging."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.labeling import purge_overlap, triple_barrier_labels


def _make_df(closes, highs, lows, signals, atr=1.0):
    n = len(closes)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "close": closes,
            "high": highs,
            "low": lows,
            "atr": [atr] * n,
            "signal": signals,
        },
        index=idx,
    )


class TestTripleBarrierLabels:
    def test_long_hits_take_profit(self):
        # entry 100, atr 1, upper = 100 + 2*1 = 102
        closes = [100] + [100.5] * 9
        highs = [100] + [101] * 4 + [102.5] + [101] * 4  # bar 5 breaks upper
        lows = [100] + [99.5] * 9
        signals = [1] + [0] * 9
        df = _make_df(closes, highs, lows, signals)
        mask = pd.Series([True] + [False] * 9, index=df.index)

        labels = triple_barrier_labels(df, mask, upper_mult=2.0, lower_mult=2.0, max_bars=8)
        assert len(labels) == 1
        assert labels.iloc[0]["label"] == 1
        assert labels.iloc[0]["bars_to_exit"] == 5
        assert labels.iloc[0]["return_pct"] == pytest.approx(2.0)

    def test_long_hits_stop_loss(self):
        # entry 100, lower = 100 - 2*1 = 98
        closes = [100] + [99.5] * 9
        highs = [100] + [101] * 9
        lows = [100] + [99] * 2 + [97.5] + [99] * 6  # bar 3 breaks lower
        signals = [1] + [0] * 9
        df = _make_df(closes, highs, lows, signals)
        mask = pd.Series([True] + [False] * 9, index=df.index)

        labels = triple_barrier_labels(df, mask, upper_mult=2.0, lower_mult=2.0, max_bars=8)
        assert labels.iloc[0]["label"] == 0
        assert labels.iloc[0]["bars_to_exit"] == 3
        assert labels.iloc[0]["return_pct"] == pytest.approx(-2.0)

    def test_time_barrier_gives_label_zero(self):
        # price never breaks either barrier within max_bars
        closes = [100] + [100.2] * 9
        highs = [100] + [101] * 9
        lows = [100] + [99] * 9
        signals = [1] + [0] * 9
        df = _make_df(closes, highs, lows, signals)
        mask = pd.Series([True] + [False] * 9, index=df.index)

        labels = triple_barrier_labels(df, mask, upper_mult=2.0, lower_mult=2.0, max_bars=5)
        assert labels.iloc[0]["label"] == 0
        assert labels.iloc[0]["bars_to_exit"] == 5

    def test_short_hits_take_profit(self):
        # short entry 100: upper barrier = 100 - 2*1 = 98 (profit side)
        closes = [100] + [99.5] * 9
        highs = [100] + [101] * 9
        lows = [100] + [99] * 3 + [97.5] + [99] * 5  # bar 4 breaks below 98
        signals = [-1] + [0] * 9
        df = _make_df(closes, highs, lows, signals)
        mask = pd.Series([True] + [False] * 9, index=df.index)

        labels = triple_barrier_labels(df, mask, upper_mult=2.0, lower_mult=2.0, max_bars=8)
        assert labels.iloc[0]["label"] == 1
        assert labels.iloc[0]["bars_to_exit"] == 4
        assert labels.iloc[0]["return_pct"] == pytest.approx(2.0)

    def test_empty_signal_mask_returns_empty(self):
        df = _make_df([100] * 5, [101] * 5, [99] * 5, [0] * 5)
        mask = pd.Series([False] * 5, index=df.index)
        labels = triple_barrier_labels(df, mask)
        assert labels.empty
        assert list(labels.columns) == ["label", "bars_to_exit", "return_pct"]

    def test_multiple_signals(self):
        closes = [100] * 20
        highs = [100] + [102.5] * 19   # every future bar breaks upper
        lows = [100] * 20
        signals = [1, 0, 0, 0, 0, 1] + [0] * 14
        df = _make_df(closes, highs, lows, signals)
        mask = pd.Series([i in (0, 5) for i in range(20)], index=df.index)
        labels = triple_barrier_labels(df, mask, upper_mult=2.0, max_bars=8)
        assert len(labels) == 2
        assert (labels["label"] == 1).all()


class TestPurgeOverlap:
    def test_overlapping_labels_removed(self):
        n = 100
        idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
        feature_df = pd.DataFrame({"x": np.arange(n)}, index=idx)
        # two labels only 2 bars apart → second overlaps with max_bars=48
        labels = pd.DataFrame(
            {"label": [1, 0], "bars_to_exit": [10, 10], "return_pct": [1.0, -1.0]},
            index=[idx[0], idx[2]],
        )
        feats_p, labels_p = purge_overlap(labels, feature_df, max_bars=48)
        assert len(labels_p) == 1
        assert labels_p.index[0] == idx[0]

    def test_non_overlapping_labels_kept(self):
        n = 200
        idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
        feature_df = pd.DataFrame({"x": np.arange(n)}, index=idx)
        labels = pd.DataFrame(
            {"label": [1, 0], "bars_to_exit": [10, 10], "return_pct": [1.0, -1.0]},
            index=[idx[0], idx[60]],   # 60 bars apart > max_bars
        )
        feats_p, labels_p = purge_overlap(labels, feature_df, max_bars=48)
        assert len(labels_p) == 2

    def test_features_and_labels_aligned(self):
        n = 200
        idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
        feature_df = pd.DataFrame({"x": np.arange(n)}, index=idx)
        labels = pd.DataFrame(
            {"label": [1, 0, 1], "bars_to_exit": [5, 5, 5], "return_pct": [1.0, -1.0, 2.0]},
            index=[idx[0], idx[60], idx[120]],
        )
        feats_p, labels_p = purge_overlap(labels, feature_df, max_bars=48)
        assert list(feats_p.index) == list(labels_p.index)
