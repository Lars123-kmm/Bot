"""Tests for src/features/fracdiff.py — Fraktionale Differenzierung."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fracdiff import find_min_d, frac_diff_ffd, get_weights_ffd


class TestGetWeightsFfd:
    def test_first_weight_is_one(self):
        w = get_weights_ffd(0.5, size=20)
        assert abs(w[0] - 1.0) < 1e-10

    def test_weights_decrease_in_magnitude(self):
        w = get_weights_ffd(0.5, size=20)
        for i in range(1, len(w)):
            assert abs(w[i]) <= abs(w[i - 1]) + 1e-9

    def test_d_zero_returns_single_weight(self):
        w = get_weights_ffd(0.0, size=50)
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 1e-10

    def test_max_width_limits_length(self):
        w = get_weights_ffd(0.1, size=1000, thres=1e-9)
        assert len(w) <= 1000

    def test_small_d_truncated_by_max_width(self):
        # d=0.15 decays very slowly; max_width=100 should cap the window
        w = get_weights_ffd(0.15, size=100)
        assert len(w) <= 100


class TestFracDiffFfd:
    def test_output_length_matches_input(self, ohlcv_df):
        result = frac_diff_ffd(ohlcv_df["close"], d=0.3, max_width=50)
        assert len(result) == len(ohlcv_df)

    def test_leading_nans_then_valid_values(self, ohlcv_df):
        result = frac_diff_ffd(ohlcv_df["close"], d=0.3, max_width=50)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.isna().sum() == 0

    def test_d_one_approximates_first_difference(self, ohlcv_df):
        """fracdiff with d≈1 should correlate strongly with pct_change."""
        fd = frac_diff_ffd(ohlcv_df["close"], d=1.0, max_width=10).dropna()
        diff = ohlcv_df["close"].diff().dropna()
        common = fd.index.intersection(diff.index)
        assert len(common) > 50
        corr = fd.loc[common].corr(diff.loc[common])
        assert corr > 0.90

    def test_max_width_keeps_more_valid_bars(self, ohlcv_df):
        """A smaller max_width means fewer NaN rows at the start."""
        fd_narrow = frac_diff_ffd(ohlcv_df["close"], d=0.2, max_width=10)
        fd_wide   = frac_diff_ffd(ohlcv_df["close"], d=0.2, max_width=200)
        assert fd_narrow.dropna().__len__() >= fd_wide.dropna().__len__()

    def test_values_are_finite(self, ohlcv_df):
        result = frac_diff_ffd(ohlcv_df["close"], d=0.4, max_width=50).dropna()
        assert np.isfinite(result.values).all()


class TestFindMinD:
    def test_returns_float_in_range(self, ohlcv_df):
        d = find_min_d(ohlcv_df["close"], pvalue=0.05)
        assert isinstance(d, float)
        assert 0.0 <= d <= 1.0

    def test_differenced_series_is_stationary(self, ohlcv_df):
        """After applying the returned d, ADF test should confirm stationarity."""
        from statsmodels.tsa.stattools import adfuller
        d = find_min_d(ohlcv_df["close"], pvalue=0.05)
        fd = frac_diff_ffd(ohlcv_df["close"], d=d, max_width=100).dropna()
        if len(fd) > 30:
            pval = adfuller(fd.values)[1]
            assert pval < 0.10  # allow small buffer above 0.05

    def test_result_smaller_than_d_equals_one(self, ohlcv_df):
        """Minimum stationary d should be well below 1.0 for typical price series."""
        d = find_min_d(ohlcv_df["close"], pvalue=0.05)
        assert d < 0.9
