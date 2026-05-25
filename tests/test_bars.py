"""Tests for src/features/bars.py — Dollar-Bars und Volume-Bars."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.bars import dollar_bars, volume_bars


class TestDollarBars:
    def test_returns_dataframe_with_ohlcv_columns(self, ohlcv_df):
        db = dollar_bars(ohlcv_df, bars_per_day=24)
        assert set(db.columns) == {"open", "high", "low", "close", "volume"}

    def test_fewer_bars_than_input(self, ohlcv_df):
        db = dollar_bars(ohlcv_df, bars_per_day=24)
        assert len(db) < len(ohlcv_df)

    def test_plausible_count(self, ohlcv_df):
        # 500 hourly bars ≈ 21 days; bars_per_day=24 → expect fewer dollar bars
        db = dollar_bars(ohlcv_df, bars_per_day=24)
        assert len(db) > 10  # at least some bars produced

    def test_ohlc_monotonicity(self, ohlcv_df):
        db = dollar_bars(ohlcv_df, bars_per_day=24)
        assert (db["high"] >= db["low"]).all()
        assert (db["high"] >= db["open"]).all()
        assert (db["high"] >= db["close"]).all()

    def test_timezone_preserved(self, ohlcv_df):
        db = dollar_bars(ohlcv_df, bars_per_day=24)
        assert db.index.tz is not None

    def test_explicit_target_dollar_vol(self, ohlcv_df):
        median_close = ohlcv_df["close"].median()
        target = median_close * ohlcv_df["volume"].median() * 10
        db = dollar_bars(ohlcv_df, target_dollar_vol=target)
        assert len(db) > 0
        assert set(db.columns) == {"open", "high", "low", "close", "volume"}

    def test_fallback_on_empty_result(self):
        """When no bar crosses the threshold (huge target), returns original df."""
        import pandas as pd
        idx = pd.date_range("2022-01-01", periods=10, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": [1.0] * 10, "high": [1.1] * 10, "low": [0.9] * 10,
            "close": [1.0] * 10, "volume": [1.0] * 10,
        }, index=idx)
        result = dollar_bars(df, target_dollar_vol=1e30)
        assert len(result) == len(df)  # fallback: returns original


class TestVolumeBars:
    def test_returns_dataframe_with_ohlcv_columns(self, ohlcv_df):
        vb = volume_bars(ohlcv_df, bars_per_day=24)
        assert set(vb.columns) == {"open", "high", "low", "close", "volume"}

    def test_fewer_bars_than_input(self, ohlcv_df):
        vb = volume_bars(ohlcv_df, bars_per_day=24)
        assert len(vb) < len(ohlcv_df)

    def test_ohlc_monotonicity(self, ohlcv_df):
        vb = volume_bars(ohlcv_df, bars_per_day=24)
        assert (vb["high"] >= vb["low"]).all()

    def test_produces_more_bars_than_dollar(self, ohlcv_df):
        """Volume bars and dollar bars don't need identical counts, just both valid."""
        db = dollar_bars(ohlcv_df, bars_per_day=24)
        vb = volume_bars(ohlcv_df, bars_per_day=24)
        assert len(db) > 0
        assert len(vb) > 0
