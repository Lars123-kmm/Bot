"""
Shared test fixtures: synthetic OHLCV data and a minimal bot config.
All tests import these via pytest's conftest auto-discovery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Synthetic OHLCV fixture ────────────────────────────────────────────────

@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """500-bar synthetic OHLCV DataFrame with UTC DatetimeIndex."""
    return _make_ohlcv(500, seed=0)


@pytest.fixture
def ohlcv_df_large() -> pd.DataFrame:
    """2500-bar synthetic OHLCV DataFrame for tests that need longer series."""
    return _make_ohlcv(2500, seed=1)


def _make_ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    log_ret = rng.normal(0.0001, 0.012, n)
    close = 40_000 * np.exp(np.cumsum(log_ret))
    opens  = close * np.exp(rng.normal(0, 0.001, n))
    highs  = np.maximum(opens, close) * (1 + rng.uniform(0.001, 0.005, n))
    lows   = np.minimum(opens, close) * (1 - rng.uniform(0.001, 0.005, n))
    vols   = 80_000 * (1 + rng.uniform(0, 1, n))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": close, "volume": vols},
        index=idx,
    )


# ── Minimal config fixture ─────────────────────────────────────────────────

@pytest.fixture
def cfg() -> dict:
    """Minimal bot config sufficient for unit tests (no MT5/Binance needed)."""
    return {
        "symbol": "BTCUSDT",
        "timeframe": "H1",
        "timezone": "UTC",
        "data": {"bars": 500},
        "strategy": {
            "ema_fast": 20, "ema_slow": 50,
            "adx_period": 14, "adx_threshold": 10,
            "rsi_period": 14,
            "rsi_long_min": 25, "rsi_long_max": 75,
            "rsi_short_min": 25, "rsi_short_max": 75,
            "volume_ma_period": 20, "volume_ratio_min": 0.5,
            "extended_max_pct": 10.0,
            "atr_period": 14, "atr_stop_mult": 2.0, "atr_tp_mult": 3.0,
            "trailing_activate_atr": 1.5, "trailing_distance_atr": 1.0,
            "time_exit_bars": 48, "signal_on_close": True,
        },
        "risk": {
            "account_risk_per_trade": 0.01,
            "max_total_drawdown": 0.10,
            "max_daily_drawdown": 0.02,
            "max_consecutive_losses": 5,
            "cooldown_minutes": 30,
            "max_spread_atr_ratio": 0.15,
        },
        "execution": {"mode": "demo", "magic_number": 240101,
                      "slippage_points": 10, "deviation_points": 20},
        "ml": {
            "enabled": True,
            "min_probability": 0.52,
            "model_path": "/tmp/test_models",
            "label_upper_mult": 2.0, "label_lower_mult": 2.0,
            "label_max_bars": 48,
            "train_bars": 800, "test_bars": 200,
            "step_bars": 150, "purge_bars": 24,
            "lgbm_num_leaves": 15, "lgbm_learning_rate": 0.1,
            "lgbm_n_estimators": 50, "lgbm_min_child_samples": 5,
        },
        "regime": {
            "enabled": True, "window": 50,
            "hurst_trend_threshold": 0.55, "hurst_range_threshold": 0.45,
            "vol_volatile_threshold": 0.80, "adx_min_for_trend": 15,
            "half_size_in_volatile": True,
        },
        "bars": {"type": "time", "target_bars_per_day": 50},
        "binance": {"enabled": False, "symbol": "BTCUSDT", "order_book_levels": 5},
        "hmm": {"enabled": True, "n_components": 2,
                "model_path": "/tmp/test_models/hmm.pkl", "retrain_bars": 300},
        "champion_challenger": {"enabled": True, "n_windows_needed": 2, "metric": "auc"},
        "online_model": {"enabled": True, "blend_weight": 0.3,
                         "model_path": "/tmp/test_models/online.pkl"},
    }
