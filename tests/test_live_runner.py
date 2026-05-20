"""Tests for src/runner/live_runner.py — multi-symbol trading loop."""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.Execution.order_state import ActiveTrade
from src.runner.live_runner import LiveRunner, _seconds_to_next_bar
from src.runner.paper_broker import PaperBroker


class _FakeFetcher:
    """Returns a fixed synthetic OHLCV frame instead of hitting Binance."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def fetch_ohlcv(self, symbol: str, timeframe: str = "H1", bars: int = 500) -> pd.DataFrame:
        return self._df.iloc[-bars:].copy()


@pytest.fixture
def runner_cfg(cfg) -> dict:
    """Config tuned for fast, network-free runner tests."""
    c = copy.deepcopy(cfg)
    c["hmm"]["enabled"] = False          # skip HMM fit for speed
    c["ml"]["enabled"] = False           # no champion model on disk
    c["data"]["bars"] = 300
    c["ml"]["train_bars"] = 200
    return c


@pytest.fixture
def runner(runner_cfg, ohlcv_df_large) -> LiveRunner:
    r = LiveRunner(
        cfg=runner_cfg,
        mode="paper",
        symbols=["BTCUSDT", "ETHUSDT"],
        initial_capital=10_000.0,
    )
    r.fetcher = _FakeFetcher(ohlcv_df_large)
    return r


class TestSecondsToNextBar:
    def test_h1_within_bounds(self):
        secs = _seconds_to_next_bar("H1")
        assert 2 <= secs <= 3602

    def test_m5_within_bounds(self):
        secs = _seconds_to_next_bar("M5")
        assert 2 <= secs <= 302

    def test_unknown_timeframe_defaults_to_hour(self):
        secs = _seconds_to_next_bar("XYZ")
        assert 2 <= secs <= 3602


class TestConstruction:
    def test_states_created_per_symbol(self, runner):
        assert set(runner._states.keys()) == {"BTCUSDT", "ETHUSDT"}

    def test_paper_broker_selected(self, runner):
        assert isinstance(runner.broker, PaperBroker)

    def test_invalid_mode_rejected(self, runner_cfg):
        with pytest.raises(ValueError):
            LiveRunner(cfg=runner_cfg, mode="bogus")

    def test_no_model_disables_ml(self, runner):
        assert runner._meta_model is None


class TestTickLoop:
    def test_run_once_does_not_crash(self, runner):
        runner.run_once()
        assert runner._total_bars == 1

    def test_run_once_twice_increments(self, runner):
        runner.run_once()
        runner.run_once()
        assert runner._total_bars == 2

    def test_account_stays_valid(self, runner):
        runner.run_once()
        info = runner.broker.get_account_info()
        assert info["equity"] > 0


class TestDashboardState:
    def test_state_has_expected_keys(self, runner):
        state = runner._dashboard_state()
        for key in ("mode", "symbols_list", "account", "positions", "recent_trades"):
            assert key in state

    def test_symbols_listed(self, runner):
        state = runner._dashboard_state()
        assert state["symbols_list"] == ["BTCUSDT", "ETHUSDT"]


class TestPortfolioRisk:
    def test_position_scaled_when_budget_tight(self, runner):
        runner._max_portfolio_risk = 0.02      # budget = 200 USD at 10k equity
        # BTC already consumes 150 of the 200 budget
        btc = runner._states["BTCUSDT"]
        btc.active_trade = ActiveTrade.from_entry(
            ticket=1, direction=1, symbol="BTCUSDT", entry_price=50_000,
            size=0.1, stop=49_000, take_profit=53_000, atr_at_entry=500,
        )
        btc.current_risk = 150.0

        eth = runner._states["ETHUSDT"]
        runner.broker.set_price("ETHUSDT", 3_000.0)
        last_bar = pd.Series({"close": 3_000.0, "high": 3_010.0, "low": 2_990.0})
        runner._open_position(eth, direction=1, price=3_000.0, atr=30.0,
                              last_bar=last_bar, ml_prob=None)

        # desired risk ~100, only 50 available → scaled to ~50
        assert eth.active_trade is not None
        assert eth.current_risk == pytest.approx(50.0, abs=1.0)

    def test_position_skipped_when_budget_exhausted(self, runner):
        runner._max_portfolio_risk = 0.02      # budget = 200 USD
        btc = runner._states["BTCUSDT"]
        btc.active_trade = ActiveTrade.from_entry(
            ticket=1, direction=1, symbol="BTCUSDT", entry_price=50_000,
            size=0.1, stop=49_000, take_profit=53_000, atr_at_entry=500,
        )
        btc.current_risk = 197.0               # almost the whole budget

        eth = runner._states["ETHUSDT"]
        runner.broker.set_price("ETHUSDT", 3_000.0)
        last_bar = pd.Series({"close": 3_000.0, "high": 3_010.0, "low": 2_990.0})
        runner._open_position(eth, direction=1, price=3_000.0, atr=30.0,
                              last_bar=last_bar, ml_prob=None)

        assert eth.active_trade is None        # not enough budget → skipped


class TestDegradationDetection:
    def test_low_winrate_triggers_retrain(self, runner, monkeypatch):
        called = {"retrain": False}
        monkeypatch.setattr(runner, "_retrain", lambda: called.__setitem__("retrain", True))

        # fill the window with mostly losses (win-rate 10% < 35% threshold)
        runner._recent_outcomes.clear()
        for i in range(runner._degr_window):
            runner._recent_outcomes.append(1 if i < 2 else 0)
        runner._check_degradation()

        assert called["retrain"] is True

    def test_good_winrate_does_not_retrain(self, runner, monkeypatch):
        called = {"retrain": False}
        monkeypatch.setattr(runner, "_retrain", lambda: called.__setitem__("retrain", True))

        runner._recent_outcomes.clear()
        for i in range(runner._degr_window):
            runner._recent_outcomes.append(1 if i < 15 else 0)  # 75% win-rate
        runner._check_degradation()

        assert called["retrain"] is False

    def test_partial_window_does_not_retrain(self, runner, monkeypatch):
        called = {"retrain": False}
        monkeypatch.setattr(runner, "_retrain", lambda: called.__setitem__("retrain", True))

        runner._recent_outcomes.clear()
        runner._recent_outcomes.append(0)   # only one outcome, window not full
        runner._check_degradation()

        assert called["retrain"] is False
