"""Tests for src/runner/paper_broker.py — in-memory paper trading broker."""
from __future__ import annotations

import pytest

from src.runner.paper_broker import PaperBroker


@pytest.fixture
def broker() -> PaperBroker:
    b = PaperBroker(initial_balance=10_000.0, fee_rate=0.0004, spread_pct=0.0)
    b.set_price("BTCUSDT", 50_000.0)
    return b


class TestOrderPlacement:
    def test_tickets_increment(self, broker):
        t1 = broker.place_market_order("BTCUSDT", 1, 0.01, 48_000, 53_000)
        t2 = broker.place_market_order("BTCUSDT", 1, 0.01, 48_000, 53_000)
        assert t2 == t1 + 1

    def test_fee_is_deducted_on_entry(self, broker):
        before = broker.get_account_info()["balance"]
        broker.place_market_order("BTCUSDT", 1, 0.01, 48_000, 53_000)
        after = broker.get_account_info()["balance"]
        # fee = 50000 * 0.01 * 0.0004 = 0.2
        assert after == pytest.approx(before - 0.2, abs=1e-6)

    def test_open_position_appears(self, broker):
        broker.place_market_order("BTCUSDT", 1, 0.01, 48_000, 53_000)
        positions = broker.get_open_positions("BTCUSDT")
        assert len(positions) == 1
        assert positions[0]["direction"] == 1

    def test_open_positions_filter_by_symbol(self, broker):
        broker.set_price("ETHUSDT", 3_000.0)
        broker.place_market_order("BTCUSDT", 1, 0.01, 48_000, 53_000)
        broker.place_market_order("ETHUSDT", -1, 0.1, 3_100, 2_800)
        assert len(broker.get_open_positions("BTCUSDT")) == 1
        assert len(broker.get_open_positions("ETHUSDT")) == 1
        assert len(broker.get_open_positions()) == 2


class TestStopLossTakeProfit:
    def test_long_sl_hit(self, broker):
        t = broker.place_market_order("BTCUSDT", 1, 0.01, 49_000, 53_000)
        closed = broker.update_bar("BTCUSDT", 50_000, 50_000, 48_500, 49_500)
        assert closed == [t]
        trade = broker.get_closed_trade(t)
        assert trade["close_reason"] == "sl"
        assert trade["pnl"] < 0

    def test_long_tp_hit(self, broker):
        t = broker.place_market_order("BTCUSDT", 1, 0.01, 49_000, 52_000)
        closed = broker.update_bar("BTCUSDT", 50_000, 52_500, 50_000, 51_800)
        assert closed == [t]
        trade = broker.get_closed_trade(t)
        assert trade["close_reason"] == "tp"
        assert trade["pnl"] > 0

    def test_short_sl_hit(self, broker):
        t = broker.place_market_order("BTCUSDT", -1, 0.01, 51_000, 47_000)
        closed = broker.update_bar("BTCUSDT", 50_000, 51_500, 50_000, 51_200)
        assert closed == [t]
        assert broker.get_closed_trade(t)["close_reason"] == "sl"

    def test_short_tp_hit(self, broker):
        t = broker.place_market_order("BTCUSDT", -1, 0.01, 51_000, 47_000)
        closed = broker.update_bar("BTCUSDT", 50_000, 50_000, 46_500, 47_200)
        assert closed == [t]
        trade = broker.get_closed_trade(t)
        assert trade["close_reason"] == "tp"
        assert trade["pnl"] > 0

    def test_no_hit_keeps_position_open(self, broker):
        t = broker.place_market_order("BTCUSDT", 1, 0.01, 48_000, 53_000)
        closed = broker.update_bar("BTCUSDT", 50_000, 50_500, 49_500, 50_100)
        assert closed == []
        assert len(broker.get_open_positions()) == 1


class TestAccountAndSummary:
    def test_equity_reflects_unrealized_pnl(self, broker):
        broker.place_market_order("BTCUSDT", 1, 0.1, 48_000, 53_000)
        broker.set_price("BTCUSDT", 51_000.0)  # +1000 * 0.1 = +100
        info = broker.get_account_info()
        assert info["profit"] == pytest.approx(100.0, abs=1e-6)
        assert info["equity"] > info["balance"]

    def test_manual_close(self, broker):
        t = broker.place_market_order("BTCUSDT", 1, 0.01, 48_000, 53_000)
        assert broker.close_position(t, "BTCUSDT", 0.01, 1) is True
        assert len(broker.get_open_positions()) == 0
        assert broker.get_closed_trade(t)["close_reason"] == "manual"

    def test_close_unknown_ticket(self, broker):
        assert broker.close_position(999, "BTCUSDT", 0.01, 1) is False

    def test_summary_win_rate(self, broker):
        # one winner
        t1 = broker.place_market_order("BTCUSDT", 1, 0.01, 49_000, 52_000)
        broker.update_bar("BTCUSDT", 50_000, 52_500, 50_000, 51_800)
        # one loser
        broker.set_price("BTCUSDT", 50_000.0)
        t2 = broker.place_market_order("BTCUSDT", 1, 0.01, 49_000, 53_000)
        broker.update_bar("BTCUSDT", 50_000, 50_000, 48_500, 49_500)
        summary = broker.summary()
        assert summary["n_trades"] == 2
        assert summary["win_rate"] == pytest.approx(0.5)

    def test_summary_empty(self, broker):
        summary = broker.summary()
        assert summary["n_trades"] == 0
        assert summary["win_rate"] == 0.0
