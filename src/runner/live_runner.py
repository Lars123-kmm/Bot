from __future__ import annotations

"""
LiveRunner — multi-symbol continuous trading loop.

Modes:
  paper   — simulates trades in memory, no API key needed
  live    — real orders via BinanceBroker (requires BINANCE_API_KEY/SECRET)

Features:
  - Fetches live OHLCV from Binance Futures (public endpoint)
  - Runs the full strategy pipeline per symbol each bar
  - Online learning (River SRP) updates after each closed paper trade
  - Auto-retraining via Walk-Forward + Champion-Challenger every N bars
  - Ctrl+C for graceful shutdown with final P&L summary
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.binance_fetch import BinanceFetcher
from src.Execution.order_state import ActiveTrade
from src.features.data_prep import prepare_market_data
from src.features.indicators import compute_indicators
from src.features.ml_features import ML_FEATURE_COLS, build_ml_features
from src.ml.model_store import ModelStore
from src.ml.online_model import OnlineMetaFilter
from src.risk.guards import ConsecutiveLossGuard, DrawdownGuard
from src.risk.sizing import calculate_position, probability_scaled_position
from src.runner.paper_broker import PaperBroker
from src.strategies.ema_atr import generate_signals

logger = logging.getLogger(__name__)

_TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


def _seconds_to_next_bar(timeframe: str) -> float:
    """Seconds until the next bar opens (aligned to wall-clock boundaries)."""
    bar_sec = _TIMEFRAME_SECONDS.get(timeframe, 3600)
    now = time.time()
    elapsed = now % bar_sec
    return bar_sec - elapsed + 2   # +2 s buffer for Binance to close candle


class _SymbolState:
    """Per-symbol mutable state."""

    def __init__(self, symbol: str, online_filter: OnlineMetaFilter):
        self.symbol = symbol
        self.online_filter = online_filter
        self.active_trade: Optional[ActiveTrade] = None
        self.bars_since_retrain: int = 0
        self._entry_features: Optional[Dict[str, float]] = None
        self._entry_ticket: Optional[int] = None


class LiveRunner:
    """
    Runs a multi-symbol trading loop using live Binance data.

    Args:
        cfg:            Bot config dict (from default.yaml)
        mode:           "paper" or "live"
        symbols:        List of Binance Futures symbols, e.g. ["BTCUSDT", "ETHUSDT"]
        initial_capital: Starting balance for paper trading
        retrain_every:  Bars between Walk-Forward retrains (0 = disabled)
        report_dir:     Directory for daily reports (optional)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        mode: str = "paper",
        symbols: Optional[List[str]] = None,
        initial_capital: float = 10_000.0,
        retrain_every: int = 0,
        report_dir: Optional[Path] = None,
    ):
        if mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {mode!r}")

        self.cfg = cfg
        self.mode = mode
        self.symbols: List[str] = symbols or [cfg.get("symbol", "BTCUSDT")]
        self.timeframe: str = cfg.get("timeframe", "H1")
        self.retrain_every = retrain_every
        self.report_dir = report_dir

        # Broker
        if mode == "paper":
            self.broker = PaperBroker(
                initial_balance=initial_capital,
                fee_rate=cfg.get("live", {}).get("fee_rate", 0.0004),
            )
        else:
            from src.Execution.binance_broker import BinanceBroker
            self.broker = BinanceBroker(cfg)

        # Shared data fetcher
        self.fetcher = BinanceFetcher()

        # Load or init model
        self._store = ModelStore(cfg.get("ml", {}).get("model_path", "models/"))
        self._meta_model = self._load_model()

        # Per-symbol state
        self._states: Dict[str, _SymbolState] = {
            sym: _SymbolState(sym, OnlineMetaFilter(cfg)) for sym in self.symbols
        }

        # Global guards
        self._dd_guard = DrawdownGuard(
            max_dd=cfg.get("risk", {}).get("max_total_drawdown", 0.10)
        )
        self._loss_guard = ConsecutiveLossGuard(
            max_losses=cfg.get("risk", {}).get("max_consecutive_losses", 5)
        )

        # Retrain counter
        self._total_bars = 0

        logger.info(
            "LiveRunner init: mode=%s symbols=%s tf=%s retrain_every=%d",
            mode, self.symbols, self.timeframe, retrain_every,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("=== LiveRunner started (mode=%s) — Ctrl+C to stop ===", self.mode)
        try:
            while True:
                self._tick()
                sleep_sec = _seconds_to_next_bar(self.timeframe)
                logger.info("Waiting %.0f s for next %s bar...", sleep_sec, self.timeframe)
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            logger.info("Shutdown requested.")
        finally:
            self._shutdown()

    def run_once(self) -> None:
        """Single tick — useful for testing or external scheduling."""
        self._tick()

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        self._total_bars += 1

        # Equity snapshot for drawdown guard
        acct = self.broker.get_account_info()
        equity = acct["equity"]
        self._dd_guard.update(equity)

        if self._dd_guard.tripped:
            logger.warning("DRAWDOWN GUARD tripped — no new trades this bar.")

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as exc:
                logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

        # Auto-retrain
        if self.retrain_every > 0 and self._total_bars % self.retrain_every == 0:
            logger.info("Auto-retrain triggered (bar %d).", self._total_bars)
            self._retrain()

    def _process_symbol(self, symbol: str) -> None:
        state = self._states[symbol]
        state.bars_since_retrain += 1

        # --- Fetch latest OHLCV ---
        bars_needed = max(
            self.cfg.get("data", {}).get("bars", 500),
            int(self.cfg.get("ml", {}).get("train_bars", 2000)) + 100,
        )
        df = self.fetcher.fetch_ohlcv(symbol, self.timeframe, bars=bars_needed)
        if df is None or len(df) < 100:
            logger.warning("%s: not enough data (%d bars) — skipping.", symbol, len(df) if df is not None else 0)
            return

        # Update paper broker with latest close for SL/TP check
        last = df.iloc[-1]
        if isinstance(self.broker, PaperBroker):
            closed_tickets = self.broker.update_bar(
                symbol, float(last["open"]), float(last["high"]),
                float(last["low"]), float(last["close"])
            )
            for ticket in closed_tickets:
                self._on_position_closed(state, ticket)
        else:
            self.broker.set_price(symbol, float(last["close"])) if hasattr(self.broker, "set_price") else None

        # --- Strategy pipeline ---
        df = prepare_market_data(df, self.cfg)
        df = compute_indicators(df, self.cfg)
        df = generate_signals(df, self.cfg)

        last = df.iloc[-1]
        signal = int(last.get("signal", 0))
        atr = float(last.get("atr", 0.0))
        close_price = float(last["close"])

        # --- Trailing stop update for open trade ---
        if state.active_trade is not None:
            self._update_trailing(state, last, atr)
            self._check_time_exit(state, last, df)

        # --- Open new position if no trade active ---
        if signal != 0 and state.active_trade is None:
            if self._dd_guard.tripped:
                return
            if self._loss_guard.tripped:
                logger.warning("%s: loss guard active — skipping signal.", symbol)
                return

            # ML filter
            ml_prob = self._ml_probability(df, last, symbol)
            min_prob = self.cfg.get("ml", {}).get("min_probability", 0.55)
            if ml_prob is not None and ml_prob < min_prob:
                logger.info("%s: ML filter blocked signal (prob=%.3f < %.3f)", symbol, ml_prob, min_prob)
                return

            self._open_position(state, signal, close_price, atr, last, ml_prob)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_position(
        self,
        state: _SymbolState,
        direction: int,
        price: float,
        atr: float,
        last_bar: pd.Series,
        ml_prob: Optional[float],
    ) -> None:
        symbol = state.symbol
        risk_cfg = self.cfg.get("risk", {})
        strat_cfg = self.cfg.get("strategy", {})

        acct = self.broker.get_account_info()
        capital = acct["equity"]

        if ml_prob is not None and ml_prob >= self.cfg.get("ml", {}).get("min_probability", 0.55):
            pos = probability_scaled_position(
                capital=capital,
                entry_price=price,
                atr=atr,
                risk_pct=risk_cfg.get("account_risk_per_trade", 0.01),
                meta_prob=ml_prob,
                min_prob=self.cfg.get("ml", {}).get("min_probability", 0.55),
                stop_mult=strat_cfg.get("atr_stop_mult", 2.0),
                tp_mult=strat_cfg.get("atr_tp_mult", 3.0),
                direction=direction,
            )
        else:
            pos = calculate_position(
                capital=capital,
                entry_price=price,
                atr=atr,
                risk_pct=risk_cfg.get("account_risk_per_trade", 0.01),
                stop_mult=strat_cfg.get("atr_stop_mult", 2.0),
                tp_mult=strat_cfg.get("atr_tp_mult", 3.0),
                direction=direction,
            )

        if pos["size"] <= 0:
            return

        ticket = self.broker.place_market_order(
            symbol=symbol,
            direction=direction,
            size=pos["size"],
            sl=pos["stop"],
            tp=pos["take_profit"],
        )

        state.active_trade = ActiveTrade.from_entry(
            ticket=ticket,
            direction=direction,
            symbol=symbol,
            entry_price=price,
            size=pos["size"],
            stop=pos["stop"],
            take_profit=pos["take_profit"],
            atr_at_entry=atr,
        )

        # Store features for online learning
        state._entry_ticket = ticket
        try:
            ml_feats = build_ml_features(last_bar.to_frame().T)
            state._entry_features = {
                col: float(ml_feats[col].iloc[-1]) if col in ml_feats.columns else 0.0
                for col in ML_FEATURE_COLS
            }
        except Exception:
            state._entry_features = {}

        logger.info(
            "%s OPEN dir=%d size=%.6f @ %.5f sl=%.5f tp=%.5f prob=%s",
            symbol, direction, pos["size"], price, pos["stop"], pos["take_profit"],
            f"{ml_prob:.3f}" if ml_prob else "n/a",
        )

    def _on_position_closed(self, state: _SymbolState, ticket: int) -> None:
        trade = self.broker.get_closed_trade(ticket)
        if trade is None:
            return

        pnl = trade.get("pnl", 0.0) or 0.0
        outcome = 1 if pnl > 0 else 0

        # Update loss guard
        self._loss_guard.record(outcome == 0)

        # Online learning
        if state._entry_features:
            try:
                state.online_filter.partial_fit(state._entry_features, outcome)
            except Exception as exc:
                logger.warning("Online filter update failed: %s", exc)

        state.active_trade = None
        state._entry_features = None
        state._entry_ticket = None

        logger.info(
            "%s CLOSED ticket=%d pnl=%.4f (%s)",
            state.symbol, ticket, pnl, trade.get("close_reason", "?"),
        )

    def _update_trailing(
        self, state: _SymbolState, last_bar: pd.Series, atr: float
    ) -> None:
        trade = state.active_trade
        if trade is None or atr <= 0:
            return

        strat = self.cfg.get("strategy", {})
        new_sl = trade.update_trailing(
            current_high=float(last_bar.get("high", last_bar["close"])),
            current_low=float(last_bar.get("low", last_bar["close"])),
            current_atr=atr,
            activate_mult=strat.get("trailing_activate_atr", 1.5),
            distance_mult=strat.get("trailing_distance_atr", 1.0),
        )
        if new_sl is not None:
            self.broker.modify_stop(trade.ticket, new_sl)

    def _check_time_exit(
        self, state: _SymbolState, last_bar: pd.Series, df: pd.DataFrame
    ) -> None:
        trade = state.active_trade
        if trade is None:
            return

        max_bars = self.cfg.get("strategy", {}).get("time_exit_bars", 48)
        # count bars since open
        try:
            entry_idx = df.index.searchsorted(pd.Timestamp(trade.open_time))
            bars_held = len(df) - 1 - entry_idx
        except Exception:
            return

        if bars_held >= max_bars:
            logger.info("%s: time exit after %d bars", state.symbol, bars_held)
            self.broker.close_position(
                trade.ticket, state.symbol, trade.size, trade.direction
            )
            self._on_position_closed(state, trade.ticket)

    # ------------------------------------------------------------------
    # ML helpers
    # ------------------------------------------------------------------

    def _ml_probability(
        self, df: pd.DataFrame, last_bar: pd.Series, symbol: str
    ) -> Optional[float]:
        if not self.cfg.get("ml", {}).get("enabled", True):
            return None
        if self._meta_model is None:
            return None

        try:
            ml_df = build_ml_features(df)
            if ml_df.empty:
                return None
            last_features = ml_df[ML_FEATURE_COLS].iloc[[-1]].fillna(0.0)
            base_prob = float(self._meta_model.predict_proba(last_features))

            # Blend with symbol-specific online filter
            feat_dict = {col: float(last_features[col].iloc[0]) for col in ML_FEATURE_COLS}
            state = self._states.get(symbol)
            if state is not None:
                base_prob = state.online_filter.blend(base_prob, feat_dict)

            return base_prob
        except Exception as exc:
            logger.warning("ML probability failed: %s", exc)
            return None

    def _load_model(self):
        try:
            model, info = self._store.load_champion(self.cfg)
            logger.info("Loaded champion model v%d", info.get("version", 0))
            return model
        except FileNotFoundError:
            logger.warning("No champion model found — ML filter disabled. Run --mode train first.")
            return None

    # ------------------------------------------------------------------
    # Auto-retrain
    # ------------------------------------------------------------------

    def _retrain(self) -> None:
        from src.ml.walk_forward import walk_forward_train

        # Collect all per-symbol data and concatenate
        dfs = []
        for sym in self.symbols:
            try:
                df = self.fetcher.fetch_ohlcv(sym, self.timeframe, bars=5000)
                if df is not None and len(df) > 200:
                    df = prepare_market_data(df, self.cfg)
                    df = compute_indicators(df, self.cfg)
                    dfs.append(df)
            except Exception as exc:
                logger.warning("Retrain data fetch failed for %s: %s", sym, exc)

        if not dfs:
            logger.warning("Retrain skipped — no data available.")
            return

        combined = pd.concat(dfs).sort_index()
        logger.info("Retraining on %d bars from %d symbols...", len(combined), len(dfs))

        try:
            best_model, results = walk_forward_train(combined, self.cfg)
            avg_auc = sum(r.test_metrics.get("auc", 0) for r in results) / max(len(results), 1)
            self._store.save(best_model, metadata={
                "description": f"Auto-retrain bar {self._total_bars}",
                "avg_test_auc": round(avg_auc, 4),
                "n_folds": len(results),
                "champion_status": "champion",
            })
            self._meta_model = best_model
            logger.info("Retrain complete. avg_test_auc=%.4f", avg_auc)
        except Exception as exc:
            logger.error("Retrain failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("=== LiveRunner shutdown ===")
        if isinstance(self.broker, PaperBroker):
            summary = self.broker.summary()
            print("\n" + "=" * 50)
            print("  PAPER TRADING SUMMARY")
            print("=" * 50)
            print(f"  Balance:      {summary['balance']:>12,.2f} USD")
            print(f"  Equity:       {summary['equity']:>12,.2f} USD")
            print(f"  Total PnL:    {summary['total_pnl']:>12,.4f} USD")
            print(f"  Trades:       {summary['n_trades']:>12d}")
            print(f"  Open:         {summary['n_open']:>12d}")
            print(f"  Win Rate:     {summary['win_rate']:>12.1%}")
            print(f"  Avg PnL:      {summary['avg_pnl']:>12.4f} USD")
            print("=" * 50 + "\n")
