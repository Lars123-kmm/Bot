from __future__ import annotations

"""
LiveRunner — multi-symbol continuous trading loop.

Modes:
  paper   — simulates trades in memory, no API key needed
  live    — real orders via BinanceBroker (requires BINANCE_API_KEY/SECRET)

Features:
  - Live OHLCV from Binance Futures (public endpoint)
  - Full strategy pipeline per symbol each bar
  - Online learning (River) after each closed paper trade
  - Portfolio-level risk cap across all symbols
  - Multi-timeframe trend filter (higher-TF must agree with the signal)
  - Model-degradation detection → auto-retrain when win-rate collapses
  - Telegram notifications, CSV trade log, live HTTP dashboard
  - Auto-retraining via Walk-Forward + Champion-Challenger every N bars
"""

import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import pandas as pd

from src.data.binance_fetch import BinanceFetcher
from src.Execution.order_state import ActiveTrade
from src.features.data_prep import prepare_market_data
from src.features.indicators import compute_indicators
from src.features.ml_features import ML_FEATURE_COLS, build_ml_features
from src.ml.adaptive_features import AdaptiveFeatureSelector
from src.ml.automl import autotune_lgbm, merge_params
from src.ml.data_store import TradeDataStore
from src.ml.model_store import ModelStore
from src.ml.online_model import OnlineMetaFilter
from src.risk.guards import ConsecutiveLossGuard, DrawdownGuard
from src.risk.sizing import calculate_position, probability_scaled_position
from src.runner.dashboard import DashboardServer
from src.runner.notifier import Notifier
from src.runner.paper_broker import PaperBroker
from src.runner.trade_log import TradeLogger
from src.strategies.ema_atr import generate_signals

logger = logging.getLogger(__name__)

_TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


def _seconds_to_next_bar(timeframe: str) -> float:
    """Seconds until the next bar opens (aligned to wall-clock boundaries)."""
    bar_sec = _TIMEFRAME_SECONDS.get(timeframe, 3600)
    elapsed = time.time() % bar_sec
    return bar_sec - elapsed + 2  # +2 s buffer for Binance to close the candle


class _SymbolState:
    """Per-symbol mutable state."""

    def __init__(self, symbol: str, online_filter: OnlineMetaFilter):
        self.symbol = symbol
        self.online_filter = online_filter
        self.active_trade: Optional[ActiveTrade] = None
        self.current_risk: float = 0.0          # risk_quote of the open trade
        self.bars_since_retrain: int = 0
        self._entry_features: Optional[Dict[str, float]] = None
        self._entry_ticket: Optional[int] = None


class LiveRunner:
    """
    Runs a multi-symbol trading loop using live Binance data.

    Args:
        cfg:             Bot config dict (from default.yaml)
        mode:            "paper" or "live"
        symbols:         Binance Futures symbols, e.g. ["BTCUSDT", "ETHUSDT"]
        initial_capital: Starting balance for paper trading
        retrain_every:   Bars between Walk-Forward retrains (0 = disabled)
        report_dir:      Directory for CSV trade logs (optional)
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
        live_cfg = cfg.get("live", {})
        self.symbols: List[str] = symbols or live_cfg.get("symbols") or [cfg.get("symbol", "BTCUSDT")]
        self.timeframe: str = cfg.get("timeframe", "H1")
        self.retrain_every = retrain_every

        # --- Broker ---
        if mode == "paper":
            self.broker = PaperBroker(
                initial_balance=initial_capital,
                fee_rate=live_cfg.get("fee_rate", 0.0004),
                spread_pct=live_cfg.get("spread_pct", 0.0001),
            )
        else:
            from src.Execution.binance_broker import BinanceBroker
            self.broker = BinanceBroker(cfg)

        self.fetcher = BinanceFetcher()

        # --- Model ---
        self._store = ModelStore(cfg.get("ml", {}).get("model_path", "models/"))
        self._meta_model = self._load_model()

        # --- Per-symbol state ---
        self._states: Dict[str, _SymbolState] = {
            sym: _SymbolState(sym, OnlineMetaFilter(cfg)) for sym in self.symbols
        }

        # --- Guards ---
        self._dd_guard = DrawdownGuard(
            max_dd=cfg.get("risk", {}).get("max_total_drawdown", 0.10)
        )
        self._loss_guard = ConsecutiveLossGuard(
            max_losses=cfg.get("risk", {}).get("max_consecutive_losses", 5)
        )
        self._dd_tripped = False
        self._current_dd = 0.0

        # --- Portfolio risk ---
        self._max_portfolio_risk = cfg.get("risk", {}).get("max_portfolio_risk", 0.03)

        # --- Multi-timeframe filter ---
        mtf_cfg = cfg.get("mtf", {})
        self._mtf_enabled = bool(mtf_cfg.get("enabled", False))
        self._higher_tf = mtf_cfg.get("higher_timeframe", "H4")

        # --- Degradation detection ---
        self._degr_window = int(live_cfg.get("degradation_window", 20))
        self._degr_min_winrate = float(live_cfg.get("degradation_min_winrate", 0.35))
        self._recent_outcomes: Deque[int] = deque(maxlen=self._degr_window)

        # --- Notifications / CSV log ---
        self._notifier = Notifier(cfg)
        self._trade_logger = (
            TradeLogger(report_dir) if report_dir is not None else None
        )

        # --- Autonomous learning infrastructure ---
        automl_cfg = cfg.get("automl", {})
        store_path = automl_cfg.get("data_store_path", "data/trade_history.db")
        feat_path = automl_cfg.get("feature_db_path", "data/feature_importance.db")
        self._data_store = TradeDataStore(path=store_path)
        self._feature_selector = AdaptiveFeatureSelector(
            path=feat_path,
            drop_percentile=float(automl_cfg.get("drop_percentile", 0.10)),
            min_cycles_to_drop=int(automl_cfg.get("min_cycles_to_drop", 3)),
            recovery_cycles=int(automl_cfg.get("recovery_cycles", 2)),
        )
        self._automl_trials = int(automl_cfg.get("n_trials", 40))
        self._automl_min_samples = int(automl_cfg.get("min_samples", 50))

        # --- Dashboard ---
        dash_cfg = cfg.get("dashboard", {})
        self._dashboard: Optional[DashboardServer] = None
        if dash_cfg.get("enabled", False):
            self._dashboard = DashboardServer(
                self._dashboard_state, port=int(dash_cfg.get("port", 8080))
            )

        # --- Counters / day tracking ---
        self._total_bars = 0
        self._recent_closed: Deque[Dict[str, Any]] = deque(maxlen=15)
        self._last_summary_date = datetime.now(timezone.utc).date()

        logger.info(
            "LiveRunner init: mode=%s symbols=%s tf=%s retrain_every=%d mtf=%s",
            mode, self.symbols, self.timeframe, retrain_every, self._mtf_enabled,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("=== LiveRunner started (mode=%s) — Ctrl+C to stop ===", self.mode)
        if self._dashboard is not None:
            self._dashboard.start()
        self._notifier.info(
            f"Bot gestartet — Modus {self.mode}, Symbole {', '.join(self.symbols)}"
        )
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

        # Equity snapshot → drawdown guard
        acct = self.broker.get_account_info()
        allowed, dd = self._dd_guard.update(acct["equity"])
        self._current_dd = dd
        if not allowed and not self._dd_tripped:
            self._dd_tripped = True
            logger.warning("DRAWDOWN GUARD tripped at %.1f%% — no new trades.", dd * 100)
            self._notifier.guard_tripped(
                "Drawdown-Guard", f"Drawdown {dd:.1%} ≥ Limit {self._dd_guard.max_dd:.0%}"
            )

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error processing %s: %s", symbol, exc, exc_info=True)
                self._notifier.error(f"{symbol}: {exc}")

        # Auto-retrain on a fixed schedule
        if self.retrain_every > 0 and self._total_bars % self.retrain_every == 0:
            logger.info("Auto-retrain triggered (bar %d).", self._total_bars)
            self._retrain()

        self._maybe_send_daily_summary()

    def _process_symbol(self, symbol: str) -> None:
        state = self._states[symbol]
        state.bars_since_retrain += 1

        bars_needed = max(
            self.cfg.get("data", {}).get("bars", 500),
            int(self.cfg.get("ml", {}).get("train_bars", 2000)) + 100,
        )
        df = self.fetcher.fetch_ohlcv(symbol, self.timeframe, bars=bars_needed)
        if df is None or len(df) < 100:
            logger.warning("%s: not enough data — skipping.", symbol)
            return

        # Feed paper broker the latest bar → checks SL/TP
        last_raw = df.iloc[-1]
        if isinstance(self.broker, PaperBroker):
            closed = self.broker.update_bar(
                symbol, float(last_raw["open"]), float(last_raw["high"]),
                float(last_raw["low"]), float(last_raw["close"]),
            )
            for ticket in closed:
                self._on_position_closed(state, ticket)

        # Strategy pipeline
        df = prepare_market_data(df, self.cfg)
        df = compute_indicators(df, self.cfg)
        df = generate_signals(df, self.cfg)

        last = df.iloc[-1]
        signal = int(last.get("signal", 0))
        atr = float(last.get("atr", 0.0))
        close_price = float(last["close"])

        # Manage an open trade
        if state.active_trade is not None:
            self._update_trailing(state, last, atr)
            self._check_time_exit(state, last, df)

        # Open a new trade
        if signal != 0 and state.active_trade is None:
            if self._dd_tripped:
                return
            if not self._loss_guard.is_allowed():
                logger.warning("%s: loss guard active — skipping signal.", symbol)
                return

            # Multi-timeframe filter
            if self._mtf_enabled:
                htf_trend = self._higher_tf_trend(symbol)
                if htf_trend != 0 and htf_trend != signal:
                    logger.info(
                        "%s: signal %d blocked by %s trend %d",
                        symbol, signal, self._higher_tf, htf_trend,
                    )
                    return

            # ML filter
            ml_prob = self._ml_probability(df, symbol)
            min_prob = self.cfg.get("ml", {}).get("min_probability", 0.55)
            if ml_prob is not None and ml_prob < min_prob:
                logger.info("%s: ML filter blocked (prob=%.3f)", symbol, ml_prob)
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
        if atr <= 0:
            return

        risk_cfg = self.cfg.get("risk", {})
        strat_cfg = self.cfg.get("strategy", {})
        acct = self.broker.get_account_info()
        equity = acct["equity"]
        min_prob = self.cfg.get("ml", {}).get("min_probability", 0.55)

        if ml_prob is not None and ml_prob >= min_prob:
            pos = probability_scaled_position(
                capital=equity, entry_price=price, atr=atr,
                risk_pct=risk_cfg.get("account_risk_per_trade", 0.01),
                meta_prob=ml_prob, min_prob=min_prob,
                stop_mult=strat_cfg.get("atr_stop_mult", 2.0),
                tp_mult=strat_cfg.get("atr_tp_mult", 3.0),
                direction=direction,
            )
        else:
            pos = calculate_position(
                capital=equity, entry_price=price, atr=atr,
                risk_pct=risk_cfg.get("account_risk_per_trade", 0.01),
                stop_mult=strat_cfg.get("atr_stop_mult", 2.0),
                tp_mult=strat_cfg.get("atr_tp_mult", 3.0),
                direction=direction,
            )

        if pos["size"] <= 0:
            return

        # --- Portfolio-level risk cap ---
        open_risk = sum(
            s.current_risk for s in self._states.values() if s.active_trade is not None
        )
        budget = self._max_portfolio_risk * equity
        new_risk = pos["risk_quote"]
        if open_risk + new_risk > budget:
            available = budget - open_risk
            if available <= new_risk * 0.10:
                logger.info(
                    "%s: portfolio risk limit reached (%.2f/%.2f) — skipping.",
                    symbol, open_risk, budget,
                )
                return
            scale = available / new_risk
            pos["size"] = round(pos["size"] * scale, 6)
            pos["risk_quote"] = round(new_risk * scale, 2)
            logger.info("%s: position scaled to %.0f%% (portfolio cap).", symbol, scale * 100)

        ticket = self.broker.place_market_order(
            symbol=symbol, direction=direction, size=pos["size"],
            sl=pos["stop"], tp=pos["take_profit"],
        )

        state.active_trade = ActiveTrade.from_entry(
            ticket=ticket, direction=direction, symbol=symbol,
            entry_price=price, size=pos["size"],
            stop=pos["stop"], take_profit=pos["take_profit"], atr_at_entry=atr,
        )
        state.current_risk = pos["risk_quote"]
        state._entry_ticket = ticket
        state._entry_features = self._extract_features(last_bar)

        # Persist entry features — outcome will be filled in on close
        try:
            self._data_store.log_entry(
                symbol=symbol,
                bar_time=last_bar.name if hasattr(last_bar, "name") else "",
                direction=direction,
                features=state._entry_features,
                ticket=ticket,
            )
        except Exception as exc:
            logger.warning("DataStore.log_entry failed: %s", exc)

        logger.info(
            "%s OPEN dir=%d size=%.6f @ %.5f sl=%.5f tp=%.5f prob=%s",
            symbol, direction, pos["size"], price, pos["stop"], pos["take_profit"],
            f"{ml_prob:.3f}" if ml_prob is not None else "n/a",
        )
        self._notifier.trade_opened(symbol, direction, pos["size"], price, ml_prob)

    def _on_position_closed(self, state: _SymbolState, ticket: int) -> None:
        trade = self.broker.get_closed_trade(ticket)
        if trade is None:
            return

        pnl = trade.get("pnl", 0.0) or 0.0
        outcome = 1 if pnl > 0 else 0

        self._loss_guard.record(pnl)
        self._recent_outcomes.append(outcome)
        self._recent_closed.append(trade)

        if state._entry_features:
            try:
                state.online_filter.partial_fit(state._entry_features, outcome)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Online filter update failed: %s", exc)

        # Write realized outcome back to persistent store (feeds next HPO run)
        if state._entry_ticket is not None:
            try:
                self._data_store.update_outcome(state._entry_ticket, outcome, pnl)
            except Exception as exc:
                logger.warning("DataStore.update_outcome failed: %s", exc)

        if self._trade_logger is not None:
            self._trade_logger.log(trade)

        state.active_trade = None
        state.current_risk = 0.0
        state._entry_features = None
        state._entry_ticket = None

        logger.info(
            "%s CLOSED ticket=%d pnl=%.4f (%s)",
            state.symbol, ticket, pnl, trade.get("close_reason", "?"),
        )
        self._notifier.trade_closed(state.symbol, pnl, trade.get("close_reason", "?"))

        if not self._loss_guard.is_allowed():
            self._notifier.guard_tripped(
                "Loss-Guard",
                f"{self._loss_guard.consecutive_losses} Verluste in Folge",
            )

        self._check_degradation()

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
        try:
            entry_idx = df.index.searchsorted(pd.Timestamp(trade.open_time))
            bars_held = len(df) - 1 - entry_idx
        except Exception:  # noqa: BLE001
            return
        if bars_held >= max_bars:
            logger.info("%s: time exit after %d bars", state.symbol, bars_held)
            self.broker.close_position(
                trade.ticket, state.symbol, trade.size, trade.direction
            )
            if isinstance(self.broker, PaperBroker):
                self._on_position_closed(state, trade.ticket)
            else:
                state.active_trade = None
                state.current_risk = 0.0

    # ------------------------------------------------------------------
    # ML / filters
    # ------------------------------------------------------------------

    def _extract_features(self, last_bar: pd.Series) -> Dict[str, float]:
        try:
            ml_feats = build_ml_features(last_bar.to_frame().T)
            return {
                col: float(ml_feats[col].iloc[-1]) if col in ml_feats.columns else 0.0
                for col in ML_FEATURE_COLS
            }
        except Exception:  # noqa: BLE001
            return {}

    def _ml_probability(self, df: pd.DataFrame, symbol: str) -> Optional[float]:
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
            feat_dict = {c: float(last_features[c].iloc[0]) for c in ML_FEATURE_COLS}
            state = self._states.get(symbol)
            if state is not None:
                base_prob = state.online_filter.blend(base_prob, feat_dict)
            return base_prob
        except Exception as exc:  # noqa: BLE001
            logger.warning("ML probability failed: %s", exc)
            return None

    def _higher_tf_trend(self, symbol: str) -> int:
        """Returns 1 (up), -1 (down) or 0 (unknown) from the higher timeframe EMAs."""
        try:
            df = self.fetcher.fetch_ohlcv(symbol, self._higher_tf, bars=200)
            if df is None or len(df) < 60:
                return 0
            strat = self.cfg.get("strategy", {})
            ema_fast = df["close"].ewm(span=strat.get("ema_fast", 20), adjust=False).mean()
            ema_slow = df["close"].ewm(span=strat.get("ema_slow", 50), adjust=False).mean()
            if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                return 1
            if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                return -1
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Higher-TF trend fetch failed for %s: %s", symbol, exc)
            return 0

    def _load_model(self):
        try:
            model, info = self._store.load_champion(self.cfg)
            logger.info("Loaded champion model v%d", info.get("version", 0))
            return model
        except FileNotFoundError:
            logger.warning("No champion model — ML filter disabled. Run --mode train first.")
            return None

    # ------------------------------------------------------------------
    # Degradation detection + retraining
    # ------------------------------------------------------------------

    def _check_degradation(self) -> None:
        """Triggers a retrain when recent win-rate collapses below the threshold."""
        if len(self._recent_outcomes) < self._degr_window:
            return
        win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
        if win_rate < self._degr_min_winrate:
            logger.warning(
                "Model degradation: win-rate %.1f%% over last %d trades < %.0f%% — retraining.",
                win_rate * 100, self._degr_window, self._degr_min_winrate * 100,
            )
            self._notifier.info(
                f"Modell-Degradation erkannt (Win-Rate {win_rate:.0%}) — Retraining startet."
            )
            self._recent_outcomes.clear()
            self._retrain()

    def _retrain(self) -> None:
        from src.ml.walk_forward import walk_forward_train

        # ── Step 1: Fetch market data ──────────────────────────────────
        dfs = []
        for sym in self.symbols:
            try:
                df = self.fetcher.fetch_ohlcv(sym, self.timeframe, bars=5000)
                if df is not None and len(df) > 200:
                    df = prepare_market_data(df, self.cfg)
                    df = compute_indicators(df, self.cfg)
                    dfs.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Retrain fetch failed for %s: %s", sym, exc)

        if not dfs:
            logger.warning("Retrain skipped — no data.")
            return

        combined = pd.concat(dfs).sort_index()

        # ── Step 2: AutoML — tune LightGBM on real trade outcomes ─────
        # Real-trade data has better signal than synthetic triple-barrier
        # labels, so we use it for HPO when enough samples exist.
        tuned_cfg = self.cfg
        real_data = self._data_store.get_training_data(
            min_samples=self._automl_min_samples
        )
        if real_data is not None:
            X_real, y_real = real_data
            logger.info(
                "AutoML HPO on %d real trades (%d trials)...",
                len(X_real), self._automl_trials,
            )
            best_params = autotune_lgbm(
                X_real, y_real, n_trials=self._automl_trials
            )
            tuned_cfg = merge_params(self.cfg, best_params)
            logger.info("AutoML complete — injecting best params: %s", best_params)
            self._notifier.info(
                f"AutoML: beste LightGBM-Params gefunden"
                f" (leaves={best_params.get('lgbm_num_leaves')},"
                f" lr={best_params.get('lgbm_learning_rate'):.4f})"
            )
        else:
            logger.info("AutoML skipped — not enough real-trade samples yet.")

        # ── Step 3: Inject active features from AdaptiveFeatureSelector ─
        active_feats = self._feature_selector.get_active_features()
        tuned_cfg.setdefault("ml", {})["active_features"] = active_feats
        if len(active_feats) < len(ML_FEATURE_COLS):
            logger.info(
                "Using %d/%d active features (dropped: %s)",
                len(active_feats), len(ML_FEATURE_COLS),
                self._feature_selector.dropped_features(),
            )

        # ── Step 4: Walk-Forward train with tuned config ───────────────
        logger.info(
            "Walk-forward retrain: %d bars, %d symbols...",
            len(combined), len(dfs),
        )
        try:
            best_model, results = walk_forward_train(combined, tuned_cfg)
            avg_auc = sum(
                r.test_metrics.get("auc", 0) for r in results
            ) / max(len(results), 1)

            # ── Step 5: Update adaptive feature selector ───────────────
            try:
                self._feature_selector.update(best_model)
            except Exception as exc:
                logger.warning("AdaptiveFeatureSelector.update failed: %s", exc)

            self._store.save(best_model, metadata={
                "description": f"Auto-retrain bar {self._total_bars}",
                "avg_test_auc": round(avg_auc, 4),
                "n_folds": len(results),
                "champion_status": "champion",
                "active_features": len(active_feats),
                "automl_used": real_data is not None,
            })
            self._meta_model = best_model
            logger.info(
                "Retrain complete: avg_test_auc=%.4f, active_features=%d/%d",
                avg_auc, len(active_feats), len(ML_FEATURE_COLS),
            )
            self._notifier.info(
                f"Retraining fertig — AUC {avg_auc:.3f},"
                f" {len(active_feats)} aktive Features"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrain failed: %s", exc, exc_info=True)
            self._notifier.error(f"Retraining fehlgeschlagen: {exc}")

    # ------------------------------------------------------------------
    # Dashboard / reporting
    # ------------------------------------------------------------------

    def _dashboard_state(self) -> Dict[str, Any]:
        acct = self.broker.get_account_info()
        summary = (
            self.broker.summary() if isinstance(self.broker, PaperBroker)
            else {"n_trades": 0, "win_rate": 0.0}
        )
        return {
            "mode": self.mode,
            "symbols_list": self.symbols,
            "total_bars": self._total_bars,
            "account": acct,
            "drawdown": round(self._current_dd, 4),
            "n_trades": summary.get("n_trades", 0),
            "win_rate": summary.get("win_rate", 0.0),
            "positions": self.broker.get_open_positions(),
            "recent_trades": list(self._recent_closed),
            "autonomous_learning": {
                "data_store": self._data_store.stats(),
                "active_features": len(self._feature_selector.get_active_features()),
                "total_features": len(ML_FEATURE_COLS),
                "dropped_features": self._feature_selector.dropped_features(),
            },
        }

    def _maybe_send_daily_summary(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today == self._last_summary_date:
            return
        self._last_summary_date = today
        if isinstance(self.broker, PaperBroker):
            self._notifier.daily_summary(self.broker.summary())

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("=== LiveRunner shutdown ===")
        if self._dashboard is not None:
            self._dashboard.stop()
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
            self._notifier.daily_summary(summary)
        self._notifier.info("Bot gestoppt.")
