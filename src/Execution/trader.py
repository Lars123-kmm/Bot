from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5

from src.Execution.mt5_broker import MT5Broker
from src.Execution.order_state import ActiveTrade
from src.common.types import ClosedTrade
from src.data.mt5._fetch import MT5FetchConfig, fetch_rates, mt5_initialize, mt5_shutdown
from src.features.indicators import compute_indicators
from src.risk.guards import ConsecutiveLossGuard, DrawdownGuard
from src.risk.sizing import calculate_position
from src.strategies.ema_atr import generate_signals

logger = logging.getLogger(__name__)

_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class LiveTrader:
    """
    Orchestriert den gesamten Live-Trading-Loop.

    Ablauf je Bar:
      1. DrawdownGuard und ConsecutiveLossGuard prüfen
      2. Neue Candles von MT5 fetchen
      3. Indikatoren + Signale auf letzter geschlossener Bar berechnen
      4. Offene Positionen auf Trailing Stop prüfen
      5. Signal vorhanden? → Position eröffnen
      6. Am Tagesende: Daily Report erzeugen
      7. Bis zur nächsten Bar warten
    """

    def __init__(self, cfg: Dict[str, Any], report_dir: Optional[Path] = None):
        self.cfg = cfg
        self.symbol = cfg["symbol"]
        self.broker = MT5Broker(magic=cfg["execution"]["magic_number"])
        self.dd_guard = DrawdownGuard(max_dd=cfg["risk"]["max_total_drawdown"])
        self.loss_guard = ConsecutiveLossGuard(max_losses=cfg["risk"]["max_consecutive_losses"])
        self.active_trade: Optional[ActiveTrade] = None
        self._day_trades: List[ClosedTrade] = []
        self._equity_day_start: float = 0.0
        self._last_report_date: Optional[datetime] = None

        self._report_dir: Optional[Path] = report_dir
        if report_dir:
            from src.reporting.daily_report import DailyReporter
            self._reporter = DailyReporter(report_dir)
        else:
            self._reporter = None

        tf_str = cfg["timeframe"]
        if tf_str not in _TIMEFRAME_MAP:
            raise ValueError(f"Unbekannter Timeframe: {tf_str}. Erlaubt: {list(_TIMEFRAME_MAP)}")
        self._mt5_tf = _TIMEFRAME_MAP[tf_str]

    def run(self) -> None:
        """Startet den Trading-Loop. Läuft bis zum Programmabbruch oder Circuit-Breaker."""
        mt5_initialize()
        logger.info("LiveTrader gestartet: symbol=%s tf=%s mode=%s",
                    self.symbol, self.cfg["timeframe"], self.cfg["execution"]["mode"])

        try:
            account = self.broker.get_account_info()
            self.dd_guard.peak = account["balance"]
            self._equity_day_start = account["equity"]
            self._last_report_date = datetime.now(timezone.utc).date()

            while True:
                self._tick()
                self._wait_for_next_bar()

        except KeyboardInterrupt:
            logger.info("LiveTrader gestoppt (KeyboardInterrupt).")
        finally:
            mt5_shutdown()

    def _tick(self) -> None:
        """Führt einen vollständigen Bar-Zyklus aus."""
        fetch_cfg = MT5FetchConfig(
            symbol=self.symbol,
            timeframe=self._mt5_tf,
            n_bars=self.cfg["data"]["bars"],
        )
        df = fetch_rates(fetch_cfg)
        df = compute_indicators(df, self.cfg)
        df = generate_signals(df, self.cfg)

        account = self.broker.get_account_info()
        allowed, current_dd = self.dd_guard.update(account["equity"])

        if not allowed:
            logger.warning("CIRCUIT BREAKER: Drawdown %.1f%% >= Limit %.1f%%. Bot pausiert.",
                           current_dd * 100, self.cfg["risk"]["max_total_drawdown"] * 100)
            return

        if not self.loss_guard.is_allowed():
            logger.warning("LOSS GUARD: %d aufeinanderfolgende Verluste. Bot pausiert.",
                           self.loss_guard.consecutive_losses)
            return

        # Trailing Stop für offene Position aktualisieren
        if self.active_trade is not None:
            self._update_trailing(df)

        # Signal der letzten abgeschlossenen Bar (index -1, shift(1) bereits angewendet)
        last = df.iloc[-1]
        signal = int(last["signal"])
        atr = float(last["atr"])

        if self.active_trade is None and signal != 0:
            self._open_position(signal, float(last["close"]), atr)

        # Tagesabschluss prüfen
        today = datetime.now(timezone.utc).date()
        if self._last_report_date is not None and today > self._last_report_date:
            self._end_of_day_report(self._last_report_date, account)
            self._last_report_date = today
            self._equity_day_start = account["equity"]
            self._day_trades.clear()

    def _update_trailing(self, df: Any) -> None:
        last = df.iloc[-1]
        s = self.cfg["strategy"]
        new_stop = self.active_trade.update_trailing(
            current_high=float(last["high"]),
            current_low=float(last["low"]),
            current_atr=float(last["atr"]),
            activate_mult=s["trailing_activate_atr"],
            distance_mult=s["trailing_distance_atr"],
        )
        if new_stop is not None:
            success = self.broker.modify_stop(self.active_trade.ticket, new_stop)
            if success:
                logger.info("Trailing Stop auf %.5f gesetzt (ticket=%d)",
                            new_stop, self.active_trade.ticket)

    def _open_position(self, signal: int, price: float, atr: float) -> None:
        account = self.broker.get_account_info()
        pos_info = calculate_position(
            capital=account["equity"],
            entry_price=price,
            atr=atr,
            risk_pct=self.cfg["risk"]["account_risk_per_trade"],
            stop_mult=self.cfg["strategy"]["atr_stop_mult"],
            tp_mult=self.cfg["strategy"]["atr_tp_mult"],
            direction=signal,
        )

        ticket = self.broker.place_market_order(
            symbol=self.symbol,
            direction=signal,
            size=pos_info["size"],
            sl=pos_info["stop"],
            tp=pos_info["take_profit"],
        )

        self.active_trade = ActiveTrade.from_entry(
            ticket=ticket,
            direction=signal,
            symbol=self.symbol,
            entry_price=price,
            size=pos_info["size"],
            stop=pos_info["stop"],
            take_profit=pos_info["take_profit"],
            atr_at_entry=atr,
        )
        logger.info("Position eröffnet: %s dir=%d size=%.6f sl=%.5f tp=%.5f",
                    self.symbol, signal, pos_info["size"], pos_info["stop"], pos_info["take_profit"])

    def _end_of_day_report(self, report_date: Any, account: Dict[str, float]) -> None:
        if self._reporter is None:
            return
        try:
            path = self._reporter.generate(
                date=report_date,
                trades=list(self._day_trades),
                equity_start=self._equity_day_start,
                equity_end=account["equity"],
                drawdown_guard=self.dd_guard,
                loss_guard=self.loss_guard,
            )
            logger.info("Daily Report gespeichert: %s", path)
        except Exception as exc:
            logger.error("Daily Report fehlgeschlagen: %s", exc)

    def _wait_for_next_bar(self) -> None:
        """Einfaches Sleep bis zur nächsten vollen Bar."""
        tf_seconds = {
            "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
            "H1": 3600, "H4": 14400, "D1": 86400,
        }
        seconds = tf_seconds.get(self.cfg["timeframe"], 3600)
        logger.debug("Warte %ds bis zur nächsten Bar.", seconds)
        time.sleep(seconds)
