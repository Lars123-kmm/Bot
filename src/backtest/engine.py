from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.backtest.costs import CostModel
from src.common.types import ClosedTrade, PositionState
from src.risk.sizing import calculate_position


def run_backtest(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    initial_capital: float = 10_000.0,
    cost_model: Optional[CostModel] = None,
) -> Tuple[pd.DataFrame, List[ClosedTrade], Dict[str, float]]:
    """
    Bar-für-Bar Backtest der Multi-Filter EMA-Crossover-Strategie.

    Erwartet DataFrame mit Spalten: open, high, low, close, atr, signal
    (erzeugt durch compute_indicators + generate_signals).

    Exit-Priorität:
      1. Stop Loss  (SL)
      2. Take Profit (TP)
      3. Trailing Stop (wenn aktiviert)
      4. Signal-Exit (gegenläufiges Signal)
      5. Time-Exit (max. time_exit_bars Bars)

    Returns:
        df_result:  Original-DataFrame mit Equity-Spalte
        trades:     Liste aller abgeschlossenen Trades
        metrics:    Performance-Metriken
    """
    if cost_model is None:
        cost_model = CostModel()

    s = cfg["strategy"]
    risk_pct = cfg["risk"]["account_risk_per_trade"]
    stop_mult = s["atr_stop_mult"]
    tp_mult = s["atr_tp_mult"]
    trail_activate = s["trailing_activate_atr"]
    trail_dist = s["trailing_distance_atr"]
    time_exit_bars = s["time_exit_bars"]

    cap = initial_capital
    position = PositionState.FLAT
    entry_price = 0.0
    stop = 0.0
    take_profit = 0.0
    size = 0.0
    entry_bar = 0
    highest = 0.0
    lowest = 0.0
    trailing_active = False

    trades: List[ClosedTrade] = []
    equity: List[float] = []

    for i in range(len(df)):
        row = df.iloc[i]
        atr = float(row["atr"])

        # --- Trailing Stop aktualisieren ---
        if position == PositionState.LONG:
            highest = max(highest, float(row["high"]))
            if highest - entry_price >= trail_activate * atr:
                new_stop = highest - trail_dist * atr
                if new_stop > stop:
                    stop = new_stop
                    trailing_active = True

        elif position == PositionState.SHORT:
            lowest = min(lowest, float(row["low"]))
            if entry_price - lowest >= trail_activate * atr:
                new_stop = lowest + trail_dist * atr
                if new_stop < stop:
                    stop = new_stop
                    trailing_active = True

        # --- Exit-Logik ---
        if position != PositionState.FLAT:
            exit_now = False
            exit_price = float(row["close"])
            reason = ""
            bars_held = i - entry_bar

            if position == PositionState.LONG:
                if float(row["low"]) <= stop:
                    exit_price = stop
                    exit_now = True
                    reason = "Trailing" if trailing_active else "SL"
                elif float(row["high"]) >= take_profit:
                    exit_price = take_profit
                    exit_now = True
                    reason = "TP"
                elif int(row["signal"]) == -1:
                    exit_now = True
                    reason = "Signal"
                elif bars_held >= time_exit_bars:
                    exit_now = True
                    reason = "Time"

            else:  # SHORT
                if float(row["high"]) >= stop:
                    exit_price = stop
                    exit_now = True
                    reason = "Trailing" if trailing_active else "SL"
                elif float(row["low"]) <= take_profit:
                    exit_price = take_profit
                    exit_now = True
                    reason = "TP"
                elif int(row["signal"]) == 1:
                    exit_now = True
                    reason = "Signal"
                elif bars_held >= time_exit_bars:
                    exit_now = True
                    reason = "Time"

            if exit_now:
                raw_pnl = (exit_price - entry_price) * size * position.value
                costs = cost_model.round_trip(size * entry_price, size * exit_price)
                pnl = raw_pnl - costs
                cap += pnl

                trades.append(ClosedTrade(
                    direction=position.value,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=size,
                    pnl=pnl,
                    reason=reason,
                    bars_held=bars_held,
                ))
                position = PositionState.FLAT
                trailing_active = False

        # --- Entry-Logik ---
        if position == PositionState.FLAT and int(row["signal"]) != 0:
            direction = int(row["signal"])
            pos_info = calculate_position(
                capital=cap,
                entry_price=float(row["close"]),
                atr=atr,
                risk_pct=risk_pct,
                stop_mult=stop_mult,
                tp_mult=tp_mult,
                direction=direction,
            )
            position = PositionState(direction)
            entry_price = float(row["close"])
            size = pos_info["size"]
            stop = pos_info["stop"]
            take_profit = pos_info["take_profit"]
            entry_bar = i
            highest = entry_price
            lowest = entry_price
            trailing_active = False
            cap -= cost_model.apply(size * entry_price)

        # Equity inkl. unrealisiertem P&L
        unrealized = (float(row["close"]) - entry_price) * size * position.value if position != PositionState.FLAT else 0.0
        equity.append(cap + unrealized)

    df_result = df.copy()
    df_result["equity"] = equity

    metrics = compute_metrics(df_result["equity"], trades, initial_capital)
    return df_result, trades, metrics


def compute_metrics(
    equity: pd.Series,
    trades: List[ClosedTrade],
    initial_capital: float,
) -> Dict[str, float]:
    """
    Berechnet alle Performance-Metriken aus dem PDF:
    Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor, Expectancy, Recovery Factor.
    """
    returns = equity.pct_change().dropna()

    # Annualisierungsfaktor für 1h-Daten
    ann = math.sqrt(24 * 365)

    std = returns.std()
    sharpe = (returns.mean() / std * ann) if std > 0 else 0.0

    downside_std = returns[returns < 0].std()
    sortino = (returns.mean() / downside_std * ann) if downside_std > 0 else math.inf

    rolling_max = equity.cummax()
    dd_series = (equity - rolling_max) / rolling_max
    max_dd = float(dd_series.min())

    n_years = len(returns) / (24 * 365)
    if n_years > 0 and equity.iloc[0] > 0:
        annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    else:
        annual_return = 0.0

    calmar = (annual_return / abs(max_dd)) if max_dd != 0 else math.inf

    total_return = (equity.iloc[-1] / initial_capital - 1) if initial_capital > 0 else 0.0
    recovery = (total_return / abs(max_dd)) if max_dd != 0 else math.inf

    n = len(trades)
    if n == 0:
        return {
            "total_return_pct": round(total_return * 100, 2),
            "annual_return_pct": round(annual_return * 100, 2),
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "recovery_factor": 0.0,
            "n_trades": 0,
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_rate = len(wins) / n
    avg_win = mean(t.pnl for t in wins) if wins else 0.0
    avg_loss = abs(mean(t.pnl for t in losses)) if losses else 0.0

    sum_wins = sum(t.pnl for t in wins)
    sum_losses = abs(sum(t.pnl for t in losses))
    profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else math.inf

    expectancy = avg_win * win_rate - avg_loss * (1.0 - win_rate)

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 3),
        "expectancy": round(expectancy, 2),
        "recovery_factor": round(recovery, 3),
        "n_trades": n,
        "avg_bars_held": round(mean(t.bars_held for t in trades), 1),
    }
