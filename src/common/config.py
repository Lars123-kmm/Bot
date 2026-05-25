from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_TOP_LEVEL = ["symbol", "timeframe", "timezone", "data", "strategy", "risk", "execution"]

REQUIRED_STRATEGY_KEYS = [
    "ema_fast", "ema_slow",
    "adx_period", "adx_threshold",
    "rsi_period", "rsi_long_min", "rsi_long_max", "rsi_short_min", "rsi_short_max",
    "volume_ma_period", "volume_ratio_min",
    "extended_max_pct",
    "atr_period", "atr_stop_mult", "atr_tp_mult",
    "trailing_activate_atr", "trailing_distance_atr",
    "time_exit_bars",
]

REQUIRED_RISK_KEYS = [
    "account_risk_per_trade",
    "max_total_drawdown",
    "max_daily_drawdown",
    "max_consecutive_losses",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a YAML mapping (dict).")

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"Missing required top-level keys: {missing}")

    if not isinstance(cfg["data"], dict) or "bars" not in cfg["data"]:
        raise ValueError("cfg.data must be a mapping with key 'bars'.")

    strat = cfg["strategy"]
    missing_strat = [k for k in REQUIRED_STRATEGY_KEYS if k not in strat]
    if missing_strat:
        raise ValueError(f"Missing strategy keys: {missing_strat}")

    if strat["ema_fast"] >= strat["ema_slow"]:
        raise ValueError("strategy.ema_fast must be < strategy.ema_slow.")

    if not (0 <= strat["rsi_long_min"] < strat["rsi_long_max"] <= 100):
        raise ValueError("Invalid rsi_long range.")
    if not (0 <= strat["rsi_short_min"] < strat["rsi_short_max"] <= 100):
        raise ValueError("Invalid rsi_short range.")

    if strat["atr_stop_mult"] <= 0:
        raise ValueError("atr_stop_mult must be > 0.")
    if strat["atr_tp_mult"] <= 0:
        raise ValueError("atr_tp_mult must be > 0.")
    if strat["trailing_activate_atr"] <= 0:
        raise ValueError("trailing_activate_atr must be > 0.")

    risk = cfg["risk"]
    missing_risk = [k for k in REQUIRED_RISK_KEYS if k not in risk]
    if missing_risk:
        raise ValueError(f"Missing risk keys: {missing_risk}")

    if not (0 < risk["account_risk_per_trade"] <= 0.05):
        raise ValueError("account_risk_per_trade must be between 0 and 5%.")
    if not (0 < risk["max_total_drawdown"] <= 0.50):
        raise ValueError("max_total_drawdown must be between 0 and 50%.")

    return cfg
