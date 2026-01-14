from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_TOP_LEVEL = ["symbol", "timeframe", "timezone", "data", "strategy", "risk", "execution"]


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load and minimally validate YAML config.
    Returns a plain dict to keep v1 simple; can be migrated to dataclasses later.
    """
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

    # Minimal sanity checks
    if not isinstance(cfg["data"], dict) or "bars" not in cfg["data"]:
        raise ValueError("cfg.data must be a mapping with key 'bars'.")

    if cfg["strategy"]["ema_fast"] >= cfg["strategy"]["ema_slow"]:
        raise ValueError("strategy.ema_fast must be < strategy.ema_slow.")

    return cfg

