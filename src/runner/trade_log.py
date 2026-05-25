from __future__ import annotations

"""CSV logging of closed paper/live trades — one file per UTC day."""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Appends every closed trade to reports/paper_trades_YYYY-MM-DD.csv.
    The header is written automatically when a new file is created.
    """

    FIELDS = [
        "close_time", "symbol", "direction", "size",
        "entry_price", "close_price", "pnl", "close_reason", "open_time",
    ]

    def __init__(self, report_dir: str | Path = "reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.report_dir / f"paper_trades_{day}.csv"

    def log(self, trade: Dict[str, Any]) -> None:
        """Append one closed-trade dict (as returned by PaperBroker.get_closed_trade)."""
        path = self._path()
        write_header = not path.exists()
        try:
            with open(path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.FIELDS, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                row = {k: trade.get(k, "") for k in self.FIELDS}
                # normalise direction to readable text
                row["direction"] = "long" if trade.get("direction", 1) == 1 else "short"
                writer.writerow(row)
        except Exception as exc:
            logger.warning("TradeLogger write failed: %s", exc)
