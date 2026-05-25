# src/common/logger.py
import logging
from typing import Union


def setup_logger(
    name: str = "root",
    level: Union[int, str] = logging.INFO,
) -> logging.Logger:
    """
    Konfiguriert einen Logger.
    name='root' konfiguriert den Root-Logger (empfohlen für main.py).
    level kann int (logging.INFO) oder String ("INFO") sein.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    target = logging.getLogger(None if name == "root" else name)
    if target.handlers:
        target.setLevel(level)
        return target

    target.setLevel(level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    target.addHandler(handler)

    if name != "root":
        target.propagate = False

    return target
