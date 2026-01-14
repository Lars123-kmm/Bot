# src/common/logger.py
import logging

def setup_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # idempotent

    logger.setLevel(level)

    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    handler.setFormatter(fmt)

    logger.addHandler(handler)
    logger.propagate = False
    return logger
