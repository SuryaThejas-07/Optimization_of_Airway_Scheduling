from __future__ import annotations

import logging
from typing import Optional


_LOGGER_INITIALIZED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGER_INITIALIZED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name or "agno_rs_plus")
