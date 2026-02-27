"""Verbosity-aware logging for Project Fairy Queen.

Three levels:
  LOW    – critical errors and final output paths only
  MEDIUM – progress updates, experiment start/completion, key metrics
  HIGH   – step-by-step detail, circuit depth, gate counts, intermediate results
"""

from __future__ import annotations

import logging
import sys
from enum import IntEnum

_LOGGER_NAME = "fairy_queen"


class Verbosity(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


_LEVEL_MAP = {
    Verbosity.LOW: logging.WARNING,
    Verbosity.MEDIUM: logging.INFO,
    Verbosity.HIGH: logging.DEBUG,
}


def setup_logging(verbosity: Verbosity | str = Verbosity.MEDIUM) -> logging.Logger:
    """Configure the project-wide logger and return it."""
    if isinstance(verbosity, str):
        verbosity = Verbosity[verbosity.upper()]

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(_LEVEL_MAP[verbosity])
        fmt = logging.Formatter(
            "[%(levelname)s] %(message)s"
            if verbosity != Verbosity.HIGH
            else "[%(levelname)s %(name)s:%(lineno)d] %(message)s"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    else:
        for h in logger.handlers:
            h.setLevel(_LEVEL_MAP[verbosity])

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)
