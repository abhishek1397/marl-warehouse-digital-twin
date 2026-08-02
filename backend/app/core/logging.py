"""Structured logging setup for FastAPI application."""

import logging
import sys

def setup_logging() -> logging.Logger:
    """Configures structured stream logging for FastAPI backend."""
    logger = logging.getLogger("warehouse_backend")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logging()
