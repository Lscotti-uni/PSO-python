"""Structured logging helpers for experiment execution."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("pso_lab")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_kv(logger: logging.Logger | None, level: int, event: str, **fields: object) -> None:
    if logger is None:
        return
    parts = [f"event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.log(level, " | ".join(parts))
