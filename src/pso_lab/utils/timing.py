"""Timing context managers and helpers for lightweight instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Iterator


@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    start = perf_counter()

    def elapsed() -> float:
        return perf_counter() - start

    yield elapsed
