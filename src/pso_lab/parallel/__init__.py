"""Factory helpers for interchangeable fitness-evaluation backends."""

from __future__ import annotations

from pso_lab.parallel.base import EvaluationStats, FitnessEvaluator
from pso_lab.parallel.processes import ProcessPoolEvaluator
from pso_lab.parallel.sequential import SequentialEvaluator
from pso_lab.parallel.threaded import ThreadPoolEvaluator


def build_evaluator(
    strategy: str,
    *,
    workers: int | None = None,
    batch_size: int = 1,
) -> FitnessEvaluator:
    normalized = strategy.strip().lower()
    if normalized in {"sequential", "seq", "v0"}:
        return SequentialEvaluator(batch_size=batch_size)
    if normalized in {"thread", "threading", "threads", "v1"}:
        return ThreadPoolEvaluator(workers=workers, batch_size=batch_size)
    if normalized in {"process", "multiprocessing", "processes", "v2"}:
        return ProcessPoolEvaluator(workers=workers, batch_size=batch_size)
    raise KeyError(f"Unknown evaluation strategy '{strategy}'.")


__all__ = [
    "EvaluationStats",
    "FitnessEvaluator",
    "ProcessPoolEvaluator",
    "SequentialEvaluator",
    "ThreadPoolEvaluator",
    "build_evaluator",
]
