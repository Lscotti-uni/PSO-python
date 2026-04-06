"""Factory helpers for interchangeable fitness-evaluation backends."""

from __future__ import annotations

from pso_lab.parallel.asyncio_eval import AsyncioEvaluator
from pso_lab.parallel.base import EvaluationStats, FitnessEvaluator
from pso_lab.parallel.joblib_eval import JoblibEvaluator
from pso_lab.parallel.processes import ProcessPoolEvaluator
from pso_lab.parallel.sequential import SequentialEvaluator
from pso_lab.parallel.threaded import ThreadPoolEvaluator
from pso_lab.parallel.vectorized import VectorizedEvaluator


def build_evaluator(
    strategy: str,
    *,
    workers: int | None = None,
    batch_size: int = 1,
    joblib_backend: str = "loky",
    seed: int | None = None,
    async_latency_ms: float = 0.0,
    async_jitter_ms: float = 0.0,
) -> FitnessEvaluator:
    normalized = strategy.strip().lower()
    if normalized in {"sequential", "seq", "v0"}:
        return SequentialEvaluator(batch_size=batch_size)
    if normalized in {"thread", "threading", "threads", "v1"}:
        return ThreadPoolEvaluator(workers=workers, batch_size=batch_size)
    if normalized in {"process", "multiprocessing", "processes", "v2"}:
        return ProcessPoolEvaluator(workers=workers, batch_size=batch_size)
    if normalized in {"asyncio", "async", "v3"}:
        return AsyncioEvaluator(
            latency_ms=async_latency_ms,
            jitter_ms=async_jitter_ms,
            seed=seed,
        )
    if normalized in {"vectorized", "numpy", "v4"}:
        return VectorizedEvaluator()
    if normalized in {"joblib", "v5"}:
        return JoblibEvaluator(workers=workers, batch_size=batch_size, backend=joblib_backend)
    raise KeyError(f"Unknown evaluation strategy '{strategy}'.")


__all__ = [
    "AsyncioEvaluator",
    "EvaluationStats",
    "FitnessEvaluator",
    "JoblibEvaluator",
    "ProcessPoolEvaluator",
    "SequentialEvaluator",
    "ThreadPoolEvaluator",
    "VectorizedEvaluator",
    "build_evaluator",
]
