"""Joblib-based evaluator used as the optional V5 parallel backend."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from joblib import Parallel, delayed

from pso_lab.parallel.base import (
    EvaluationStats,
    ObjectiveFn,
    chunk_ranges,
    coerce_values,
    estimate_parallel_overhead,
)


def _evaluate_chunk(chunk: np.ndarray, objective: ObjectiveFn) -> tuple[np.ndarray, float]:
    start = perf_counter()
    values = coerce_values(objective(chunk if len(chunk) > 1 else chunk[0]), len(chunk))
    return values, perf_counter() - start


class JoblibEvaluator:
    name = "joblib"

    def __init__(self, workers: int | None = None, batch_size: int = 8, backend: str = "loky") -> None:
        self.workers = workers
        self.batch_size = max(1, batch_size)
        self.backend = backend
        self.last_stats = EvaluationStats(strategy=self.name, batch_size=self.batch_size)

    def evaluate(self, positions: np.ndarray, objective: ObjectiveFn) -> np.ndarray:
        chunks = chunk_ranges(len(positions), self.batch_size)
        wall_start = perf_counter()
        raw_results = Parallel(
            n_jobs=self.workers if self.workers is not None else -1,
            # The backend is configurable so experiments can compare process-
            # like and thread-like behaviour under the same public API.
            backend=self.backend,
        )(
            delayed(_evaluate_chunk)(positions[start:stop], objective)
            for start, stop in chunks
        )

        values = [chunk_values for chunk_values, _ in raw_results]
        worker_times = [worker_time for _, worker_time in raw_results]
        wall_time = perf_counter() - wall_start
        overhead_time, critical_path = estimate_parallel_overhead(wall_time, worker_times)
        self.last_stats = EvaluationStats(
            strategy=self.name,
            wall_time=wall_time,
            worker_time_total=sum(worker_times),
            critical_path_time=critical_path,
            overhead_time=overhead_time,
            tasks_submitted=len(chunks),
            batch_size=self.batch_size,
        )
        return np.concatenate(values) if values else np.empty(0, dtype=float)

    def close(self) -> None:
        return None
