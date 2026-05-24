"""Joblib-backed evaluator (V5) that exposes loky/threading/multiprocessing backends.

joblib is treated as an optional dependency. The constructor raises a clear
ImportError if it is not installed so the rest of the project keeps working in
environments without joblib.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from pso_lab.parallel.base import (
    EvaluationStats,
    ObjectiveFn,
    chunk_ranges,
    coerce_values,
    estimate_parallel_overhead,
)


def _joblib_worker(chunk: np.ndarray, objective: ObjectiveFn) -> tuple[np.ndarray, float]:
    start = perf_counter()
    values = coerce_values(objective(chunk if len(chunk) > 1 else chunk[0]), len(chunk))
    return values, perf_counter() - start


class JoblibEvaluator:
    name = "joblib"

    def __init__(
        self,
        workers: int | None = None,
        batch_size: int = 1,
        *,
        backend: str = "loky",
    ) -> None:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise ImportError(
                "joblib is required for the V5 evaluator. Install it with `pip install joblib`."
            ) from exc

        self.workers = workers if workers else -1
        self.batch_size = max(1, batch_size)
        self.backend = backend
        self._Parallel = Parallel
        self._delayed = delayed
        self.last_stats = EvaluationStats(strategy=self.name, batch_size=self.batch_size)

    def evaluate(self, positions: np.ndarray, objective: ObjectiveFn) -> np.ndarray:
        chunks = chunk_ranges(len(positions), self.batch_size)
        wall_start = perf_counter()
        results = self._Parallel(n_jobs=self.workers, backend=self.backend, prefer=None)(
            self._delayed(_joblib_worker)(positions[start:stop], objective)
            for start, stop in chunks
        )
        wall_time = perf_counter() - wall_start
        values = [chunk_values for chunk_values, _ in results]
        worker_times = [worker_time for _, worker_time in results]
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
