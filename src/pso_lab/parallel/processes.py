"""Process-pool evaluator for CPU-bound fitness computation."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np

from pso_lab.parallel.base import (
    EvaluationStats,
    ObjectiveFn,
    chunk_ranges,
    coerce_values,
    estimate_parallel_overhead,
)


def _process_worker(chunk: np.ndarray, objective: ObjectiveFn) -> tuple[np.ndarray, float]:
    start = perf_counter()
    values = coerce_values(objective(chunk if len(chunk) > 1 else chunk[0]), len(chunk))
    return values, perf_counter() - start


class ProcessPoolEvaluator:
    name = "process"

    def __init__(self, workers: int | None = None, batch_size: int = 8) -> None:
        self.workers = workers
        self.batch_size = max(1, batch_size)
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            # "spawn" is the most portable choice on Windows and avoids inheriting
            # unexpected process state from the parent interpreter.
            mp_context=mp.get_context("spawn"),
        )
        self.last_stats = EvaluationStats(strategy=self.name, batch_size=self.batch_size)

    def evaluate(self, positions: np.ndarray, objective: ObjectiveFn) -> np.ndarray:
        chunks = chunk_ranges(len(positions), self.batch_size)
        wall_start = perf_counter()
        futures = []
        for chunk_id, (start, stop) in enumerate(chunks):
            futures.append((chunk_id, self._executor.submit(_process_worker, positions[start:stop], objective)))

        results: list[np.ndarray | None] = [None] * len(chunks)
        worker_times: list[float] = []
        for chunk_id, future in futures:
            # Chunk ids preserve deterministic output ordering even though
            # subprocesses may finish in a different order.
            values, worker_time = future.result()
            results[chunk_id] = values
            worker_times.append(worker_time)

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
        return np.concatenate([item for item in results if item is not None]) if results else np.empty(0, dtype=float)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
