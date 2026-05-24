"""Asyncio-based evaluator that demonstrates cooperative concurrency on I/O-bound workloads.

The evaluator dispatches one coroutine per particle (or per batch) and runs them
through ``asyncio.gather``. The optional simulated latency models the typical
use case for async evaluation: objectives that wait on network or disk I/O.
"""

from __future__ import annotations

import asyncio
import random
from time import perf_counter

import numpy as np

from pso_lab.parallel.base import (
    EvaluationStats,
    ObjectiveFn,
    chunk_ranges,
    coerce_values,
    estimate_parallel_overhead,
)


class AsyncioEvaluator:
    name = "async"

    def __init__(
        self,
        workers: int | None = None,
        batch_size: int = 1,
        *,
        latency_ms: float = 0.0,
        jitter_ms: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.workers = max(1, workers) if workers else None
        self.batch_size = max(1, batch_size)
        self.latency_ms = max(0.0, latency_ms)
        self.jitter_ms = max(0.0, jitter_ms)
        self._rng = random.Random(seed)
        self._semaphore = asyncio.Semaphore(self.workers) if self.workers else None
        self.last_stats = EvaluationStats(strategy=self.name, batch_size=self.batch_size)

    async def _evaluate_chunk(
        self,
        chunk: np.ndarray,
        objective: ObjectiveFn,
    ) -> tuple[np.ndarray, float]:
        async def _runner() -> tuple[np.ndarray, float]:
            start = perf_counter()
            if self.latency_ms > 0 or self.jitter_ms > 0:
                # Simulate a slow async call to make concurrent gains observable.
                wait_ms = self.latency_ms + self._rng.uniform(0.0, self.jitter_ms)
                await asyncio.sleep(wait_ms / 1000.0)
            loop = asyncio.get_running_loop()
            # Offload the actual compute to a worker thread to keep the event
            # loop responsive when the objective itself is CPU-bound.
            values = await loop.run_in_executor(
                None,
                lambda: coerce_values(
                    objective(chunk if len(chunk) > 1 else chunk[0]),
                    len(chunk),
                ),
            )
            return values, perf_counter() - start

        if self._semaphore is None:
            return await _runner()
        async with self._semaphore:
            return await _runner()

    async def _evaluate_async(
        self,
        positions: np.ndarray,
        objective: ObjectiveFn,
    ) -> tuple[list[np.ndarray], list[float]]:
        chunks = chunk_ranges(len(positions), self.batch_size)
        tasks = [self._evaluate_chunk(positions[start:stop], objective) for start, stop in chunks]
        results = await asyncio.gather(*tasks)
        values = [chunk_values for chunk_values, _ in results]
        worker_times = [worker_time for _, worker_time in results]
        return values, worker_times

    def evaluate(self, positions: np.ndarray, objective: ObjectiveFn) -> np.ndarray:
        wall_start = perf_counter()
        values, worker_times = asyncio.run(self._evaluate_async(positions, objective))
        wall_time = perf_counter() - wall_start
        overhead_time, critical_path = estimate_parallel_overhead(wall_time, worker_times)
        self.last_stats = EvaluationStats(
            strategy=self.name,
            wall_time=wall_time,
            worker_time_total=sum(worker_times),
            critical_path_time=critical_path,
            overhead_time=overhead_time,
            tasks_submitted=len(worker_times),
            batch_size=self.batch_size,
        )
        return np.concatenate(values) if values else np.empty(0, dtype=float)
