"""Async evaluator used to model latency-bound objective calls."""

from __future__ import annotations

import asyncio
from time import perf_counter

import numpy as np

from pso_lab.parallel.base import EvaluationStats, ObjectiveFn, coerce_values, estimate_parallel_overhead


class AsyncioEvaluator:
    name = "asyncio"

    def __init__(self, latency_ms: float = 3.0, jitter_ms: float = 0.0, seed: int | None = None) -> None:
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.rng = np.random.default_rng(seed)
        self.last_stats = EvaluationStats(strategy=self.name, batch_size=1)

    async def _evaluate_one(
        self,
        particle: np.ndarray,
        objective: ObjectiveFn,
        delay_ms: float,
    ) -> tuple[float, float]:
        # This variant models latency-bound evaluation. It is intentionally not
        # meant to outperform process-based CPU-bound execution.
        delay_seconds = max(0.0, delay_ms / 1000.0)
        start = perf_counter()
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
        value = float(np.asarray(objective(particle), dtype=float))
        elapsed = perf_counter() - start
        return value, elapsed

    async def _evaluate_async(self, positions: np.ndarray, objective: ObjectiveFn) -> tuple[np.ndarray, list[float]]:
        delays = np.full(len(positions), self.latency_ms, dtype=float)
        if self.jitter_ms > 0:
            delays += self.rng.uniform(0.0, self.jitter_ms, size=len(positions))
        # One task per particle exposes waiting overlap, which is exactly what
        # cooperative scheduling is supposed to exploit.
        tasks = [
            self._evaluate_one(particle, objective, float(delay_ms))
            for particle, delay_ms in zip(positions, delays)
        ]
        raw = await asyncio.gather(*tasks)
        values = np.asarray([item[0] for item in raw], dtype=float)
        worker_times = [item[1] for item in raw]
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
            tasks_submitted=len(positions),
            batch_size=1,
        )
        return coerce_values(values, len(positions))
