"""Sequential baseline evaluator used by the V0 reference implementation."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from pso_lab.parallel.base import EvaluationStats, ObjectiveFn, chunk_ranges, coerce_values


class SequentialEvaluator:
    name = "sequential"

    def __init__(self, batch_size: int = 1) -> None:
        self.batch_size = max(1, batch_size)
        self.last_stats = EvaluationStats(strategy=self.name, batch_size=self.batch_size)

    def evaluate(self, positions: np.ndarray, objective: ObjectiveFn) -> np.ndarray:
        outputs: list[np.ndarray] = []
        worker_times: list[float] = []
        wall_start = perf_counter()

        for start, stop in chunk_ranges(len(positions), self.batch_size):
            chunk = positions[start:stop]
            compute_start = perf_counter()
            chunk_values = objective(chunk if len(chunk) > 1 else chunk[0])
            worker_times.append(perf_counter() - compute_start)
            outputs.append(coerce_values(chunk_values, len(chunk)))

        wall_time = perf_counter() - wall_start
        worker_total = sum(worker_times)
        self.last_stats = EvaluationStats(
            strategy=self.name,
            wall_time=wall_time,
            worker_time_total=worker_total,
            critical_path_time=worker_total,
            overhead_time=max(0.0, wall_time - worker_total),
            tasks_submitted=len(worker_times),
            batch_size=self.batch_size,
        )
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=float)
