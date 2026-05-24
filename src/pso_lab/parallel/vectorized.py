"""NumPy-vectorized evaluator that scores the whole swarm with one call."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from pso_lab.parallel.base import EvaluationStats, ObjectiveFn, coerce_values


class VectorizedEvaluator:
    name = "vectorized"

    def __init__(self, batch_size: int = 1) -> None:
        # batch_size is accepted for API symmetry but ignored: vectorized eval
        # always processes the full swarm as a single batched NumPy operation.
        self.batch_size = max(1, batch_size)
        self.last_stats = EvaluationStats(strategy=self.name, batch_size=self.batch_size)

    def evaluate(self, positions: np.ndarray, objective: ObjectiveFn) -> np.ndarray:
        wall_start = perf_counter()
        compute_start = perf_counter()
        values = coerce_values(objective(positions), len(positions))
        worker_time = perf_counter() - compute_start
        wall_time = perf_counter() - wall_start
        self.last_stats = EvaluationStats(
            strategy=self.name,
            wall_time=wall_time,
            worker_time_total=worker_time,
            critical_path_time=worker_time,
            overhead_time=max(0.0, wall_time - worker_time),
            tasks_submitted=1,
            batch_size=len(positions),
        )
        return values
