"""Vectorized evaluator for NumPy-based swarm fitness computation."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from pso_lab.parallel.base import EvaluationStats, ObjectiveFn, coerce_values


class VectorizedEvaluator:
    name = "vectorized"

    def __init__(self) -> None:
        self.last_stats = EvaluationStats(strategy=self.name, batch_size=0)

    def evaluate(self, positions: np.ndarray, objective: ObjectiveFn) -> np.ndarray:
        wall_start = perf_counter()
        values = coerce_values(objective(positions), len(positions))
        wall_time = perf_counter() - wall_start
        self.last_stats = EvaluationStats(
            strategy=self.name,
            wall_time=wall_time,
            worker_time_total=wall_time,
            critical_path_time=wall_time,
            overhead_time=0.0,
            tasks_submitted=1,
            batch_size=len(positions),
        )
        return values
