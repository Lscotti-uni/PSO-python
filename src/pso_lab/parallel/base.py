"""Shared abstractions and timing helpers for fitness evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol

import numpy as np

ObjectiveFn = Callable[[np.ndarray], float | np.ndarray]


@dataclass(slots=True)
class EvaluationStats:
    strategy: str
    wall_time: float = 0.0
    worker_time_total: float = 0.0
    critical_path_time: float = 0.0
    overhead_time: float = 0.0
    tasks_submitted: int = 0
    batch_size: int = 1

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class FitnessEvaluator(Protocol):
    name: str
    last_stats: EvaluationStats

    def evaluate(self, positions: np.ndarray, objective: ObjectiveFn) -> np.ndarray:
        ...


def chunk_ranges(length: int, batch_size: int) -> list[tuple[int, int]]:
    # Batching lets process- and thread-based evaluators amortize scheduling
    # overhead instead of submitting one tiny task per particle.
    return [(start, min(length, start + batch_size)) for start in range(0, length, batch_size)]


def coerce_values(values: float | np.ndarray, expected_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(expected_length, float(array), dtype=float)
    return array.reshape(expected_length)


def estimate_parallel_overhead(wall_time: float, worker_times: list[float]) -> tuple[float, float]:
    # This is a practical proxy: wall time minus the slowest worker captures
    # time spent on scheduling, serialization, synchronization, and startup.
    critical_path = max(worker_times, default=0.0)
    return max(0.0, wall_time - critical_path), critical_path
