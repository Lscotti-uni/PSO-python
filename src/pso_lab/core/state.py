"""State containers used by the PSO core and result serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class SwarmState:
    positions: np.ndarray
    velocities: np.ndarray
    personal_best_positions: np.ndarray
    personal_best_values: np.ndarray
    global_best_position: np.ndarray
    global_best_value: float
    iteration: int = 0

    def copy(self) -> "SwarmState":
        return SwarmState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            personal_best_positions=self.personal_best_positions.copy(),
            personal_best_values=self.personal_best_values.copy(),
            global_best_position=self.global_best_position.copy(),
            global_best_value=float(self.global_best_value),
            iteration=self.iteration,
        )


@dataclass(slots=True)
class IterationMetrics:
    iteration: int
    best_fitness: float
    mean_fitness: float
    min_fitness_current: float
    eval_time: float
    update_time: float
    overhead_time: float
    worker_time_total: float
    critical_path_time: float
    tasks_submitted: int
    total_iter_time: float
    gbest_improved: bool
    previous_gbest: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
