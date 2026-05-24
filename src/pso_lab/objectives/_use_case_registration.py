"""Register use-case objectives into the central registry on import."""

from __future__ import annotations

from pso_lab.objectives.registry import ObjectiveSpec, register_objective
from pso_lab.use_cases.inverted_pendulum import (
    DEFAULT_BOUNDS_PER_DIM,
    PENDULUM_PID_DIMENSIONS,
    PENDULUM_PID_NAME,
    PENDULUM_PID_REFERENCE_GAINS,
    pendulum_pid_cost,
)


def _pendulum_optimum_position(_dimensions: int):
    # The known LQR-style baseline gains are returned for reference; PSO is
    # expected to land near (but not exactly at) this configuration.
    return PENDULUM_PID_REFERENCE_GAINS


register_objective(
    ObjectiveSpec(
        key="pendulum_pid",
        name=PENDULUM_PID_NAME,
        fn=pendulum_pid_cost,
        default_bounds=DEFAULT_BOUNDS_PER_DIM,
        known_optimum_value=0.0,
        known_optimum_position=_pendulum_optimum_position,
        fixed_dimensions=PENDULUM_PID_DIMENSIONS,
        tags=("use_case", "multimodal", "non-separable", "noisy"),
    )
)
