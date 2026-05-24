"""Sanity tests for the inverted-pendulum PID use case."""

from __future__ import annotations

import numpy as np

from pso_lab.objectives import get_objective
from pso_lab.use_cases.inverted_pendulum import (
    DEFAULT_BOUNDS_PER_DIM,
    PENDULUM_PID_DIMENSIONS,
    PENDULUM_PID_REFERENCE_GAINS,
    pendulum_pid_cost,
    simulate_pendulum,
)


def test_registry_exposes_pendulum_pid_with_fixed_dimensions() -> None:
    spec = get_objective("pendulum_pid")
    assert spec.fixed_dimensions == PENDULUM_PID_DIMENSIONS
    assert spec.default_bounds == DEFAULT_BOUNDS_PER_DIM


def test_baseline_gains_keep_pole_upright() -> None:
    record = simulate_pendulum(PENDULUM_PID_REFERENCE_GAINS)
    assert not record["failed"]
    # The conservative baseline must keep the pole well under the failure angle.
    assert np.max(np.abs(record["states"][:, 2])) < 0.6


def test_cost_is_lower_for_baseline_than_for_zero_gains() -> None:
    baseline = pendulum_pid_cost(PENDULUM_PID_REFERENCE_GAINS)
    bad = pendulum_pid_cost(np.zeros(PENDULUM_PID_DIMENSIONS))
    assert baseline < bad


def test_batched_cost_matches_per_particle_cost() -> None:
    gains = np.stack([PENDULUM_PID_REFERENCE_GAINS, np.zeros(PENDULUM_PID_DIMENSIONS)])
    batched = pendulum_pid_cost(gains)
    assert batched.shape == (2,)
    assert batched[0] == pendulum_pid_cost(PENDULUM_PID_REFERENCE_GAINS)
    assert batched[1] == pendulum_pid_cost(np.zeros(PENDULUM_PID_DIMENSIONS))
