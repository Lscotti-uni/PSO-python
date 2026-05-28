"""Tests for explicit boundary-handling policies and velocity clamping."""

import numpy as np

from pso_lab.core.bounds import ClampBoundsPolicy, ReflectBoundsPolicy
from pso_lab.core.config import PSOConfig
from pso_lab.experiments.runner import run_single_experiment


def test_clamp_bounds_policy_keeps_positions_inside_box() -> None:
    policy = ClampBoundsPolicy()
    positions = np.array([[-2.0, 0.2], [1.5, 3.0]])
    velocities = np.array([[0.1, 0.2], [0.3, 0.4]])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])

    new_positions, new_velocities = policy.apply(positions, velocities, lower, upper)

    assert np.all(new_positions >= lower)
    assert np.all(new_positions <= upper)
    assert np.all(new_velocities[new_positions != positions] == 0.0)


def test_reflect_bounds_policy_reflects_and_flips_velocity() -> None:
    policy = ReflectBoundsPolicy()
    positions = np.array([[1.2, -1.4]])
    velocities = np.array([[0.5, -0.7]])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])

    new_positions, new_velocities = policy.apply(positions, velocities, lower, upper)

    assert np.all(new_positions >= lower)
    assert np.all(new_positions <= upper)
    assert new_velocities[0, 0] < 0
    assert new_velocities[0, 1] > 0


def test_velocity_clamp_bounds_displacement_per_iteration() -> None:
    # velocity_clamp limits each velocity component to a fraction of the box
    # span; consecutive position deltas must stay within that envelope.
    clamp_fraction = 0.05
    bounds = [-5.0, 5.0]
    config = PSOConfig.from_mapping({
        "objective": "sphere",
        "dimensions": 4,
        "swarm_size": 12,
        "iterations": 6,
        "inertia": 1.5,           # over-inflate to provoke large unclamped velocities
        "cognitive": 2.0,
        "social": 2.0,
        "bounds": bounds,
        "boundary_strategy": "clamp",
        "velocity_clamp": clamp_fraction,
        "enable_early_stopping": False,
        "track_trajectory": True,
        "seed": 1,
    })
    info = run_single_experiment(config, save_results=False)
    trajectory = info["result"].trajectory
    span = bounds[1] - bounds[0]
    max_step = clamp_fraction * span + 1e-9
    for prev, curr in zip(trajectory[:-1], trajectory[1:]):
        # After bounds clipping, the per-iteration displacement of every
        # particle must never exceed the velocity clamp envelope.
        assert np.max(np.abs(curr - prev)) <= max_step
