"""Tests for explicit boundary-handling policies."""

import numpy as np

from pso_lab.core.bounds import ClampBoundsPolicy, ReflectBoundsPolicy


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
