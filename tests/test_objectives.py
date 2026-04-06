"""Tests for benchmark objective registration and evaluation behaviour."""

import numpy as np

from pso_lab.objectives import get_objective, list_objectives


def test_registry_exposes_four_core_objectives() -> None:
    assert {"sphere", "rosenbrock", "rastrigin", "ackley"}.issubset(set(list_objectives()))


def test_objectives_reach_known_optimum_value() -> None:
    for name in ["sphere", "rosenbrock", "rastrigin", "ackley"]:
        spec = get_objective(name)
        optimum = spec.known_optimum_position(5)
        value = spec.fn(optimum)
        assert np.isclose(value, spec.known_optimum_value, atol=1e-9)


def test_batch_interface_returns_one_value_per_sample() -> None:
    spec = get_objective("sphere")
    samples = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 2.0]])
    values = spec.fn(samples)
    assert isinstance(values, np.ndarray)
    assert values.shape == (3,)
