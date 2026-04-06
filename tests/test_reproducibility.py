"""Tests that identical seeds reproduce the same PSO trajectory and result."""

import numpy as np

from pso_lab.core import PSOConfig, build_bounds_policy, build_topology
from pso_lab.core.pso import PSO
from pso_lab.objectives import get_objective
from pso_lab.parallel import build_evaluator


def _run_once(seed: int):
    config = PSOConfig(
        objective="sphere",
        dimensions=8,
        swarm_size=30,
        iterations=120,
        seed=seed,
        track_trajectory=False,
        strategy="sequential",
        update_mode="loop",
    )
    evaluator = build_evaluator("sequential")
    pso = PSO(
        config=config,
        objective=get_objective("sphere").fn,
        evaluator=evaluator,
        bounds_policy=build_bounds_policy("clamp"),
        topology=build_topology("global"),
    )
    result = pso.run()
    return result.best_value, result.best_position, result.history


def test_same_seed_produces_same_result() -> None:
    value_a, position_a, history_a = _run_once(123)
    value_b, position_b, history_b = _run_once(123)

    assert value_a == value_b
    assert np.allclose(position_a, position_b)
    deterministic_keys = ["iteration", "best_fitness", "mean_fitness", "min_fitness_current", "gbest_improved", "previous_gbest"]
    reduced_a = [{key: row[key] for key in deterministic_keys} for row in history_a]
    reduced_b = [{key: row[key] for key in deterministic_keys} for row in history_b]
    assert reduced_a == reduced_b
