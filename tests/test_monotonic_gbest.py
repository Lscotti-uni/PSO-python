"""Tests that the historical global best never degrades across iterations."""

from pso_lab.core import PSOConfig, build_bounds_policy, build_topology
from pso_lab.core.pso import PSO
from pso_lab.objectives import get_objective
from pso_lab.parallel import build_evaluator


def test_global_best_history_is_monotonic_non_increasing() -> None:
    config = PSOConfig(
        objective="sphere",
        dimensions=6,
        swarm_size=25,
        iterations=80,
        seed=321,
        strategy="sequential",
    )
    pso = PSO(
        config=config,
        objective=get_objective("sphere").fn,
        evaluator=build_evaluator("sequential"),
        bounds_policy=build_bounds_policy("clamp"),
        topology=build_topology("global"),
    )
    result = pso.run()
    best_values = [row["best_fitness"] for row in result.history]
    assert all(left >= right for left, right in zip(best_values, best_values[1:]))
