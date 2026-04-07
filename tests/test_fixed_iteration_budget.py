"""Tests that fixed-budget runs ignore early-stopping criteria."""

from pso_lab.core import PSOConfig, build_bounds_policy, build_topology
from pso_lab.core.pso import PSO
from pso_lab.objectives import get_objective
from pso_lab.parallel import build_evaluator


def test_fixed_iteration_budget_ignores_tolerance() -> None:
    config = PSOConfig(
        objective="sphere",
        dimensions=2,
        swarm_size=20,
        iterations=15,
        seed=123,
        tolerance=1e9,
        enable_early_stopping=False,
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

    assert result.state.iteration == 15
    assert result.stop_reason == "max_iterations"
