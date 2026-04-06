"""Tests that Sphere converges close to its known optimum with reasonable settings."""

from pso_lab.core import PSOConfig, build_bounds_policy, build_topology
from pso_lab.core.pso import PSO
from pso_lab.objectives import get_objective
from pso_lab.parallel import build_evaluator


def test_sphere_converges_with_reasonable_parameters() -> None:
    config = PSOConfig(
        objective="sphere",
        dimensions=10,
        swarm_size=40,
        iterations=200,
        seed=123,
        tolerance=1e-8,
        strategy="vectorized",
        update_mode="vectorized",
        velocity_clamp=0.2,
    )
    pso = PSO(
        config=config,
        objective=get_objective("sphere").fn,
        evaluator=build_evaluator("vectorized"),
        bounds_policy=build_bounds_policy("clamp"),
        topology=build_topology("global"),
    )
    result = pso.run()

    assert result.best_value < 1e-6
    assert result.stop_reason in {"tolerance", "max_iterations"}
