"""Benchmark objective exports and registry accessors."""

from pso_lab.objectives.ackley import ackley
from pso_lab.objectives.rastrigin import rastrigin
from pso_lab.objectives.registry import ObjectiveSpec, get_objective, list_objectives
from pso_lab.objectives.rosenbrock import rosenbrock
from pso_lab.objectives.sphere import sphere

__all__ = [
    "ObjectiveSpec",
    "ackley",
    "get_objective",
    "list_objectives",
    "rastrigin",
    "rosenbrock",
    "sphere",
]
