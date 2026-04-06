"""Central registry for benchmark objectives and their metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ackley import DEFAULT_BOUNDS as ACKLEY_BOUNDS
from .ackley import KNOWN_OPTIMUM as ACKLEY_OPTIMUM
from .ackley import NAME as ACKLEY_NAME
from .ackley import ackley
from .rastrigin import DEFAULT_BOUNDS as RASTRIGIN_BOUNDS
from .rastrigin import KNOWN_OPTIMUM as RASTRIGIN_OPTIMUM
from .rastrigin import NAME as RASTRIGIN_NAME
from .rastrigin import rastrigin
from .rosenbrock import DEFAULT_BOUNDS as ROSENBROCK_BOUNDS
from .rosenbrock import KNOWN_OPTIMUM as ROSENBROCK_OPTIMUM
from .rosenbrock import NAME as ROSENBROCK_NAME
from .rosenbrock import rosenbrock
from .sphere import DEFAULT_BOUNDS as SPHERE_BOUNDS
from .sphere import KNOWN_OPTIMUM as SPHERE_OPTIMUM
from .sphere import NAME as SPHERE_NAME
from .sphere import sphere

Array = np.ndarray
ObjectiveFn = Callable[[Array | list[float]], float | Array]


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    key: str
    name: str
    fn: ObjectiveFn
    default_bounds: tuple[float, float]
    known_optimum_value: float
    known_optimum_position: Callable[[int], np.ndarray]


OBJECTIVES: dict[str, ObjectiveSpec] = {
    "sphere": ObjectiveSpec(
        key="sphere",
        name=SPHERE_NAME,
        fn=sphere,
        default_bounds=SPHERE_BOUNDS,
        known_optimum_value=SPHERE_OPTIMUM["value"],
        known_optimum_position=SPHERE_OPTIMUM["position"],
    ),
    "rosenbrock": ObjectiveSpec(
        key="rosenbrock",
        name=ROSENBROCK_NAME,
        fn=rosenbrock,
        default_bounds=ROSENBROCK_BOUNDS,
        known_optimum_value=ROSENBROCK_OPTIMUM["value"],
        known_optimum_position=ROSENBROCK_OPTIMUM["position"],
    ),
    "rastrigin": ObjectiveSpec(
        key="rastrigin",
        name=RASTRIGIN_NAME,
        fn=rastrigin,
        default_bounds=RASTRIGIN_BOUNDS,
        known_optimum_value=RASTRIGIN_OPTIMUM["value"],
        known_optimum_position=RASTRIGIN_OPTIMUM["position"],
    ),
    "ackley": ObjectiveSpec(
        key="ackley",
        name=ACKLEY_NAME,
        fn=ackley,
        default_bounds=ACKLEY_BOUNDS,
        known_optimum_value=ACKLEY_OPTIMUM["value"],
        known_optimum_position=ACKLEY_OPTIMUM["position"],
    ),
}


def list_objectives() -> list[str]:
    return sorted(OBJECTIVES)


def get_objective(name: str) -> ObjectiveSpec:
    key = name.strip().lower()
    if key not in OBJECTIVES:
        available = ", ".join(list_objectives())
        raise KeyError(f"Unknown objective '{name}'. Available: {available}")
    return OBJECTIVES[key]
