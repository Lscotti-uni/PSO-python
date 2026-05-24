"""Central registry for benchmark objectives and their metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
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
BoundsSpec = tuple[float, float] | list[tuple[float, float]] | list[list[float]]


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    key: str
    name: str
    fn: ObjectiveFn
    default_bounds: BoundsSpec
    known_optimum_value: float
    known_optimum_position: Callable[[int], np.ndarray]
    fixed_dimensions: int | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def _make_spec(
    *,
    key: str,
    name: str,
    fn: ObjectiveFn,
    bounds: BoundsSpec,
    optimum_value: float,
    optimum_position: Callable[[int], np.ndarray],
    fixed_dimensions: int | None = None,
    tags: tuple[str, ...] = (),
) -> ObjectiveSpec:
    return ObjectiveSpec(
        key=key,
        name=name,
        fn=fn,
        default_bounds=bounds,
        known_optimum_value=optimum_value,
        known_optimum_position=optimum_position,
        fixed_dimensions=fixed_dimensions,
        tags=tags,
    )


OBJECTIVES: dict[str, ObjectiveSpec] = {
    "sphere": _make_spec(
        key="sphere",
        name=SPHERE_NAME,
        fn=sphere,
        bounds=SPHERE_BOUNDS,
        optimum_value=SPHERE_OPTIMUM["value"],
        optimum_position=SPHERE_OPTIMUM["position"],
        tags=("unimodal", "separable"),
    ),
    "rosenbrock": _make_spec(
        key="rosenbrock",
        name=ROSENBROCK_NAME,
        fn=rosenbrock,
        bounds=ROSENBROCK_BOUNDS,
        optimum_value=ROSENBROCK_OPTIMUM["value"],
        optimum_position=ROSENBROCK_OPTIMUM["position"],
        tags=("unimodal", "non-separable", "valley"),
    ),
    "rastrigin": _make_spec(
        key="rastrigin",
        name=RASTRIGIN_NAME,
        fn=rastrigin,
        bounds=RASTRIGIN_BOUNDS,
        optimum_value=RASTRIGIN_OPTIMUM["value"],
        optimum_position=RASTRIGIN_OPTIMUM["position"],
        tags=("multimodal", "separable"),
    ),
    "ackley": _make_spec(
        key="ackley",
        name=ACKLEY_NAME,
        fn=ackley,
        bounds=ACKLEY_BOUNDS,
        optimum_value=ACKLEY_OPTIMUM["value"],
        optimum_position=ACKLEY_OPTIMUM["position"],
        tags=("multimodal", "non-separable"),
    ),
}


def register_objective(spec: ObjectiveSpec) -> None:
    OBJECTIVES[spec.key] = spec


def list_objectives() -> list[str]:
    return sorted(OBJECTIVES)


def get_objective(name: str) -> ObjectiveSpec:
    key = name.strip().lower()
    if key not in OBJECTIVES:
        available = ", ".join(list_objectives())
        raise KeyError(f"Unknown objective '{name}'. Available: {available}")
    return OBJECTIVES[key]


# Side-effect registration: importing the registry should make every objective
# (including use cases) discoverable, without requiring callers to import them
# individually.
from . import _use_case_registration  # noqa: E402, F401  (registration only)
