"""
registry.py

This module acts as a central registry (catalog) of all available objective
functions in the project.

Why do we need this?

Instead of scattering conditionals like:

    if name == "ackley":
        ...
    elif name == "rastrigin":
        ...

across multiple files (runners, experiments, grid search, etc.), we define
a single source of truth that maps:

    "objective_name" (string from CLI/YAML)
            ↓
    ObjectiveSpec (function + metadata)

This keeps the architecture modular, maintainable and scalable.
Adding a new objective only requires:
    1. Creating the objective file.
    2. Registering it here.
"""

# objectives/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union
import numpy as np

from . import sphere as sphere_mod
from . import rosenbrock as rosenbrock_mod
from . import rastrigin as rastrigin_mod
from . import ackley as ackley_mod

Array = np.ndarray
Bounds = Tuple[float, float]
ObjectiveFn = Callable[[Union[Array, list]], Union[float, Array]]


@dataclass(frozen=True)
class ObjectiveSpec:
    key: str                 # machine-friendly key, e.g. "ackley"
    name: str                # human-friendly name, e.g. "Ackley"
    fn: ObjectiveFn
    default_bounds: Bounds
    known_optimum_value: float


# Central registry (single source of truth for "what objectives exist")
OBJECTIVES: Dict[str, ObjectiveSpec] = {
    "sphere": ObjectiveSpec(
        key="sphere",
        name=sphere_mod.NAME,
        fn=sphere_mod.sphere,
        default_bounds=sphere_mod.DEFAULT_BOUNDS,
        known_optimum_value=sphere_mod.KNOWN_OPTIMUM["value"],
    ),
    "rosenbrock": ObjectiveSpec(
        key="rosenbrock",
        name=rosenbrock_mod.NAME,
        fn=rosenbrock_mod.rosenbrock,
        default_bounds=rosenbrock_mod.DEFAULT_BOUNDS,
        known_optimum_value=rosenbrock_mod.KNOWN_OPTIMUM["value"],
    ),
    "rastrigin": ObjectiveSpec(
        key="rastrigin",
        name=rastrigin_mod.NAME,
        fn=rastrigin_mod.rastrigin,
        default_bounds=rastrigin_mod.DEFAULT_BOUNDS,
        known_optimum_value=rastrigin_mod.KNOWN_OPTIMUM["value"],
    ),
    "ackley": ObjectiveSpec(
        key="ackley",
        name=ackley_mod.NAME,
        fn=ackley_mod.ackley,
        default_bounds=ackley_mod.DEFAULT_BOUNDS,
        known_optimum_value=ackley_mod.KNOWN_OPTIMUM["value"],
    ),
}


def list_objectives() -> list[str]:
    return sorted(OBJECTIVES.keys())


def get_objective(name: str) -> ObjectiveSpec:
    key = name.strip().lower()
    if key not in OBJECTIVES:
        available = ", ".join(list_objectives())
        raise KeyError(f"Unknown objective '{name}'. Available: {available}")
    return OBJECTIVES[key]