"""Thin wrappers around three scipy.optimize baselines.

These give a recognised yardstick to compare against PSO on every objective in
the registry. scipy is treated as an optional dependency and imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np


@dataclass(slots=True)
class BaselineResult:
    method: str
    best_value: float
    best_position: np.ndarray
    iterations: int
    nfev: int
    wall_time: float
    success: bool
    message: str


def expand_bounds_list(
    bounds, dimensions: int
) -> list[tuple[float, float]]:
    array = np.asarray(bounds, dtype=float)
    if array.shape == (2,):
        return [(float(array[0]), float(array[1]))] * dimensions
    if array.shape == (dimensions, 2):
        return [(float(low), float(high)) for low, high in array]
    raise ValueError("bounds must be shaped as (2,) or (dimensions, 2)")


def _wrap_single(objective: Callable[[np.ndarray], float | np.ndarray]) -> Callable[[np.ndarray], float]:
    def _wrapped(x: np.ndarray) -> float:
        value = objective(np.asarray(x, dtype=float))
        if isinstance(value, np.ndarray):
            return float(value.ravel()[0])
        return float(value)

    return _wrapped


def run_differential_evolution(
    objective,
    bounds,
    dimensions: int,
    *,
    seed: int | None = None,
    maxiter: int = 200,
    popsize: int = 15,
    tol: float = 1e-8,
) -> BaselineResult:
    from scipy.optimize import differential_evolution

    bounds_list = expand_bounds_list(bounds, dimensions)
    wrapped = _wrap_single(objective)
    start = perf_counter()
    result = differential_evolution(
        wrapped,
        bounds=bounds_list,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=True,
        updating="deferred",
    )
    wall = perf_counter() - start
    return BaselineResult(
        method="differential_evolution",
        best_value=float(result.fun),
        best_position=np.asarray(result.x, dtype=float),
        iterations=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        wall_time=wall,
        success=bool(result.success),
        message=str(result.message),
    )


def run_dual_annealing(
    objective,
    bounds,
    dimensions: int,
    *,
    seed: int | None = None,
    maxiter: int = 1000,
) -> BaselineResult:
    from scipy.optimize import dual_annealing

    bounds_list = expand_bounds_list(bounds, dimensions)
    wrapped = _wrap_single(objective)
    start = perf_counter()
    result = dual_annealing(wrapped, bounds=bounds_list, seed=seed, maxiter=maxiter)
    wall = perf_counter() - start
    return BaselineResult(
        method="dual_annealing",
        best_value=float(result.fun),
        best_position=np.asarray(result.x, dtype=float),
        iterations=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        wall_time=wall,
        success=bool(result.success),
        message=str(result.message),
    )


def run_lbfgsb(
    objective,
    bounds,
    dimensions: int,
    *,
    seed: int | None = None,
    n_starts: int = 5,
    maxiter: int = 200,
) -> BaselineResult:
    from scipy.optimize import minimize

    bounds_list = expand_bounds_list(bounds, dimensions)
    wrapped = _wrap_single(objective)
    rng = np.random.default_rng(seed)
    lows = np.array([low for low, _ in bounds_list], dtype=float)
    highs = np.array([high for _, high in bounds_list], dtype=float)

    best: BaselineResult | None = None
    total_nfev = 0
    total_iter = 0
    start = perf_counter()
    for _ in range(max(1, n_starts)):
        x0 = rng.uniform(lows, highs)
        result = minimize(
            wrapped,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": maxiter},
        )
        total_nfev += int(getattr(result, "nfev", 0))
        total_iter += int(getattr(result, "nit", 0))
        if best is None or result.fun < best.best_value:
            best = BaselineResult(
                method="lbfgsb",
                best_value=float(result.fun),
                best_position=np.asarray(result.x, dtype=float),
                iterations=int(getattr(result, "nit", 0)),
                nfev=int(getattr(result, "nfev", 0)),
                wall_time=0.0,  # filled after the loop
                success=bool(result.success),
                message=str(result.message),
            )

    wall = perf_counter() - start
    assert best is not None
    best.wall_time = wall
    best.iterations = total_iter
    best.nfev = total_nfev
    return best
