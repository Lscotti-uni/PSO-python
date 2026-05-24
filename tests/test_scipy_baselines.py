"""Smoke tests for the SciPy baselines wrappers."""

from __future__ import annotations

import pytest

from pso_lab.objectives import get_objective


def test_lbfgsb_finds_the_sphere_minimum() -> None:
    pytest.importorskip("scipy")
    from pso_lab.baselines import run_lbfgsb

    spec = get_objective("sphere")
    result = run_lbfgsb(spec.fn, spec.default_bounds, dimensions=5, seed=7, n_starts=3)
    assert result.method == "lbfgsb"
    assert result.best_value < 1e-6
    assert result.nfev > 0


def test_differential_evolution_runs_and_returns_position() -> None:
    pytest.importorskip("scipy")
    from pso_lab.baselines import run_differential_evolution

    spec = get_objective("sphere")
    result = run_differential_evolution(
        spec.fn, spec.default_bounds, dimensions=3, seed=11, maxiter=20, popsize=5
    )
    assert result.method == "differential_evolution"
    assert result.best_position.shape == (3,)
    assert result.best_value < 1.0
