"""Consistency tests for the V3 (async), V4 (vectorized) and V5 (joblib) evaluators."""

from __future__ import annotations

import numpy as np
import pytest

from pso_lab.objectives import get_objective
from pso_lab.parallel import build_evaluator


@pytest.fixture
def positions() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.5],
            [2.0, 2.0, -2.0],
            [-3.0, 4.0, 1.0],
        ],
        dtype=float,
    )


@pytest.fixture
def objective():
    return get_objective("sphere").fn


def test_asyncio_matches_sequential(positions, objective) -> None:
    sequential = build_evaluator("sequential", batch_size=2)
    async_eval = build_evaluator("async", workers=2, batch_size=2, async_latency_ms=0.0)

    expected = sequential.evaluate(positions, objective)
    actual = async_eval.evaluate(positions, objective)
    assert np.allclose(actual, expected)
    assert async_eval.last_stats.tasks_submitted == 2


def test_vectorized_matches_sequential(positions, objective) -> None:
    sequential = build_evaluator("sequential")
    vectorized = build_evaluator("vectorized")

    expected = sequential.evaluate(positions, objective)
    actual = vectorized.evaluate(positions, objective)
    assert np.allclose(actual, expected)
    assert vectorized.last_stats.tasks_submitted == 1


def test_joblib_matches_sequential(positions, objective) -> None:
    joblib = pytest.importorskip("joblib")  # noqa: F841
    sequential = build_evaluator("sequential")
    jb_eval = build_evaluator("joblib", workers=2, batch_size=2, joblib_backend="threading")

    expected = sequential.evaluate(positions, objective)
    actual = jb_eval.evaluate(positions, objective)
    assert np.allclose(actual, expected)
