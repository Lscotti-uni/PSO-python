"""Tests for the optional V5 joblib evaluator backend."""

import numpy as np

from pso_lab.objectives import get_objective
from pso_lab.parallel import build_evaluator


def test_joblib_matches_sequential_on_sphere() -> None:
    objective = get_objective("sphere").fn
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [2.0, 2.0],
            [-3.0, 4.0],
        ],
        dtype=float,
    )

    sequential = build_evaluator("sequential", batch_size=2)
    joblib = build_evaluator("joblib", workers=2, batch_size=2)

    try:
        expected = sequential.evaluate(positions, objective)
        actual = joblib.evaluate(positions, objective)
    finally:
        close_method = getattr(joblib, "close", None)
        if callable(close_method):
            close_method()

    assert np.allclose(actual, expected)
    assert joblib.last_stats.tasks_submitted == 2
