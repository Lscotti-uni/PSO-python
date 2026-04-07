"""Evaluator consistency tests for V0, V1 and V2."""

import numpy as np

from pso_lab.objectives import get_objective
from pso_lab.parallel import build_evaluator


def test_thread_and_process_match_sequential_on_sphere() -> None:
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
    threaded = build_evaluator("thread", workers=2, batch_size=2)
    process = build_evaluator("process", workers=2, batch_size=2)

    try:
        expected = sequential.evaluate(positions, objective)
        threaded_values = threaded.evaluate(positions, objective)
        process_values = process.evaluate(positions, objective)
    finally:
        threaded.close()
        process.close()

    assert np.allclose(threaded_values, expected)
    assert np.allclose(process_values, expected)
    assert threaded.last_stats.tasks_submitted == 2
    assert process.last_stats.tasks_submitted == 2
