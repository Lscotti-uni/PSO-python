"""
rosenbrock.py

This module implements the Rosenbrock benchmark function.

This implementation supports both single input `(d,)` and batch input
`(n, d)`, which lets the same objective function work with the sequential,
threaded, and process-based evaluators.
"""
import numpy as np

NAME = "Rosenbrock"
DEFAULT_BOUNDS = (-5, 10)  # Standard bounds for Rosenbrock function
KNOWN_OPTIMUM = {
    "position": lambda d: np.ones(d),  # Global minimum at (1, ..., 1) for any dimension d
    "value": 0.0
}

def rosenbrock(x):
    """Rosenbrock is a non-convex function.
    It has one global minimum at x = (1, ..., 1), where f(x) = 0

    Args:
        x (array-like): Input vector of shape (d)

    Returns:
        float: Rosenbrock function value at x
    """
    x = np.asarray(x)

    single_input = x.ndim == 1
    if single_input:
        x = x[None, :]

    xi = x[:, :-1]
    xnext = x[:, 1:]

    result = np.sum(100 * (xnext - xi**2)**2 + (1 - xi)**2, axis=1)

    return float(result[0]) if single_input else result
