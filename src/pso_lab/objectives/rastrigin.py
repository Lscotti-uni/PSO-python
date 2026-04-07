"""
rastrigin.py

This module implements the Rastrigin benchmark function.

This implementation supports both single input `(d,)` and batch input
`(n, d)`, which lets the same objective function work with the sequential,
threaded, and process-based evaluators.
"""
import numpy as np

NAME = "Rastrigin"
DEFAULT_BOUNDS = (-5.12, 5.12)  # Standard bounds for Rastrigin function
KNOWN_OPTIMUM = {
    "position": lambda d: np.zeros(d),  # Global minimum at (0, ..., 0) for any dimension d
    "value": 0.0
}

def rastrigin(x):
    """Rastrigin is a non-convex function.
    It has many local minima and one global minimum at x = (0, ..., 0), where f(x) = 0

    Args:
        x (array-like): Input vector of shape (d)

    Returns:
        float: Rastrigin function value at x
    """
    x = np.asarray(x)

    single_input = x.ndim == 1
    if single_input:
        x = x[None, :]

    d = x.shape[1]

    result = 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

    return float(result[0]) if single_input else result
