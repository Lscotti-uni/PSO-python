"""
sphere.py

This module implements the Sphere benchmark function.

This implementation supports both single input `(d,)` and batch input
`(n, d)`, which lets the same objective function work with the sequential,
threaded, and process-based evaluators.
"""
import numpy as np

NAME = "Sphere"
DEFAULT_BOUNDS = (-5.12, 5.12)  # Standard bounds for Sphere function
KNOWN_OPTIMUM = {
    "position": lambda d: np.zeros(d),  # Global minimum at (0, ..., 0) for any dimension d
    "value": 0.0
}

def sphere(x):
    """Sphere is a convex function.
    It has one global minimum at x = (0, ..., 0), where f(x) = 0

    Args:
        x (array-like): Input vector of shape (d)

    Returns:
        float: Sphere function value at x
    """
    x = np.asarray(x)

    single_input = x.ndim == 1
    if single_input:
        x = x[None, :]

    result = np.sum(x**2, axis=1)

    return float(result[0]) if single_input else result
