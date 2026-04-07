"""
ackley.py

This module implements the Ackley benchmark function.

This implementation supports both single input `(d,)` and batch input
`(n, d)`, which lets the same objective function work with the sequential,
threaded, and process-based evaluators.
"""
import numpy as np

NAME = "Ackley"
DEFAULT_BOUNDS = (-32.768, 32.768)  # Standard bounds for Ackley function
KNOWN_OPTIMUM = {
    "position": lambda d: np.zeros(d),  # Global minimum at (0, ..., 0) for any dimension d
    "value": 0.0
}

def ackley(x):
    """Ackley is a non-convex function.
    It has many local minima and one global minimum at x = (0, ..., 0), where f(x) = 0
    Recommended domain: [-32.768, 32.768] ^d

    Args:
        x (array-like): Input vector of shape (d)

    Returns:
        float: Ackley function value at x when x has shape (d,)
        np.ndarray: Ackley function values of shape (n,) when x has shape (n, d)
    """
    x = np.asarray(x)
    
    single_input = x.ndim == 1
    if single_input:
        x = x[None, :]

    d = x.shape[1]

    part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / d))
    part2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / d)

    result = part1 + part2 + 20 + np.e

    return float(result[0]) if single_input else result 



