"""
rastrigin.py

This module implements the Rastrigin benchmark function.

This implementation:
    - Supports both single input (d,) and batch input (n, d).
    - Is fully vectorized using Numpy.
    - Can therefore be used for:
        V0: Sequential PSO
        V1/V2: Thread/Process parallel evaluation
        V4: NumPy vectorized evaluation (implicit parallelism)
        
The function itself is stateless and purely functional.
All orchestration logic is handled by:
    - objectives/registry.py
    - core/ (PSO engine)
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
    x = np.asarray(x)  # Transform input to numpy array for vectorized operations

    single_input = (x.ndim == 1) # If x only has 1 point (1D), we will add a batch dimension to unify the computation with the case where x has multiple points (2D)
    if single_input:
        x = x[None, :] # Add a batch dimension to make x shape (1, d)

    d = x.shape[1] # Extract dimension d from the second axis of x (after ensuring x has shape (n, d))

    result = 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

    # Returns a scalar when the input is a single point and an array of shape (n) when the input is multiple points
    return float(result[0]) if single_input else result
