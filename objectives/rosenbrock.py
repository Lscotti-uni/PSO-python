"""
rosenbrock.py

This module implements the Rosenbrock benchmark function.

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
    x = np.asarray(x)  # Transform input to numpy array for vectorized operations

    single_input = (x.ndim == 1) # If x only has 1 point (1D), we will add a batch dimension to unify the computation with the case where x has multiple points (2D)
    if single_input:
        x = x[None, :] # Add a batch dimension to make x shape (1, d)

    # We must slice along the dimension axis (axis=1), not along the batch axis.
    xi = x[:, :-1]    # shape (n, d-1)
    xnext = x[:, 1:]  # shape (n, d-1)

    result = np.sum(100 * (xnext - xi**2)**2 + (1 - xi)**2, axis =1)  # shape (n,)

    # Returns a scalar when the input is a single point and an array of shape (n) when the input is multiple points
    return float(result[0]) if single_input else result