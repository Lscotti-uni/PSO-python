"""
ackley.py

This module implements the Ackley benchmark function.

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
    x = np.asarray(x) # Transform input to numpy array for vectorized operations
    
    single_input = (x.ndim == 1) # If x only has 1 point (1D), we will add a batch dimension to unify the computation with the case where x has multiple points (2D)
    if single_input:
        x = x[None, :] # Add a batch dimension to make x shape (1, d)

    d = x.shape[1] # Extract dimension d from the second axis of x (after ensuring x has shape (n, d))

    part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / d))  # Average squared magnitude across d components
    part2 = -np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / d)  # Average cosine term across d components

    result = part1 + part2 + 20 + np.e

    # Returns a scalar when the input is a single point and an array of shape (n) when the input is multiple points
    return float(result[0]) if single_input else result 




