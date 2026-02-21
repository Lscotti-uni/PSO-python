import numpy as np

def ackley(x):
    """Ackley is a non-convex function.
    It has many local minima and one global minimum at x = (0, ..., 0), where f(x) = 0

    Args:
        x (array-like): Input vector of shape (d)

    Returns:
        float: Ackley function value at x
    """
    x = np.asarray(x)
    d = x.size  # d is required in Ackley because both terms are averaged per dimension

    part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))  # Average squared magnitude across d components
    part2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / d)  # Average cosine term across d components

    return part1 + part2 + 20 + np.e
