import numpy as np

def rastrigin(x):
    """Rastrigin is a non-convex function.
    It has many local minima and one global minimum at x = (0, ..., 0), where f(x) = 0

    Args:
        x (array-like): Input vector of shape (d)

    Returns:
        float: Rastrigin function value at x
    """
    x = np.asarray(x)
    d = x.size  # d is required in standard Rastrigin because the constant term is 10*d

    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
