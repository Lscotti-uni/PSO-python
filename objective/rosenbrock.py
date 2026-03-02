import numpy as np

def rosenbrock(x):
    """Rosenbrock is a non-convex function.
    It has one global minimum at x = (1, ..., 1), where f(x) = 0

    Args:
        x (array-like): Input vector of shape (d)

    Returns:
        float: Rosenbrock function value at x
    """
    x = np.asarray(x)
    # No explicit d is needed in standard Rosenbrock: slices x[:-1] and x[1:] generate the (d-1) coupled terms

    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
