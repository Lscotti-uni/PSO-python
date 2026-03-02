import numpy as np

def sphere(x):
    """Sphere is a convex function.
    It has one global minimum at x = (0, ..., 0), where f(x) = 0

    Args:
        x (array-like): Input vector of shape (d)

    Returns:
        float: Sphere function value at x
    """
    x = np.asarray(x)
    # No explicit d is needed in standard Sphere: dimension already enters through the sum of all components
    return np.sum(x**2)
