# Mi PSO - Benchmark Objectives

This project currently contains a set of benchmark objective functions, implemented in Python with NumPy and prepared for both scalar and batch evaluation.

Current repository status:
- `objectives/` module with vectorized objective functions.
- Central registry (`registry.py`) to access each objective by name.
- PSO engine and experiment runners are not included yet.

## Current Structure

- `objectives/ackley.py`
- `objectives/rastrigin.py`
- `objectives/rosenbrock.py`
- `objectives/sphere.py`
- `objectives/registry.py`
- `.gitignore`

## Available Objectives

| Key (`registry`) | Name | Default bounds | Global optimum |
|---|---|---|---|
| `sphere` | Sphere | `(-5.12, 5.12)` | `0.0` at `x = 0` |
| `rosenbrock` | Rosenbrock | `(-5.0, 10.0)` | `0.0` at `x = 1` |
| `rastrigin` | Rastrigin | `(-5.12, 5.12)` | `0.0` at `x = 0` |
| `ackley` | Ackley | `(-32.768, 32.768)` | `0.0` at `x = 0` |

## Requirements

- Python 3.9+
- NumPy

Minimal installation:

```bash
pip install numpy
```

## Quick Usage

### 1) Direct function call

```python
import numpy as np
from objectives.sphere import sphere

x = np.array([1.0, -2.0, 0.5])
print(sphere(x))
```

Batch input `(n, d)` is also supported:

```python
X = np.array([[1.0, 0.0], [0.5, -0.5]])
print(sphere(X))
```

### 2) Registry-based usage

```python
from objectives.registry import list_objectives, get_objective

print(list_objectives())
spec = get_objective("ackley")
value = spec.fn([0.0, 0.0, 0.0])

print(spec.name)
print(spec.default_bounds)
print(spec.known_optimum_value)
print(value)
```

## Implementation Notes

- All functions accept `(d,)` and `(n, d)` inputs.
- They return `float` for a single sample and `np.ndarray` for batches.
- `get_objective(name)` is case-insensitive.
