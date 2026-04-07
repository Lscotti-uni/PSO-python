"""Core PSO engine exports."""

from pso_lab.core.bounds import ClampBoundsPolicy, ReflectBoundsPolicy, build_bounds_policy, expand_bounds
from pso_lab.core.config import PSOConfig, deep_merge, load_yaml
from pso_lab.core.pso import PSO, PSOResult
from pso_lab.core.state import IterationMetrics, SwarmState
from pso_lab.core.topology import GlobalBestTopology, build_topology

__all__ = [
    "ClampBoundsPolicy",
    "GlobalBestTopology",
    "IterationMetrics",
    "PSO",
    "PSOConfig",
    "PSOResult",
    "ReflectBoundsPolicy",
    "SwarmState",
    "build_bounds_policy",
    "build_topology",
    "deep_merge",
    "expand_bounds",
    "load_yaml",
]
