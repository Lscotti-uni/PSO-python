"""Configuration loading and validation for PSO experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("The configuration file must contain a top-level mapping.")
    return payload


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class PSOConfig:
    objective: str = "sphere"
    dimensions: int = 10
    swarm_size: int = 40
    iterations: int = 200
    inertia: float = 0.7298
    cognitive: float = 1.49618
    social: float = 1.49618
    topology: str = "global"
    bounds: tuple[float, float] | list[list[float]] | list[tuple[float, float]] = (-5.12, 5.12)
    boundary_strategy: str = "clamp"
    velocity_clamp: float | None = None
    tolerance: float = 1e-8
    stagnation_iters: int = 60
    enable_early_stopping: bool = True
    seed: int | None = 123
    strategy: str = "sequential"
    workers: int | None = None
    batch_size: int = 1
    track_trajectory: bool = False
    trajectory_stride: int = 1
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if self.swarm_size <= 0:
            raise ValueError("swarm_size must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.stagnation_iters < 0:
            raise ValueError("stagnation_iters cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.trajectory_stride <= 0:
            raise ValueError("trajectory_stride must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.velocity_clamp is not None and self.velocity_clamp <= 0:
            raise ValueError("velocity_clamp must be positive when provided")
        if self.topology.strip().lower() not in {"global", "gbest"}:
            raise ValueError("Only the global-best topology is supported in this project")
        if self.strategy.strip().lower() not in {"sequential", "seq", "v0", "thread", "threading", "threads", "v1", "process", "multiprocessing", "processes", "v2"}:
            raise ValueError("strategy must be one of: sequential, thread, process")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PSOConfig":
        aliases = {
            "dim": "dimensions",
            "dimension": "dimensions",
            "dims": "dimensions",
            "iters": "iterations",
            "max_iters": "iterations",
            "w": "inertia",
            "c1": "cognitive",
            "c2": "social",
            "boundary": "boundary_strategy",
            "boundary_policy": "boundary_strategy",
            "seed_value": "seed",
            "lower_bound": "lower_bounds",
            "upper_bound": "upper_bounds",
        }
        lowered = dict(data)

        if "lower_bounds" in lowered and "upper_bounds" in lowered:
            lower = lowered.pop("lower_bounds")
            upper = lowered.pop("upper_bounds")
            lowered["bounds"] = list(zip(lower, upper))

        payload: dict[str, Any] = {}
        allowed = {field.name for field in fields(cls)}
        for key, value in lowered.items():
            normalized = aliases.get(key, key)
            if normalized in allowed:
                payload[normalized] = value
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
