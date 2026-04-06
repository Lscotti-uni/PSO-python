"""Boundary policies and utilities for box-constrained PSO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def expand_bounds(
    bounds: tuple[float, float] | list[list[float]] | list[tuple[float, float]],
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(bounds, dtype=float)
    if raw.shape == (2,):
        # A single pair is broadcast to every dimension for the common
        # "same box everywhere" case.
        lower = np.full(dimensions, raw[0], dtype=float)
        upper = np.full(dimensions, raw[1], dtype=float)
    elif raw.shape == (dimensions, 2):
        lower = raw[:, 0].astype(float)
        upper = raw[:, 1].astype(float)
    else:
        raise ValueError("bounds must be shaped as (2,) or (dimensions, 2)")

    if np.any(lower >= upper):
        raise ValueError("Each lower bound must be strictly smaller than the upper bound")
    return lower, upper


class BoundsPolicy(Protocol):
    name: str

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(slots=True)
class ClampBoundsPolicy:
    name: str = "clamp"

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(positions, lower, upper)
        hit_bounds = clipped != positions
        # Zeroing the offending velocity components prevents particles from
        # repeatedly slamming into the same boundary in consecutive steps.
        adjusted_velocities = np.where(hit_bounds, 0.0, velocities)
        return clipped, adjusted_velocities


@dataclass(slots=True)
class ReflectBoundsPolicy:
    name: str = "reflect"

    def apply(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        reflected_positions = positions.copy()
        reflected_velocities = velocities.copy()

        # Two passes are enough for this project because velocity clamping and
        # practical PSO step sizes keep overshoots reasonably bounded.
        for _ in range(2):
            low_mask = reflected_positions < lower
            if np.any(low_mask):
                reflected_positions = np.where(low_mask, 2.0 * lower - reflected_positions, reflected_positions)
                reflected_velocities = np.where(low_mask, -reflected_velocities, reflected_velocities)

            high_mask = reflected_positions > upper
            if np.any(high_mask):
                reflected_positions = np.where(high_mask, 2.0 * upper - reflected_positions, reflected_positions)
                reflected_velocities = np.where(high_mask, -reflected_velocities, reflected_velocities)

        reflected_positions = np.clip(reflected_positions, lower, upper)
        return reflected_positions, reflected_velocities


def build_bounds_policy(name: str) -> BoundsPolicy:
    normalized = name.strip().lower()
    if normalized == "clamp":
        return ClampBoundsPolicy()
    if normalized == "reflect":
        return ReflectBoundsPolicy()
    raise KeyError(f"Unknown boundary strategy '{name}'. Use 'clamp' or 'reflect'.")
