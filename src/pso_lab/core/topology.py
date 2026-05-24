"""Topology strategies that define each particle's social neighborhood."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from pso_lab.core.state import SwarmState


class TopologyStrategy(Protocol):
    name: str

    def social_best_positions(self, state: SwarmState) -> np.ndarray:
        ...


@dataclass(slots=True)
class GlobalBestTopology:
    name: str = "global"

    def social_best_positions(self, state: SwarmState) -> np.ndarray:
        # Every particle sees the same global leader in gbest topology.
        return np.repeat(state.global_best_position[np.newaxis, :], state.positions.shape[0], axis=0)


@dataclass(slots=True)
class RingTopology:
    name: str = "ring"
    neighborhood_size: int = 2
    _neighbors: np.ndarray | None = field(default=None, init=False, repr=False)

    def _build_neighbors(self, swarm_size: int) -> np.ndarray:
        # k neighbors on each side wrap around the ring.
        k = max(1, self.neighborhood_size)
        offsets = np.arange(-k, k + 1)
        indices = np.arange(swarm_size)[:, None] + offsets[None, :]
        return indices % swarm_size

    def social_best_positions(self, state: SwarmState) -> np.ndarray:
        swarm_size = state.positions.shape[0]
        if self._neighbors is None or self._neighbors.shape[0] != swarm_size:
            self._neighbors = self._build_neighbors(swarm_size)
        # For each particle, choose the best pbest among its ring neighbors.
        neighbor_values = state.personal_best_values[self._neighbors]
        best_relative = np.argmin(neighbor_values, axis=1)
        best_indices = self._neighbors[np.arange(swarm_size), best_relative]
        return state.personal_best_positions[best_indices]


@dataclass(slots=True)
class VonNeumannTopology:
    name: str = "von_neumann"
    _neighbors: np.ndarray | None = field(default=None, init=False, repr=False)
    _shape: tuple[int, int] | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _grid_shape(swarm_size: int) -> tuple[int, int]:
        # Pick the rectangle closest to a square; pads with toroidal wrap if needed.
        for rows in range(int(np.sqrt(swarm_size)), 0, -1):
            if swarm_size % rows == 0:
                return rows, swarm_size // rows
        return 1, swarm_size

    def _build_neighbors(self, swarm_size: int) -> np.ndarray:
        rows, cols = self._grid_shape(swarm_size)
        self._shape = (rows, cols)
        indices = np.arange(rows * cols).reshape(rows, cols)
        # Toroidal 5-cell stencil: self + N/S/E/W neighbors.
        north = np.roll(indices, 1, axis=0)
        south = np.roll(indices, -1, axis=0)
        west = np.roll(indices, 1, axis=1)
        east = np.roll(indices, -1, axis=1)
        stacked = np.stack([indices, north, south, west, east], axis=-1).reshape(rows * cols, 5)
        if rows * cols > swarm_size:
            stacked = stacked[:swarm_size]
        return np.mod(stacked, swarm_size)

    def social_best_positions(self, state: SwarmState) -> np.ndarray:
        swarm_size = state.positions.shape[0]
        if self._neighbors is None or self._neighbors.shape[0] != swarm_size:
            self._neighbors = self._build_neighbors(swarm_size)
        neighbor_values = state.personal_best_values[self._neighbors]
        best_relative = np.argmin(neighbor_values, axis=1)
        best_indices = self._neighbors[np.arange(swarm_size), best_relative]
        return state.personal_best_positions[best_indices]


def build_topology(name: str, neighborhood_size: int = 2) -> TopologyStrategy:
    normalized = name.strip().lower()
    if normalized in {"global", "gbest"}:
        return GlobalBestTopology()
    if normalized in {"ring", "lbest"}:
        return RingTopology(neighborhood_size=neighborhood_size)
    if normalized in {"von_neumann", "vonneumann", "neumann"}:
        return VonNeumannTopology()
    raise KeyError(f"Unknown topology '{name}'. Supported: global, ring, von_neumann.")
