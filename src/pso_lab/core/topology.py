"""Topology strategies that define each particle's social neighborhood."""

from __future__ import annotations

from dataclasses import dataclass
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
    neighborhood_size: int = 3
    name: str = "ring"

    def __post_init__(self) -> None:
        # Ring neighborhoods must be odd-sized so each particle has a symmetric
        # number of neighbours on both sides plus itself.
        if self.neighborhood_size < 3:
            self.neighborhood_size = 3
        if self.neighborhood_size % 2 == 0:
            self.neighborhood_size += 1

    def social_best_positions(self, state: SwarmState) -> np.ndarray:
        swarm_size = state.positions.shape[0]
        radius = self.neighborhood_size // 2
        neighborhood_best = np.empty_like(state.positions)

        for particle_idx in range(swarm_size):
            # Modulo indexing wraps the neighborhood around the array, turning
            # the swarm into a logical ring instead of a line with edges.
            indices = [(particle_idx + offset) % swarm_size for offset in range(-radius, radius + 1)]
            local_best_idx = indices[int(np.argmin(state.personal_best_values[indices]))]
            neighborhood_best[particle_idx] = state.personal_best_positions[local_best_idx]

        return neighborhood_best


def build_topology(name: str, neighborhood_size: int = 3) -> TopologyStrategy:
    normalized = name.strip().lower()
    if normalized in {"global", "gbest"}:
        return GlobalBestTopology()
    if normalized in {"ring", "local", "lbest"}:
        return RingTopology(neighborhood_size=neighborhood_size)
    raise KeyError(f"Unknown topology '{name}'. Use 'global' or 'ring'.")
