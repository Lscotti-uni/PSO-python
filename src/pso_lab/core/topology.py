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


def build_topology(name: str, neighborhood_size: int = 3) -> TopologyStrategy:
    normalized = name.strip().lower()
    if normalized in {"global", "gbest"}:
        return GlobalBestTopology()
    raise KeyError(f"Unknown topology '{name}'. Only 'global' is supported.")
