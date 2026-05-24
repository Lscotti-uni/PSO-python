"""Smoke tests for the global, ring, and von Neumann topologies."""

from __future__ import annotations

import numpy as np

from pso_lab.core.state import SwarmState
from pso_lab.core.topology import build_topology


def _make_state(swarm_size: int, dim: int) -> SwarmState:
    rng = np.random.default_rng(0)
    pbest_positions = rng.uniform(-1.0, 1.0, size=(swarm_size, dim))
    pbest_values = rng.uniform(0.1, 1.0, size=swarm_size)
    best_idx = int(np.argmin(pbest_values))
    return SwarmState(
        positions=rng.uniform(-1.0, 1.0, size=(swarm_size, dim)),
        velocities=np.zeros((swarm_size, dim)),
        personal_best_positions=pbest_positions,
        personal_best_values=pbest_values,
        global_best_position=pbest_positions[best_idx].copy(),
        global_best_value=float(pbest_values[best_idx]),
        iteration=0,
    )


def test_global_topology_returns_same_leader_for_all_particles() -> None:
    state = _make_state(swarm_size=20, dim=4)
    topology = build_topology("global")
    leaders = topology.social_best_positions(state)
    assert leaders.shape == (20, 4)
    assert np.allclose(leaders[0], state.global_best_position)
    assert np.allclose(leaders, leaders[0])


def test_ring_topology_assigns_best_neighbour_inside_window() -> None:
    state = _make_state(swarm_size=12, dim=3)
    topology = build_topology("ring", neighborhood_size=2)
    leaders = topology.social_best_positions(state)
    assert leaders.shape == (12, 3)
    # Each leader must be the pbest of one of the ring neighbours, so its
    # position must appear in the personal_best_positions array.
    for row in leaders:
        match = np.any(np.all(np.isclose(state.personal_best_positions, row), axis=1))
        assert match


def test_von_neumann_topology_uses_a_five_cell_stencil() -> None:
    state = _make_state(swarm_size=16, dim=3)
    topology = build_topology("von_neumann")
    leaders = topology.social_best_positions(state)
    assert leaders.shape == (16, 3)
    # Cached neighbour table must have exactly 5 entries per particle (self + N/S/E/W).
    cached = topology._neighbors  # type: ignore[attr-defined]
    assert cached is not None
    assert cached.shape == (16, 5)
