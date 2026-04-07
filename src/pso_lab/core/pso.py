"""Canonical PSO engine with instrumentation and optional trajectory tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np

from pso_lab.core.bounds import BoundsPolicy, expand_bounds
from pso_lab.core.config import PSOConfig
from pso_lab.core.state import IterationMetrics, SwarmState
from pso_lab.core.stopping import should_stop
from pso_lab.core.topology import TopologyStrategy
from pso_lab.parallel.base import FitnessEvaluator
from pso_lab.utils.logging import log_kv


@dataclass(slots=True)
class PSOResult:
    best_position: np.ndarray
    best_value: float
    history: list[dict]
    state: SwarmState
    converged: bool
    stop_reason: str
    trajectory: list[np.ndarray] = field(default_factory=list)
    best_position_trace: list[np.ndarray] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class PSO:
    def __init__(
        self,
        config: PSOConfig,
        objective: Callable[[np.ndarray], float | np.ndarray],
        evaluator: FitnessEvaluator,
        bounds_policy: BoundsPolicy,
        topology: TopologyStrategy,
        logger=None,
    ) -> None:
        self.config = config
        self.objective = objective
        self.evaluator = evaluator
        self.bounds_policy = bounds_policy
        self.topology = topology
        self.logger = logger
        # A dedicated generator keeps experiments reproducible across strategies.
        self.rng = np.random.default_rng(config.seed)
        self.lower_bounds, self.upper_bounds = expand_bounds(config.bounds, config.dimensions)
        self.span = self.upper_bounds - self.lower_bounds
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.swarm_size < 2:
            raise ValueError("swarm_size must be at least 2")

    def _apply_velocity_clamp(self, velocities: np.ndarray) -> np.ndarray:
        if self.config.velocity_clamp is None:
            return velocities
        limits = self.span * self.config.velocity_clamp
        return np.clip(velocities, -limits, limits)

    def _record_trajectory(
        self,
        state: SwarmState,
        trajectory: list[np.ndarray],
        best_position_trace: list[np.ndarray],
    ) -> None:
        # Trajectory capture is optional because it increases storage cost for
        # long runs and is only required for visualisation workflows.
        if not self.config.track_trajectory:
            return
        if state.iteration % self.config.trajectory_stride != 0:
            return
        trajectory.append(state.positions.copy())
        best_position_trace.append(state.global_best_position.copy())

    def _evaluate(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(self.evaluator.evaluate(positions, self.objective), dtype=float)
        if values.shape != (self.config.swarm_size,):
            raise ValueError(
                f"Evaluator must return shape ({self.config.swarm_size},), received {values.shape}."
            )
        return values

    def _initialize_state(self) -> tuple[SwarmState, IterationMetrics]:
        # Particles start uniformly inside the feasible box so every run begins
        # from a valid swarm state.
        positions = self.rng.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.config.swarm_size, self.config.dimensions),
        )
        # Initial velocities are scaled to the search-space span to avoid
        # overly aggressive first moves in large domains.
        velocities = self.rng.uniform(
            low=-0.1 * self.span,
            high=0.1 * self.span,
            size=(self.config.swarm_size, self.config.dimensions),
        )

        update_start = perf_counter()
        update_time = perf_counter() - update_start
        eval_start = perf_counter()
        fitness = self._evaluate(positions)
        eval_time = perf_counter() - eval_start

        pbest_positions = positions.copy()
        pbest_values = fitness.copy()
        best_idx = int(np.argmin(pbest_values))
        gbest_position = pbest_positions[best_idx].copy()
        gbest_value = float(pbest_values[best_idx])

        state = SwarmState(
            positions=positions,
            velocities=velocities,
            personal_best_positions=pbest_positions,
            personal_best_values=pbest_values,
            global_best_position=gbest_position,
            global_best_value=gbest_value,
            iteration=0,
        )
        stats = self.evaluator.last_stats
        metrics = IterationMetrics(
            iteration=0,
            best_fitness=gbest_value,
            mean_fitness=float(np.mean(fitness)),
            min_fitness_current=float(np.min(fitness)),
            eval_time=eval_time,
            update_time=update_time,
            overhead_time=stats.overhead_time,
            worker_time_total=stats.worker_time_total,
            critical_path_time=stats.critical_path_time,
            tasks_submitted=stats.tasks_submitted,
            total_iter_time=eval_time + update_time,
            gbest_improved=True,
            previous_gbest=float("inf"),
        )
        return state, metrics

    def _update_loop(self, state: SwarmState, social_best_positions: np.ndarray) -> None:
        new_positions = np.empty_like(state.positions)
        new_velocities = np.empty_like(state.velocities)

        for particle_idx in range(self.config.swarm_size):
            r1 = self.rng.random(self.config.dimensions)
            r2 = self.rng.random(self.config.dimensions)
            cognitive = (
                self.config.cognitive
                * r1
                * (state.personal_best_positions[particle_idx] - state.positions[particle_idx])
            )
            social = (
                self.config.social
                * r2
                * (social_best_positions[particle_idx] - state.positions[particle_idx])
            )
            # Canonical PSO update: inertia preserves momentum, while cognitive
            # and social terms pull the particle toward promising regions.
            velocity = self.config.inertia * state.velocities[particle_idx] + cognitive + social
            new_velocities[particle_idx] = velocity
            new_positions[particle_idx] = state.positions[particle_idx] + velocity

        state.velocities = self._apply_velocity_clamp(new_velocities)
        state.positions = new_positions

    def _step(self, state: SwarmState) -> IterationMetrics:
        iter_start = perf_counter()
        previous_gbest = float(state.global_best_value)

        update_start = perf_counter()
        # The topology decides which social leader each particle follows. With
        # global-best, every particle is pulled toward the same best solution.
        social_best_positions = self.topology.social_best_positions(state)
        self._update_loop(state, social_best_positions)
        # Boundary handling is delegated to a policy so the core algorithm does
        # not need to know whether we clamp, reflect, or use another strategy.
        state.positions, state.velocities = self.bounds_policy.apply(
            state.positions,
            state.velocities,
            self.lower_bounds,
            self.upper_bounds,
        )
        update_time = perf_counter() - update_start

        eval_start = perf_counter()
        fitness = self._evaluate(state.positions)
        eval_time = perf_counter() - eval_start

        improved_personal = fitness < state.personal_best_values
        if np.any(improved_personal):
            # Personal bests act as a stable archive of the best position each
            # particle has ever seen, independent of its current position.
            state.personal_best_positions[improved_personal] = state.positions[improved_personal]
            state.personal_best_values[improved_personal] = fitness[improved_personal]

        # The global best is selected from the personal-best archive, not from
        # raw current positions, which preserves the monotonic gbest property.
        best_idx = int(np.argmin(state.personal_best_values))
        candidate_value = float(state.personal_best_values[best_idx])
        gbest_improved = candidate_value < state.global_best_value
        if gbest_improved:
            state.global_best_value = candidate_value
            state.global_best_position = state.personal_best_positions[best_idx].copy()

        state.iteration += 1
        total_iter_time = perf_counter() - iter_start
        stats = self.evaluator.last_stats
        return IterationMetrics(
            iteration=state.iteration,
            best_fitness=float(state.global_best_value),
            mean_fitness=float(np.mean(fitness)),
            min_fitness_current=float(np.min(fitness)),
            eval_time=eval_time,
            update_time=update_time,
            overhead_time=stats.overhead_time,
            worker_time_total=stats.worker_time_total,
            critical_path_time=stats.critical_path_time,
            tasks_submitted=stats.tasks_submitted,
            total_iter_time=total_iter_time,
            gbest_improved=gbest_improved,
            previous_gbest=previous_gbest,
        )

    def run(self) -> PSOResult:
        run_start = perf_counter()
        history: list[dict] = []
        trajectory: list[np.ndarray] = []
        best_position_trace: list[np.ndarray] = []

        state, initial_metrics = self._initialize_state()
        history.append(initial_metrics.to_dict())
        self._record_trajectory(state, trajectory, best_position_trace)

        if self.logger is not None:
            log_kv(
                self.logger,
                logging.INFO,
                "initial_state",
                iteration=0,
                best_fitness=f"{state.global_best_value:.6e}",
                strategy=self.evaluator.name,
            )

        no_improve_iters = 0
        stop_reason = should_stop(state.global_best_value, no_improve_iters, self.config)
        converged = stop_reason == "tolerance"

        while stop_reason is None and state.iteration < self.config.iterations:
            metrics = self._step(state)
            history.append(metrics.to_dict())
            self._record_trajectory(state, trajectory, best_position_trace)

            if metrics.gbest_improved:
                no_improve_iters = 0
            else:
                no_improve_iters += 1

            # We log the first step, periodic checkpoints, and any actual
            # improvement so long runs remain readable without losing signal.
            should_log = (
                state.iteration == 1
                or state.iteration % self.config.log_every == 0
                or metrics.gbest_improved
            )
            if should_log:
                log_kv(
                    self.logger,
                    logging.INFO,
                    "iteration",
                    iteration=state.iteration,
                    best_fitness=f"{state.global_best_value:.6e}",
                    eval_time=f"{metrics.eval_time:.6f}",
                    update_time=f"{metrics.update_time:.6f}",
                    overhead=f"{metrics.overhead_time:.6f}",
                    improved=metrics.gbest_improved,
                )

            stop_reason = should_stop(state.global_best_value, no_improve_iters, self.config)
            converged = stop_reason == "tolerance"

        if stop_reason is None:
            stop_reason = "max_iterations"

        total_run_time = perf_counter() - run_start
        # Store the full run time on the last history row so CSV exports keep a
        # self-contained record of the final aggregate timing.
        history[-1]["total_run_time"] = total_run_time

        return PSOResult(
            best_position=state.global_best_position.copy(),
            best_value=float(state.global_best_value),
            history=history,
            state=state.copy(),
            converged=converged,
            stop_reason=stop_reason,
            trajectory=trajectory,
            best_position_trace=best_position_trace,
            timings={
                "total_run_time": total_run_time,
                "total_eval_time": float(sum(item["eval_time"] for item in history)),
                "total_update_time": float(sum(item["update_time"] for item in history)),
                "total_overhead_time": float(sum(item["overhead_time"] for item in history)),
            },
        )
