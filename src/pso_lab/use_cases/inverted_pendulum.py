"""Inverted-pendulum-on-cart use case with a cascade PID controller.

The controller has six gains (three for the angle loop, three for the cart-position
loop). PSO searches the gain space to minimise an integral-of-squared-error
(ISE) cost subject to actuator saturation. The cost surface is multimodal and
includes a large failure penalty when the pole tips beyond a recoverable angle.

The dynamics use a fixed-step RK4 integrator so PSO runs are reproducible and
inexpensive enough to evaluate thousands of candidate gain sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PENDULUM_PID_NAME = "Inverted pendulum PID tuning"
PENDULUM_PID_DIMENSIONS = 6
# Search box: angle-loop gains can be larger because the pole is naturally
# unstable; cart-loop gains are smaller to avoid wrestling the angle loop.
DEFAULT_BOUNDS_PER_DIM: list[tuple[float, float]] = [
    (0.0, 200.0),   # Kp_theta
    (0.0, 60.0),    # Ki_theta
    (0.0, 60.0),    # Kd_theta
    (-50.0, 50.0),  # Kp_x
    (-20.0, 20.0),  # Ki_x
    (-30.0, 30.0),  # Kd_x
]
# Hand-tuned conservative gains used as a sanity baseline for PSO results.
# These keep the pole upright without aggressive integral action; PSO is
# expected to find gain sets that beat them on the ISE cost.
PENDULUM_PID_REFERENCE_GAINS = np.array([150.0, 0.0, 20.0, -5.0, 0.0, -5.0])


@dataclass(slots=True, frozen=True)
class PhysicalParams:
    cart_mass: float = 1.0           # kg
    pole_mass: float = 0.1           # kg
    pole_length: float = 0.5         # m, distance from pivot to centre of mass
    gravity: float = 9.81            # m/s^2
    friction: float = 0.05           # cart viscous friction


@dataclass(slots=True, frozen=True)
class SimParams:
    horizon_s: float = 4.0
    dt: float = 0.01
    initial_angle_rad: float = 0.20          # ~11.5 degrees off vertical
    initial_angular_velocity: float = 0.0
    initial_position: float = 0.0
    initial_velocity: float = 0.0
    target_position: float = 0.0
    force_limit: float = 15.0                # N, actuator saturation
    failure_angle_rad: float = 0.6           # ~34 degrees -> unrecoverable


@dataclass(slots=True, frozen=True)
class CostWeights:
    angle: float = 50.0
    position: float = 1.0
    angular_velocity: float = 0.5
    control_effort: float = 0.001
    failure_penalty: float = 1.0e4


DEFAULT_PHYSICAL_PARAMS = PhysicalParams()
DEFAULT_SIM_PARAMS = SimParams()
DEFAULT_WEIGHTS = CostWeights()


def _state_derivative(
    state: np.ndarray,
    force: float,
    params: PhysicalParams,
) -> np.ndarray:
    # Standard nonlinear cart-pole equations (Tedrake's lecture-notes form).
    x, x_dot, theta, theta_dot = state
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    total_mass = params.cart_mass + params.pole_mass
    pole_ml = params.pole_mass * params.pole_length

    temp = (
        force + pole_ml * theta_dot * theta_dot * sin_t - params.friction * x_dot
    ) / total_mass
    theta_acc = (params.gravity * sin_t - cos_t * temp) / (
        params.pole_length * (4.0 / 3.0 - params.pole_mass * cos_t * cos_t / total_mass)
    )
    x_acc = temp - pole_ml * theta_acc * cos_t / total_mass
    return np.array([x_dot, x_acc, theta_dot, theta_acc], dtype=float)


def _rk4_step(
    state: np.ndarray,
    force: float,
    dt: float,
    params: PhysicalParams,
) -> np.ndarray:
    k1 = _state_derivative(state, force, params)
    k2 = _state_derivative(state + 0.5 * dt * k1, force, params)
    k3 = _state_derivative(state + 0.5 * dt * k2, force, params)
    k4 = _state_derivative(state + dt * k3, force, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _split_gains(gains: Iterable[float]) -> tuple[float, float, float, float, float, float]:
    gains = np.asarray(list(gains), dtype=float)
    if gains.shape != (PENDULUM_PID_DIMENSIONS,):
        raise ValueError(
            f"Pendulum PID expects {PENDULUM_PID_DIMENSIONS} gains, received {gains.shape}."
        )
    return tuple(gains)  # type: ignore[return-value]


def simulate_pendulum(
    gains: Iterable[float],
    *,
    physical: PhysicalParams = DEFAULT_PHYSICAL_PARAMS,
    sim: SimParams = DEFAULT_SIM_PARAMS,
) -> dict[str, np.ndarray]:
    """Run a single closed-loop simulation and return the recorded trajectories."""

    Kp_t, Ki_t, Kd_t, Kp_x, Ki_x, Kd_x = _split_gains(gains)
    n_steps = int(round(sim.horizon_s / sim.dt))
    state = np.array(
        [sim.initial_position, sim.initial_velocity, sim.initial_angle_rad, sim.initial_angular_velocity],
        dtype=float,
    )

    times = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, 4))
    forces = np.zeros(n_steps + 1)
    states[0] = state

    integral_theta = 0.0
    integral_x = 0.0
    failed = False
    for step in range(n_steps):
        x, x_dot, theta, theta_dot = state
        integral_theta += theta * sim.dt
        integral_x += (x - sim.target_position) * sim.dt
        # Angle-loop PID stabilises the pole; cart-loop PID nudges the cart to
        # the target position. Their sum is saturated at the actuator limit.
        force_theta = Kp_t * theta + Ki_t * integral_theta + Kd_t * theta_dot
        force_x = Kp_x * (x - sim.target_position) + Ki_x * integral_x + Kd_x * x_dot
        force = np.clip(force_theta + force_x, -sim.force_limit, sim.force_limit)
        forces[step] = force

        state = _rk4_step(state, force, sim.dt, physical)
        states[step + 1] = state
        times[step + 1] = (step + 1) * sim.dt

        if abs(state[2]) > sim.failure_angle_rad:
            # Once the pole falls past the recoverable angle we mark the run as
            # failed and stop integrating; downstream the cost will add the
            # failure penalty proportional to the remaining horizon.
            failed = True
            states[step + 2 :] = state
            times[step + 2 :] = np.arange(step + 2, n_steps + 1) * sim.dt
            forces[step + 1 :] = force
            break

    return {
        "time": times,
        "states": states,
        "forces": forces,
        "failed": failed,
    }


def pendulum_pid_cost(
    gains,
    *,
    physical: PhysicalParams = DEFAULT_PHYSICAL_PARAMS,
    sim: SimParams = DEFAULT_SIM_PARAMS,
    weights: CostWeights = DEFAULT_WEIGHTS,
) -> float | np.ndarray:
    """ISE-based cost compatible with PSO's batched and single-input evaluators."""

    array = np.asarray(gains, dtype=float)
    single_input = array.ndim == 1
    if single_input:
        array = array[None, :]

    costs = np.zeros(array.shape[0], dtype=float)
    for idx in range(array.shape[0]):
        record = simulate_pendulum(array[idx], physical=physical, sim=sim)
        states = record["states"]
        forces = record["forces"]
        # ISE-like cost on angle, cart position, and angular rate, plus a small
        # actuator-effort term to discourage chattering controllers.
        cost = (
            weights.angle * float(np.sum(states[:, 2] ** 2))
            + weights.position * float(np.sum((states[:, 0] - sim.target_position) ** 2))
            + weights.angular_velocity * float(np.sum(states[:, 3] ** 2))
            + weights.control_effort * float(np.sum(forces ** 2))
        ) * sim.dt
        if record["failed"]:
            cost += weights.failure_penalty
        costs[idx] = cost

    return float(costs[0]) if single_input else costs


def plot_response(
    gains,
    *,
    output_path,
    physical: PhysicalParams = DEFAULT_PHYSICAL_PARAMS,
    sim: SimParams = DEFAULT_SIM_PARAMS,
    title: str | None = None,
) -> None:
    """Render a 2x2 panel with angle, position, force, and angular velocity."""

    import matplotlib.pyplot as plt

    record = simulate_pendulum(gains, physical=physical, sim=sim)
    times = record["time"]
    states = record["states"]
    forces = record["forces"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes[0, 0].plot(times, np.degrees(states[:, 2]), color="#005f73")
    axes[0, 0].axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
    axes[0, 0].set_title("Pole angle (deg)")

    axes[0, 1].plot(times, states[:, 0], color="#0a9396")
    axes[0, 1].axhline(sim.target_position, color="grey", linewidth=0.6, linestyle="--")
    axes[0, 1].set_title("Cart position (m)")

    axes[1, 0].plot(times, forces, color="#ee9b00")
    axes[1, 0].axhline(sim.force_limit, color="grey", linewidth=0.6, linestyle="--")
    axes[1, 0].axhline(-sim.force_limit, color="grey", linewidth=0.6, linestyle="--")
    axes[1, 0].set_title("Control force (N)")

    axes[1, 1].plot(times, states[:, 3], color="#ae2012")
    axes[1, 1].set_title("Angular velocity (rad/s)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")

    fig.suptitle(title or "Inverted pendulum response", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
