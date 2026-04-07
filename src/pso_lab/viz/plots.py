"""Plot and animation utilities for convergence curves and swarm trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot_convergence(history: list[dict], output_path: str | Path, *, title: str = "Convergence curve") -> None:
    iterations = [row["iteration"] for row in history]
    best = [row["best_fitness"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, best, color="#005f73", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _contour_data(
    objective: Callable[[np.ndarray], float | np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    resolution: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The contour grid is precomputed once and then reused for every frame so
    # 2D animations do not reevaluate the full objective surface repeatedly.
    lower, upper = bounds
    xs = np.linspace(lower[0], upper[0], resolution)
    ys = np.linspace(lower[1], upper[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = np.asarray(objective(grid), dtype=float).reshape(xx.shape)
    return xx, yy, zz


def save_swarm_frames(
    *,
    objective: Callable[[np.ndarray], float | np.ndarray],
    trajectory: list[np.ndarray],
    best_trace: list[np.ndarray],
    history: list[dict],
    bounds: tuple[np.ndarray, np.ndarray],
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []

    dimensions = trajectory[0].shape[1]
    if dimensions not in {2, 3}:
        raise ValueError("Visualisation is supported only for 2D or 3D trajectories.")

    contour = _contour_data(objective, bounds) if dimensions == 2 else None

    for frame_idx, positions in enumerate(trajectory):
        fig = plt.figure(figsize=(10, 4.5))
        if dimensions == 2:
            ax_swarm = fig.add_subplot(1, 2, 1)
            xx, yy, zz = contour
            # In 2D we can overlay the swarm on the objective landscape, which
            # makes the search dynamics much easier to interpret visually.
            ax_swarm.contourf(xx, yy, zz, levels=30, cmap="viridis", alpha=0.85)
            ax_swarm.scatter(positions[:, 0], positions[:, 1], color="#ee9b00", s=30, label="Particles")
            ax_swarm.scatter(
                best_trace[frame_idx][0],
                best_trace[frame_idx][1],
                color="#ae2012",
                marker="*",
                s=180,
                label="Global best",
            )
            ax_swarm.set_title("Swarm in objective space")
            ax_swarm.legend(loc="upper right")
            ax_swarm.set_xlim(bounds[0][0], bounds[1][0])
            ax_swarm.set_ylim(bounds[0][1], bounds[1][1])
        else:
            ax_swarm = fig.add_subplot(1, 2, 1, projection="3d")
            # In 3D we prioritise the swarm geometry itself; drawing a full
            # volumetric objective surface would add a lot of clutter.
            ax_swarm.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color="#ee9b00", s=28)
            ax_swarm.scatter(
                best_trace[frame_idx][0],
                best_trace[frame_idx][1],
                best_trace[frame_idx][2],
                color="#ae2012",
                marker="*",
                s=180,
            )
            ax_swarm.set_title("3D swarm trajectory")
            ax_swarm.set_xlim(bounds[0][0], bounds[1][0])
            ax_swarm.set_ylim(bounds[0][1], bounds[1][1])
            ax_swarm.set_zlim(bounds[0][2], bounds[1][2])

        ax_curve = fig.add_subplot(1, 2, 2)
        # The right panel mirrors the optimizer history so each frame shows
        # both spatial movement and optimization progress at the same time.
        iterations = [row["iteration"] for row in history[: frame_idx + 1]]
        best = [row["best_fitness"] for row in history[: frame_idx + 1]]
        ax_curve.plot(iterations, best, color="#005f73", linewidth=2)
        ax_curve.set_yscale("log")
        ax_curve.grid(True, alpha=0.3)
        ax_curve.set_title("Best fitness vs iteration")
        ax_curve.set_xlabel("Iteration")
        ax_curve.set_ylabel("Best fitness")

        fig.tight_layout()
        frame_path = output_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def save_gif_from_frames(frame_paths: list[Path], output_path: str | Path, fps: int = 8) -> None:
    import PIL.Image

    images = [PIL.Image.open(frame_path) for frame_path in frame_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def save_animation_mp4(frame_paths: list[Path], output_path: str | Path, fps: int = 8) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("off")
    image = plt.imread(frame_paths[0])
    artist = ax.imshow(image)

    def _update(frame_path: Path):
        artist.set_data(plt.imread(frame_path))
        return (artist,)

    anim = animation.FuncAnimation(fig, _update, frames=frame_paths, blit=True)
    anim.save(output_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
