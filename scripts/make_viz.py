"""CLI for generating convergence plots and swarm animations from one saved run."""

import argparse
from pathlib import Path

import numpy as np

from pso_lab.core import expand_bounds
from pso_lab.io import load_history_csv, load_json
from pso_lab.objectives import get_objective
from pso_lab.viz import plot_convergence, save_gif_from_frames, save_swarm_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Create convergence plots and swarm animations from a saved run.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = load_json(run_dir / "summary.json")
    history = load_history_csv(run_dir / "history.csv")
    plot_convergence(history, run_dir / "convergence.png")

    dimensions = int(summary["config"]["dimensions"])
    if dimensions not in {2, 3}:
        print(
            f"Saved convergence plot under {run_dir.resolve()}, "
            f"but swarm animation is only supported for 2D or 3D runs (received d={dimensions})."
        )
        return

    trajectory_file = run_dir / "trajectory.npz"
    if not trajectory_file.exists():
        print("No trajectory file found. Re-run with track_trajectory enabled.")
        return

    trajectory_payload = dict(np.load(trajectory_file))
    objective_spec = get_objective(summary["objective"])
    lower, upper = expand_bounds(summary["config"]["bounds"], summary["config"]["dimensions"])
    frame_paths = save_swarm_frames(
        objective=objective_spec.fn,
        trajectory=list(trajectory_payload["positions"]),
        best_trace=list(trajectory_payload["best_positions"]),
        history=history,
        bounds=(lower, upper),
        output_dir=run_dir / "frames",
    )
    if args.gif:
        save_gif_from_frames(frame_paths, run_dir / "swarm.gif", fps=args.fps)
    print(f"Visual assets saved under {run_dir.resolve()}")


if __name__ == "__main__":
    main()
