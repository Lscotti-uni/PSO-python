"""CLI for executing one PSO run and saving its artifacts."""

import argparse
from pathlib import Path

from pso_lab.core import PSOConfig, deep_merge, load_yaml
from pso_lab.experiments import run_single_experiment
from pso_lab.utils import build_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single PSO optimisation experiment.")
    parser.add_argument("--config", type=str, default="configs/pso_default.yaml")
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--swarm-size", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--update-mode", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--joblib-backend", type=str, default=None)
    parser.add_argument("--track-trajectory", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="WARNING")
    args = parser.parse_args()

    base = load_yaml(args.config)
    overrides = {}
    if args.objective is not None:
        overrides["objective"] = args.objective
    if args.dim is not None:
        overrides["dimensions"] = args.dim
    if args.swarm_size is not None:
        overrides["swarm_size"] = args.swarm_size
    if args.iters is not None:
        overrides["iterations"] = args.iters
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.strategy is not None:
        overrides["strategy"] = args.strategy
    if args.update_mode is not None:
        overrides["update_mode"] = args.update_mode
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.joblib_backend is not None:
        overrides["joblib_backend"] = args.joblib_backend
    if args.track_trajectory:
        overrides["track_trajectory"] = True

    config = PSOConfig.from_mapping(deep_merge(base, overrides))
    output_dir = args.output_dir or base.get("output_dir", "results/runs")

    print("\nRunning PSO...\n")
    run_info = run_single_experiment(
        config,
        output_dir=output_dir,
        run_prefix="single",
        log_level=args.log_level,
        save_results=True,
    )
    summary = run_info["summary"]
    result = run_info["result"]

    main_table = build_table(
        [
            {
                "objective": summary["objective"],
                "strategy": summary["strategy"],
                "update_mode": summary["update_mode"],
                "dimensions": summary["config"]["dimensions"],
                "best_value": summary["best_value"],
                "iterations": summary["iterations_completed"],
                "stop_reason": summary["stop_reason"],
                "total_time": summary["metrics"]["total_run_time"],
            }
        ],
        ["objective", "strategy", "update_mode", "dimensions", "best_value", "iterations", "stop_reason", "total_time"],
    )
    timing_table = build_table(
        [
            {
                "eval_time": summary["metrics"]["total_eval_time"],
                "update_time": summary["metrics"]["total_update_time"],
                "overhead_time": summary["metrics"]["total_overhead_time"],
                "auc": summary["metrics"]["auc_best_fitness"],
                "convergence_iter": summary["metrics"]["convergence_iteration"]
                if summary["metrics"]["convergence_iteration"] is not None
                else "n/a",
            }
        ],
        ["eval_time", "update_time", "overhead_time", "auc", "convergence_iter"],
    )

    print("Result summary\n")
    print(main_table)
    print("\nTiming summary\n")
    print(timing_table)
    print(f"\nBest position: {result.best_position}")
    print(f"Artifacts saved under {Path(run_info['run_dir']).resolve()}")


if __name__ == "__main__":
    main()
