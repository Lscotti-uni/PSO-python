"""Run PSO on the inverted-pendulum PID use case and save a response plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from pso_lab.core import PSOConfig, deep_merge, load_yaml
from pso_lab.experiments import run_single_experiment
from pso_lab.use_cases.inverted_pendulum import (
    PENDULUM_PID_REFERENCE_GAINS,
    pendulum_pid_cost,
    plot_response,
)
from pso_lab.utils import build_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune an inverted-pendulum PID controller with PSO.")
    parser.add_argument("--config", type=str, default="configs/use_case_pendulum.yaml")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--swarm-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="WARNING")
    args = parser.parse_args()

    base = load_yaml(args.config)
    overrides: dict[str, object] = {}
    if args.iters is not None:
        overrides["iterations"] = args.iters
    if args.swarm_size is not None:
        overrides["swarm_size"] = args.swarm_size
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.strategy is not None:
        overrides["strategy"] = args.strategy

    config = PSOConfig.from_mapping(deep_merge(base, overrides))
    output_dir = args.output_dir or base.get("output_dir", "results/use_case_pendulum")

    print("\nTuning inverted-pendulum PID controller with PSO...\n")
    run_info = run_single_experiment(
        config,
        output_dir=output_dir,
        run_prefix="pendulum",
        log_level=args.log_level,
        save_results=True,
    )
    result = run_info["result"]
    summary = run_info["summary"]
    run_dir = Path(run_info["run_dir"])

    baseline_cost = float(pendulum_pid_cost(PENDULUM_PID_REFERENCE_GAINS))
    improvement = baseline_cost - float(result.best_value)
    table = build_table(
        [
            {
                "strategy": summary["strategy"],
                "best_cost": result.best_value,
                "baseline_cost": baseline_cost,
                "improvement": improvement,
                "iterations": summary["iterations_completed"],
                "stop_reason": summary["stop_reason"],
                "total_time": summary["metrics"]["total_run_time"],
            }
        ],
        ["strategy", "best_cost", "baseline_cost", "improvement", "iterations", "stop_reason", "total_time"],
    )

    plot_response(
        result.best_position,
        output_path=run_dir / "best_response.png",
        title="Best PSO controller response",
    )
    plot_response(
        PENDULUM_PID_REFERENCE_GAINS,
        output_path=run_dir / "baseline_response.png",
        title="Baseline controller response",
    )

    print(table)
    print(f"\nBest gains: {result.best_position}")
    print(f"Artifacts saved under {run_dir.resolve()}")


if __name__ == "__main__":
    main()
