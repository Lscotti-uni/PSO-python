"""CLI entry point for PSO hyperparameter grid search experiments."""

import argparse
from pathlib import Path

from pso_lab.experiments import load_grid_search_config, run_grid_search
from pso_lab.utils import build_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configurable PSO grid search.")
    parser.add_argument("--config", type=str, default="configs/grid_search.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="WARNING")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    config = load_grid_search_config(args.config)
    outcome = run_grid_search(
        config,
        output_dir=args.output_dir,
        log_level=args.log_level,
        max_configs=args.max_configs,
    )
    best_rows = outcome["summary"][: args.top_n]
    table = build_table(
        best_rows,
        [
            "config_index",
            "mean_best_value",
            "mean_total_time",
            "mean_auc",
            "mean_convergence_iteration",
            "inertia",
            "cognitive",
            "social",
            "swarm_size",
            "topology",
        ],
    )
    print("\nGrid search ranking\n")
    print(table)
    print(f"\nArtifacts saved under {Path(outcome['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
