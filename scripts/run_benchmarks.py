"""CLI entry point for running the configured benchmark suite."""

import argparse
from pathlib import Path

from pso_lab.experiments import load_benchmark_suite, run_benchmark_suite
from pso_lab.utils import build_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark suite across PSO strategies.")
    parser.add_argument("--config", type=str, default="configs/benchmark_suite.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="WARNING")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    suite = load_benchmark_suite(args.config)
    outcome = run_benchmark_suite(
        suite,
        output_dir=args.output_dir,
        log_level=args.log_level,
        max_cases=args.max_cases,
    )
    summary = outcome["summary"]
    table = build_table(
        summary,
        [
            "variant",
            "objective",
            "dimensions",
            "runs",
            "mean_best_value",
            "mean_total_time",
            "mean_auc",
            "mean_convergence_iteration",
        ],
    )
    print("\nBenchmark summary\n")
    print(table)
    print(f"\nArtifacts saved under {Path(outcome['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
