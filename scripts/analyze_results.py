"""CLI entry point for loading saved results and generating summary analysis."""

import argparse
from pathlib import Path

from pso_lab.experiments.analysis import create_boxplot, create_speedup_plot, load_summary_rows, plot_mean_convergence
from pso_lab.utils import build_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse previously saved PSO results.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/analysis")
    args = parser.parse_args()

    rows = load_summary_rows(args.results_root)
    if not rows:
        print("No summary.json files were found under the provided results root.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compact_rows = [
        {
            "run_id": row["run_id"],
            "objective": row["objective"],
            "strategy": row["strategy"],
            "dimensions": row["config"]["dimensions"],
            "best_value": row["best_value"],
            "total_time": row["metrics"]["total_run_time"],
        }
        for row in rows
    ]
    print("\nSaved runs\n")
    print(build_table(compact_rows[:15], ["run_id", "objective", "strategy", "dimensions", "best_value", "total_time"]))

    create_boxplot(rows, output_dir / "final_fitness_boxplot.png")
    create_speedup_plot(rows, output_dir / "speedup.png")
    plot_mean_convergence(args.results_root, output_dir / "mean_convergence.png")
    print(f"\nPlots saved under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
