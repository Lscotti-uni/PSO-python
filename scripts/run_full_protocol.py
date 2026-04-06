"""CLI entry point for the full benchmark, search, analysis, and report pipeline."""

import argparse
from pathlib import Path

from pso_lab.core import deep_merge, load_yaml
from pso_lab.experiments import load_benchmark_suite, load_grid_search_config, run_benchmark_suite, run_grid_search
from pso_lab.experiments.analysis import create_boxplot, create_speedup_plot, plot_mean_convergence
from pso_lab.experiments.report import generate_protocol_report
from pso_lab.io import save_json, save_rows_csv
from pso_lab.utils import build_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full PSO experimental protocol.")
    parser.add_argument("--config", type=str, default="configs/protocol_full.yaml")
    parser.add_argument("--log-level", type=str, default="WARNING")
    args = parser.parse_args()

    protocol = load_yaml(args.config)
    output_root = Path(protocol["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_config = load_benchmark_suite(protocol["benchmark_config"])
    benchmark_output_dir = output_root / "benchmarks"
    benchmark_outcome = run_benchmark_suite(
        benchmark_config,
        output_dir=benchmark_output_dir,
        log_level=args.log_level,
    )
    benchmark_runs_root = Path(benchmark_outcome["output_dir"]) / "runs"

    base_grid = load_grid_search_config(protocol["grid_search_config"])
    consolidated_grid_rows = []
    consolidated_grid_best = []
    for variant, overrides in protocol["grid_search_strategies"].items():
        variant_config = deep_merge(base_grid, {"strategy": overrides, "output_dir": str(output_root / "grid_search" / variant)})
        outcome = run_grid_search(variant_config, output_dir=variant_config["output_dir"], log_level=args.log_level)
        for row in outcome["rows"]:
            row["variant"] = variant
            consolidated_grid_rows.append(row)
        if outcome["summary"]:
            best = dict(outcome["summary"][0])
            best["variant"] = variant
            consolidated_grid_best.append(best)

    save_rows_csv(output_root / "grid_search_overview.csv", consolidated_grid_rows)
    save_json(output_root / "grid_search_best.json", consolidated_grid_best)

    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plot_mean_convergence(benchmark_runs_root, analysis_dir / "benchmark_mean_convergence.png", title="Benchmark mean convergence")
    create_boxplot(
        [__import__("json").loads(path.read_text(encoding="utf-8")) for path in sorted(benchmark_runs_root.rglob("summary.json"))],
        analysis_dir / "benchmark_final_fitness_boxplot.png",
    )
    create_speedup_plot(
        [__import__("json").loads(path.read_text(encoding="utf-8")) for path in sorted(benchmark_runs_root.rglob("summary.json"))],
        analysis_dir / "benchmark_speedup.png",
    )

    report_path = generate_protocol_report(output_root, protocol.get("report_path", "docs/final_report.md"))

    print("\nBenchmark protocol summary\n")
    print(
        build_table(
            benchmark_outcome["summary"][:12],
            ["variant", "objective", "dimensions", "runs", "mean_best_value", "mean_total_time", "mean_auc"],
        )
    )
    print("\nBest grid-search configuration per variant\n")
    print(
        build_table(
            consolidated_grid_best,
            ["variant", "mean_best_value", "mean_total_time", "mean_auc", "inertia", "cognitive", "social"],
        )
    )
    print(f"\nArtifacts saved under {output_root.resolve()}")
    print(f"Detailed report written to {report_path.resolve()}")


if __name__ == "__main__":
    main()
