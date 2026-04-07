"""Benchmark-suite execution over objectives, dimensions, and strategies."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from pso_lab.core import PSOConfig, deep_merge, load_yaml
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.io import save_json, save_rows_csv


def load_benchmark_suite(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def aggregate_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["variant"], row["objective"], row["dimensions"])
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, Any]] = []
    for (variant, objective, dimensions), bucket in grouped.items():
        aggregated.append(
            {
                "variant": variant,
                "objective": objective,
                "dimensions": dimensions,
                "runs": len(bucket),
                "mean_best_value": mean(item["best_value"] for item in bucket),
                "std_best_value": pstdev(item["best_value"] for item in bucket) if len(bucket) > 1 else 0.0,
                "mean_total_time": mean(item["total_time"] for item in bucket),
                "mean_auc": mean(item["auc"] for item in bucket),
                "mean_convergence_iteration": mean(item["convergence_iteration"] for item in bucket),
            }
        )
    aggregated.sort(key=lambda row: (row["objective"], row["dimensions"], row["variant"]))
    return aggregated


def run_benchmark_suite(
    suite: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    log_level: str = "INFO",
    max_cases: int | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir or suite.get("output_dir", "results/benchmarks"))
    output_root.mkdir(parents=True, exist_ok=True)
    defaults = suite.get("defaults", {})
    strategies = suite.get("strategies", {})
    objectives = suite.get("objectives", [])
    dimensions = suite.get("dimensions", [])
    seeds = suite.get("seeds", [])

    raw_rows: list[dict[str, Any]] = []
    case_counter = 0

    for variant, variant_overrides in strategies.items():
        for objective in objectives:
            objective_name = objective["name"] if isinstance(objective, dict) else objective
            objective_bounds = objective.get("bounds") if isinstance(objective, dict) else None
            for dimension in dimensions:
                for seed in seeds:
                    # Each benchmark case is built from shared defaults plus the
                    # current variant, objective, dimension, and seed.
                    merged = deep_merge(defaults, variant_overrides)
                    merged = deep_merge(
                        merged,
                        {
                            "objective": objective_name,
                            "dimensions": dimension,
                            "seed": seed,
                        },
                    )
                    if objective_bounds is not None:
                        merged["bounds"] = objective_bounds
                    config = PSOConfig.from_mapping(merged)
                    run_info = run_single_experiment(
                        config,
                        output_dir=suite.get("runs_output_dir", "results/runs"),
                        run_prefix=variant,
                        log_level=log_level,
                        save_results=True,
                    )
                    summary = run_info["summary"]
                    raw_rows.append(
                        {
                            "variant": variant,
                            "objective": objective_name,
                            "dimensions": dimension,
                            "seed": seed,
                            "best_value": summary["best_value"],
                            "total_time": summary["metrics"]["total_run_time"],
                            "auc": summary["metrics"]["auc_best_fitness"],
                            "convergence_iteration": summary["metrics"]["convergence_iteration"]
                            if summary["metrics"]["convergence_iteration"] is not None
                            else config.iterations,
                            "run_id": run_info["run_id"],
                            "run_dir": run_info["run_dir"],
                        }
                    )
                    case_counter += 1
                    if max_cases is not None and case_counter >= max_cases:
                        # Early-exit mode is useful for smoke tests and quick
                        # classroom demos without changing the benchmark YAML.
                        aggregated = aggregate_benchmark_rows(raw_rows)
                        save_rows_csv(output_root / "benchmark_runs.csv", raw_rows)
                        save_rows_csv(output_root / "benchmark_summary.csv", aggregated)
                        save_json(output_root / "benchmark_summary.json", aggregated)
                        return {"rows": raw_rows, "summary": aggregated, "output_dir": str(output_root)}

    aggregated = aggregate_benchmark_rows(raw_rows)
    save_rows_csv(output_root / "benchmark_runs.csv", raw_rows)
    save_rows_csv(output_root / "benchmark_summary.csv", aggregated)
    save_json(output_root / "benchmark_summary.json", aggregated)
    return {"rows": raw_rows, "summary": aggregated, "output_dir": str(output_root)}
