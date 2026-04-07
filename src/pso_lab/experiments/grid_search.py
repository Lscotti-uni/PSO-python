"""Grid-search orchestration for PSO hyperparameter exploration."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from pso_lab.core import PSOConfig, deep_merge, load_yaml
from pso_lab.experiments.runner import run_single_experiment
from pso_lab.io import save_json, save_rows_csv


def load_grid_search_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def expand_search_space(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(search_space.keys())
    value_lists = [search_space[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*value_lists)]


def run_grid_search(
    config_data: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    log_level: str = "INFO",
    max_configs: int | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir or config_data.get("output_dir", "results/grid_search"))
    output_root.mkdir(parents=True, exist_ok=True)

    defaults = config_data.get("defaults", {})
    strategies = config_data.get("strategies")
    if strategies is None:
        strategies = {"grid": config_data.get("strategy", {})}
    objective = config_data["objective"]
    bounds = config_data.get("bounds")
    dimensions = config_data.get("dimensions", [10])
    seeds = config_data.get("seeds", [11, 23, 37])
    metric = config_data.get("metric", "mean_best_value")
    search_space = expand_search_space(config_data.get("search_space", {}))

    raw_rows: list[dict[str, Any]] = []
    aggregated_rows: list[dict[str, Any]] = []

    for variant, strategy_overrides in strategies.items():
        for config_idx, params in enumerate(search_space, start=1):
            if max_configs is not None and config_idx > max_configs:
                break

            bucket: list[dict[str, Any]] = []
            for dimension in dimensions:
                for seed in seeds:
                    # Each hyperparameter combination is evaluated across all
                    # requested dimensions and seeds for the current strategy.
                    merged = deep_merge(defaults, strategy_overrides)
                    merged = deep_merge(
                        merged,
                        params,
                    )
                    merged = deep_merge(
                        merged,
                        {
                            "objective": objective,
                            "dimensions": dimension,
                            "seed": seed,
                        },
                    )
                    if bounds is not None:
                        merged["bounds"] = bounds
                    config = PSOConfig.from_mapping(merged)
                    run_info = run_single_experiment(
                        config,
                        output_dir=config_data.get("runs_output_dir", "results/runs"),
                        run_prefix=variant,
                        log_level=log_level,
                        save_results=True,
                    )
                    summary = run_info["summary"]
                    row = {
                        "variant": variant,
                        "config_index": config_idx,
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
                        **params,
                    }
                    raw_rows.append(row)
                    bucket.append(row)

            aggregated_rows.append(
                {
                    "variant": variant,
                    "config_index": config_idx,
                    "runs": len(bucket),
                    "mean_best_value": mean(item["best_value"] for item in bucket),
                    "std_best_value": pstdev(item["best_value"] for item in bucket) if len(bucket) > 1 else 0.0,
                    "mean_total_time": mean(item["total_time"] for item in bucket),
                    "mean_auc": mean(item["auc"] for item in bucket),
                    "mean_convergence_iteration": mean(item["convergence_iteration"] for item in bucket),
                    **params,
                }
            )

    # The ranking metric is configurable, but every row still keeps the full
    # summary so downstream analysis can compare several criteria later.
    aggregated_rows.sort(key=lambda row: row.get(metric, row["mean_best_value"]))
    save_rows_csv(output_root / "grid_search_runs.csv", raw_rows)
    save_rows_csv(output_root / "grid_search_summary.csv", aggregated_rows)
    save_json(output_root / "grid_search_summary.json", aggregated_rows)
    return {
        "rows": raw_rows,
        "summary": aggregated_rows,
        "metric": metric,
        "output_dir": str(output_root),
    }
