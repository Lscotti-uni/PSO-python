"""Analysis utilities for aggregating runs and plotting comparisons."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pso_lab.io import discover_summaries, load_history_csv, load_json


def aggregate_rows(rows: list[dict[str, Any]], by: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in by)
        grouped[key].append(row)

    aggregated: list[dict[str, Any]] = []
    for key, bucket in grouped.items():
        row = {field: value for field, value in zip(by, key)}
        row["runs"] = len(bucket)
        for metric in metrics:
            row[f"mean_{metric}"] = mean(item[metric] for item in bucket)
        aggregated.append(row)
    return aggregated


def load_summary_rows(results_root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in discover_summaries(results_root):
        summary = load_json(summary_path)
        rows.append(summary)
    return rows


def plot_mean_convergence(results_root: str | Path, output_path: str | Path, *, title: str = "Mean convergence") -> None:
    grouped: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for summary_path in discover_summaries(results_root):
        summary = load_json(summary_path)
        config = summary["config"]
        history = load_history_csv(Path(summary_path).with_name("history.csv"))
        key = (summary["strategy"], int(config["dimensions"]))
        grouped[key].append(np.asarray([row["best_fitness"] for row in history], dtype=float))

    fig, ax = plt.subplots(figsize=(8, 5))
    for (strategy, dimensions), curves in sorted(grouped.items()):
        max_len = max(len(curve) for curve in curves)
        padded = []
        for curve in curves:
            if len(curve) < max_len:
                curve = np.pad(curve, (0, max_len - len(curve)), mode="edge")
            padded.append(curve)
        mean_curve = np.mean(np.vstack(padded), axis=0)
        ax.plot(mean_curve, label=f"{strategy} | d={dimensions}")

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if grouped:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def create_boxplot(rows: list[dict[str, Any]], output_path: str | Path, *, metric: str = "best_value") -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(float(row[metric]))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(grouped.keys())
    data = [grouped[label] for label in labels]
    if data:
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(f"Distribution of {metric}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def create_speedup_plot(rows: list[dict[str, Any]], output_path: str | Path, *, baseline: str = "sequential") -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(float(row["metrics"]["total_run_time"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    if baseline in grouped and grouped[baseline]:
        baseline_time = mean(grouped[baseline])
        labels = []
        speedups = []
        for strategy, values in grouped.items():
            labels.append(strategy)
            speedups.append(baseline_time / mean(values))
        ax.bar(labels, speedups)
        ax.set_ylabel("Speedup")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No baseline data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Speedup vs sequential baseline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_speedup_table(
    rows: list[dict[str, Any]],
    *,
    baseline: str = "sequential",
    workers_by_strategy: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Return mean wall time, speedup, and parallel efficiency per strategy.

    Speedup = T_baseline / T_strategy. Efficiency = speedup / max(workers, 1).
    The function groups by (strategy, dimensions) so workloads of different
    sizes are not collapsed together.
    """

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        dimensions = int(row["config"]["dimensions"])
        grouped[(row["strategy"], dimensions)].append(float(row["metrics"]["total_run_time"]))

    workers_by_strategy = workers_by_strategy or {}
    baseline_times: dict[int, float] = {}
    for (strategy, dimensions), times in grouped.items():
        if strategy == baseline:
            baseline_times[dimensions] = float(np.mean(times))

    summary: list[dict[str, Any]] = []
    for (strategy, dimensions), times in sorted(grouped.items()):
        mean_time = float(np.mean(times))
        baseline_time = baseline_times.get(dimensions)
        speedup = baseline_time / mean_time if baseline_time and mean_time > 0 else float("nan")
        workers = max(1, workers_by_strategy.get(strategy, 1))
        efficiency = speedup / workers if not np.isnan(speedup) else float("nan")
        summary.append(
            {
                "strategy": strategy,
                "dimensions": dimensions,
                "runs": len(times),
                "mean_total_time": mean_time,
                "speedup_vs_baseline": speedup,
                "parallel_efficiency": efficiency,
                "workers": workers,
            }
        )
    return summary


def auc_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate AUC of the best-fitness curve by strategy and dimensions."""

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        dimensions = int(row["config"]["dimensions"])
        grouped[(row["strategy"], dimensions)].append(float(row["metrics"]["auc_best_fitness"]))

    return [
        {
            "strategy": strategy,
            "dimensions": dimensions,
            "runs": len(values),
            "mean_auc": float(np.mean(values)),
            "std_auc": float(np.std(values)),
        }
        for (strategy, dimensions), values in sorted(grouped.items())
    ]


def create_efficiency_plot(
    summary: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Parallel efficiency by strategy and dimensions",
) -> None:
    by_strategy: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in summary:
        if not np.isnan(row.get("parallel_efficiency", float("nan"))):
            by_strategy[row["strategy"]].append((row["dimensions"], row["parallel_efficiency"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    if by_strategy:
        for strategy, points in sorted(by_strategy.items()):
            points.sort()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=strategy)
        ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", label="ideal")
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("Parallel efficiency")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No efficiency data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
