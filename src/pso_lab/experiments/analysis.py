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
