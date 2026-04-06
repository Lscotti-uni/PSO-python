"""Serialization and discovery utilities for saved experiment artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pso_lab.core.config import PSOConfig
from pso_lab.core.pso import PSOResult
from pso_lab.utils.system import get_git_commit, get_system_info


def compute_auc(history: list[dict[str, Any]]) -> float:
    # The AUC of the best-fitness curve summarizes not only the final quality
    # but also how quickly the run reached good solutions.
    xs = np.asarray([row["iteration"] for row in history], dtype=float)
    ys = np.asarray([row["best_fitness"] for row in history], dtype=float)
    if len(xs) < 2:
        return float(ys[0]) if len(ys) else 0.0
    return float(np.trapz(ys, xs))


def convergence_iteration(history: list[dict[str, Any]], tolerance: float) -> int | None:
    for row in history:
        if row["best_fitness"] <= tolerance:
            return int(row["iteration"])
    return None


def build_summary(
    *,
    run_id: str,
    config: PSOConfig,
    objective_name: str,
    evaluator_name: str,
    update_mode: str,
    boundary_strategy: str,
    topology_name: str,
    result: PSOResult,
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    # Summary JSON is the high-level artifact meant for dashboards, quick
    # inspection, and experiment traceability.
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "objective": objective_name,
        "strategy": evaluator_name,
        "update_mode": update_mode,
        "boundary_strategy": boundary_strategy,
        "topology": topology_name,
        "config": config.to_dict(),
        "best_value": float(result.best_value),
        "best_position": np.asarray(result.best_position, dtype=float).tolist(),
        "iterations_completed": int(result.state.iteration),
        "converged": bool(result.converged),
        "stop_reason": result.stop_reason,
        "metrics": {
            "final_best_fitness": float(result.best_value),
            "auc_best_fitness": compute_auc(result.history),
            "convergence_iteration": convergence_iteration(result.history, config.tolerance),
            "total_run_time": float(result.timings.get("total_run_time", 0.0)),
            "total_eval_time": float(result.timings.get("total_eval_time", 0.0)),
            "total_update_time": float(result.timings.get("total_update_time", 0.0)),
            "total_overhead_time": float(result.timings.get("total_overhead_time", 0.0)),
        },
        "git": {"commit": get_git_commit(root_dir)},
        "system": get_system_info(),
        "artifacts": {
            "history_csv": "history.csv",
            "summary_json": "summary.json",
            "trajectory_npz": "trajectory.npz" if result.trajectory else None,
        },
    }


def ensure_run_dir(base_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    path = Path(path)
    if not history:
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_trajectory(path: str | Path, result: PSOResult) -> None:
    if not result.trajectory:
        return
    # NPZ keeps large numeric traces compact while preserving fast NumPy reloads.
    np.savez_compressed(
        path,
        positions=np.stack(result.trajectory),
        best_positions=np.stack(result.best_position_trace),
    )


def save_run_artifacts(run_dir: str | Path, summary: dict[str, Any], result: PSOResult) -> None:
    run_dir = Path(run_dir)
    save_json(run_dir / "summary.json", summary)
    save_history_csv(run_dir / "history.csv", result.history)
    if result.trajectory:
        save_trajectory(run_dir / "trajectory.npz", result)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_history_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in {"True", "False"}:
                    parsed[key] = value == "True"
                    continue
                try:
                    parsed[key] = int(value)
                    continue
                except (ValueError, TypeError):
                    pass
                try:
                    parsed[key] = float(value)
                    continue
                except (ValueError, TypeError):
                    pass
                parsed[key] = value
            rows.append(parsed)
        return rows


def discover_summaries(root_dir: str | Path) -> list[Path]:
    return sorted(Path(root_dir).rglob("summary.json"))
