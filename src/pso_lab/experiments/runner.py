"""Single-run orchestration that wires objectives, topology, bounds, and evaluators."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pso_lab.core import PSO, PSOConfig, build_bounds_policy, build_topology
from pso_lab.io import build_summary, ensure_run_dir, save_run_artifacts
from pso_lab.objectives import get_objective
from pso_lab.parallel import build_evaluator
from pso_lab.utils import configure_logging, log_kv


def variant_label(config: PSOConfig) -> str:
    strategy = config.strategy.lower()
    update_mode = config.update_mode.lower()
    if strategy == "sequential" and update_mode == "loop":
        return "V0"
    if strategy == "thread" and update_mode == "loop":
        return "V1"
    if strategy == "process" and update_mode == "loop":
        return "V2"
    if strategy == "asyncio" and update_mode == "loop":
        return "V3"
    if strategy == "vectorized" and update_mode == "vectorized":
        return "V4"
    if strategy == "joblib" and update_mode == "loop":
        return "V5"
    return f"{strategy}-{update_mode}"


def build_run_id(config: PSOConfig, prefix: str | None = None) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    variant = variant_label(config)
    parts = [
        prefix or variant or "run",
        config.objective,
        f"d{config.dimensions}",
        f"s{config.seed}",
        timestamp,
    ]
    if prefix is not None and prefix != variant:
        parts.insert(1, variant)
    return "_".join(str(part) for part in parts if part is not None)


def run_single_experiment(
    config: PSOConfig,
    *,
    output_dir: str | Path = "results/runs",
    run_prefix: str | None = None,
    log_level: str = "INFO",
    save_results: bool = True,
) -> dict[str, Any]:
    objective_spec = get_objective(config.objective)
    if config.bounds == (-5.12, 5.12) and objective_spec.default_bounds != (-5.12, 5.12):
        config.bounds = objective_spec.default_bounds

    evaluator = build_evaluator(
        config.strategy,
        workers=config.workers,
        batch_size=config.batch_size,
        joblib_backend=config.joblib_backend,
        seed=config.seed,
        async_latency_ms=config.async_latency_ms,
        async_jitter_ms=config.async_jitter_ms,
    )
    bounds_policy = build_bounds_policy(config.boundary_strategy)
    topology = build_topology(config.topology, config.neighborhood_size)

    run_id = build_run_id(config, prefix=run_prefix)
    run_dir = ensure_run_dir(output_dir, run_id) if save_results else None
    logger = configure_logging(log_level, Path(run_dir) / "run.log" if run_dir else None)

    log_kv(
        logger,
        20,
        "run_start",
        run_id=run_id,
        objective=config.objective,
        strategy=config.strategy,
        update_mode=config.update_mode,
        dimensions=config.dimensions,
        swarm_size=config.swarm_size,
    )

    pso = PSO(
        config=config,
        objective=objective_spec.fn,
        evaluator=evaluator,
        bounds_policy=bounds_policy,
        topology=topology,
        logger=logger,
    )

    try:
        result = pso.run()
    finally:
        close_method = getattr(evaluator, "close", None)
        if callable(close_method):
            close_method()

    summary = build_summary(
        run_id=run_id,
        config=config,
        objective_name=objective_spec.key,
        evaluator_name=evaluator.name,
        update_mode=config.update_mode,
        boundary_strategy=bounds_policy.name,
        topology_name=topology.name,
        result=result,
        root_dir=Path.cwd(),
    )
    if run_dir is not None:
        save_run_artifacts(run_dir, summary, result)
    log_kv(
        logger,
        20,
        "run_end",
        run_id=run_id,
        best_value=f"{result.best_value:.6e}",
        stop_reason=result.stop_reason,
        total_time=f"{result.timings['total_run_time']:.4f}",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "summary": summary,
        "result": result,
    }
