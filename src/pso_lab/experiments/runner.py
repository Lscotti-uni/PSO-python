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
    strategy = config.strategy.strip().lower()
    mapping = {
        "sequential": "V0", "seq": "V0", "v0": "V0",
        "thread": "V1", "threading": "V1", "threads": "V1", "v1": "V1",
        "process": "V2", "multiprocessing": "V2", "processes": "V2", "v2": "V2",
        "async": "V3", "asyncio": "V3", "v3": "V3",
        "vectorized": "V4", "vector": "V4", "numpy": "V4", "v4": "V4",
        "joblib": "V5", "v5": "V5",
    }
    return mapping.get(strategy, strategy)


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
    if objective_spec.fixed_dimensions is not None:
        # Use cases (e.g. pendulum PID) have a meaningful dimensionality and
        # cannot be evaluated at arbitrary dim — silently pin it.
        config.dimensions = objective_spec.fixed_dimensions

    evaluator = build_evaluator(
        config.strategy,
        workers=config.workers,
        batch_size=config.batch_size,
        async_latency_ms=config.async_latency_ms,
        async_jitter_ms=config.async_jitter_ms,
        joblib_backend=config.joblib_backend,
        seed=config.seed,
    )
    bounds_policy = build_bounds_policy(config.boundary_strategy)
    topology = build_topology(config.topology, neighborhood_size=config.neighborhood_size)

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
        # Use POSIX-style separators so that CSVs containing this path remain
        # portable between Windows producers and Linux consumers (and vice
        # versa); the notebook resolves these paths with pathlib.Path.
        "run_dir": run_dir.as_posix() if run_dir is not None else None,
        "summary": summary,
        "result": result,
    }
