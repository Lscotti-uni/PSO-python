"""Report-generation helpers for protocol summaries and final deliverables."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import mean
from typing import Any


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = int(value)
                    continue
                except ValueError:
                    pass
                try:
                    parsed[key] = float(value)
                    continue
                except ValueError:
                    pass
                parsed[key] = value
            rows.append(parsed)
        return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No data available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if abs(value) < 1e-3 and value != 0:
                    values.append(f"{value:.3e}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _group_mean(rows: list[dict[str, Any]], key: str, metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(float(row[metric]))
    output = [{key: name, f"mean_{metric}": mean(values)} for name, values in grouped.items()]
    output.sort(key=lambda item: item[f"mean_{metric}"])
    return output


def _best_grid_rows(grid_root: Path) -> list[dict[str, Any]]:
    best_rows = []
    for variant_dir in sorted(path for path in grid_root.iterdir() if path.is_dir()):
        summary_path = variant_dir / "grid_search_summary.csv"
        if not summary_path.exists():
            continue
        rows = _load_csv(summary_path)
        if rows:
            row = dict(rows[0])
            row["variant"] = variant_dir.name
            best_rows.append(row)
    best_rows.sort(key=lambda row: row["variant"])
    return best_rows


def _speedup_rows(benchmark_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[int, list[float]] = {}
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in benchmark_runs:
        variant = str(row["variant"])
        dimensions = int(row["dimensions"])
        grouped.setdefault((variant, dimensions), []).append(float(row["total_time"]))
        if variant == "V0":
            baselines.setdefault(dimensions, []).append(float(row["total_time"]))

    output = []
    for (variant, dimensions), values in sorted(grouped.items()):
        baseline_time = mean(baselines[dimensions])
        output.append(
            {
                "variant": variant,
                "dimensions": dimensions,
                "mean_total_time": mean(values),
                "speedup_vs_V0": baseline_time / mean(values),
            }
        )
    return output


def _best_variant_per_case(benchmark_summary: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in benchmark_summary:
        key = (str(row["objective"]), int(row["dimensions"]))
        grouped.setdefault(key, []).append(row)

    winners = []
    for (objective, dimensions), rows in sorted(grouped.items()):
        best = min(rows, key=lambda item: float(item[metric]))
        winners.append(
            {
                "objective": objective,
                "dimensions": dimensions,
                "winner_variant": best["variant"],
                metric: best[metric],
            }
        )
    return winners


def generate_protocol_report(protocol_root: str | Path, output_path: str | Path) -> Path:
    protocol_root = Path(protocol_root)
    output_path = Path(output_path)
    benchmark_root = protocol_root / "benchmarks"
    grid_root = protocol_root / "grid_search"
    analysis_root = protocol_root / "analysis"

    benchmark_summary = _load_csv(benchmark_root / "benchmark_summary.csv")
    benchmark_runs = _load_csv(benchmark_root / "benchmark_runs.csv")
    best_grid = _best_grid_rows(grid_root)
    speedups = _speedup_rows(benchmark_runs)
    by_variant = _group_mean(benchmark_runs, "variant", "best_value")
    fastest_per_case = _best_variant_per_case(benchmark_summary, "mean_total_time")
    best_quality_per_case = _best_variant_per_case(benchmark_summary, "mean_best_value")

    overall_fastest = min(speedups, key=lambda row: float(row["mean_total_time"]))
    overall_slowest = max(speedups, key=lambda row: float(row["mean_total_time"]))
    best_quality_variant = min(by_variant, key=lambda row: float(row["mean_best_value"]))

    public_examples_root = Path("results/examples/analysis")
    if output_path.as_posix() == "docs/final_report.md" and public_examples_root.exists():
        plot_root = public_examples_root
    else:
        plot_root = analysis_root

    benchmark_plot = Path(os.path.relpath(plot_root / "benchmark_mean_convergence.png", output_path.parent)).as_posix()
    speedup_plot = Path(os.path.relpath(plot_root / "benchmark_speedup.png", output_path.parent)).as_posix()
    boxplot = Path(os.path.relpath(plot_root / "benchmark_final_fitness_boxplot.png", output_path.parent)).as_posix()

    report = f"""# Final Report: PSO Laboratory

## 1. Executive Summary

This project implements a complete Particle Swarm Optimization (PSO) laboratory
in Python with emphasis on maintainable architecture, observability, and
experimental comparison across sequential, concurrent, and parallel execution
strategies.

The algorithmic core is shared across all variants. Differences between
versions are limited to the fitness-evaluation strategy and the update mode,
which avoids maintaining multiple unrelated PSO implementations and keeps the
comparison fair.

The experimental protocol stored under `{protocol_root.as_posix()}` includes:

- benchmark runs on `Sphere`, `Rosenbrock`, `Rastrigin`, and `Ackley`
- dimensions `2`, `10`, and `30`
- five reproducible seeds per case
- variants `V0` to `V5`
- a reduced `3x3x3` grid search over `w`, `c1`, and `c2`
- structured persistence of metrics, logs, summaries, and trajectories

Key outcomes:

- The fastest variant in the study was `{overall_fastest["variant"]}` in
  dimension `{overall_fastest["dimensions"]}`, with mean time
  `{overall_fastest["mean_total_time"]:.4f}` s.
- The slowest variant was `{overall_slowest["variant"]}` in dimension
  `{overall_slowest["dimensions"]}`, which highlights the cost of overhead when
  the work per task is small.
- The best overall mean optimization quality was obtained by
  `{best_quality_variant["variant"]}` with mean fitness
  `{best_quality_variant["mean_best_value"]:.4f}`.
- In the numerical benchmarks used here, the most effective recommendation is
  to favor variants with lower overhead, especially `{overall_fastest["variant"]}`.

## 2. Project Goal

The goal was to design and implement a complete and maintainable PSO solution in
Python, following software-engineering practices, and to use it as a laboratory
for comparing concurrency and parallelism strategies.

Beyond the optimizer itself, the project required:

- standard benchmarks
- instrumentation and logging
- hyperparameter grid search
- disk persistence
- visualization
- reproducible scripts
- architecture-oriented documentation

## 3. Project Architecture

The final structure is organized as follows:

- `core/`: PSO engine, state, topology, bounds, and configuration
- `objectives/`: benchmark functions and central registry
- `parallel/`: `sequential`, `thread`, `process`, `asyncio`, `vectorized`, and
  `joblib` evaluators
- `experiments/`: runners, benchmarks, grid search, analysis, and report
  generation
- `io/`: `JSON`, `CSV`, and `NPZ` persistence
- `viz/`: convergence plots and swarm animations
- `dashboard/`: local Gradio dashboard for stored results
- `scripts/`: reproducible high-level command-line entry points

### 3.1 Dependency Diagram

```mermaid
flowchart LR
    S[scripts/*] --> R[experiments.runner]
    R --> C[core.PSO]
    R --> O[objectives.registry]
    R --> P[parallel/* evaluators]
    C --> B[core.bounds]
    C --> T[core.topology]
    C --> STOP[core.stopping]
    R --> IO[io.results]
    IO --> RES[(results/*)]
    RES --> V[viz.plots]
    RES --> D[dashboard.app]
```

### 3.2 Design Decisions

1. A single PSO core with interchangeable strategies.
2. `clamp` as the default boundary policy, with `reflect` as an alternative.
3. `global-best` as the minimum topology and `ring` as an additional local
   topology.
4. `V3` uses simulated latency so that `asyncio` is methodologically justified.
5. `V4` is treated as the reference for implicit parallelism through NumPy.
6. `V5` uses `joblib` to compare a higher-level backend on the same
   experimental API.

### 3.3 Dataclass Usage

`dataclass` is used in:

- `PSOConfig`
- `SwarmState`
- `IterationMetrics`
- `PSOResult`
- `ObjectiveSpec`
- `EvaluationStats`

This improves inspection, typing, persistence, and state traceability.

## 4. Experimental Methodology

### 4.1 Benchmarks

- Objectives: `Sphere`, `Rosenbrock`, `Rastrigin`, `Ackley`
- Dimensions: `2`, `10`, `30`
- Seeds: `11`, `23`, `37`, `47`, `59`
- Maximum iterations: `120`
- Swarm size: `40`
- Default topology: `global`
- Boundary policy: `clamp`

### 4.2 Grid Search

A reduced `3x3x3` grid was evaluated over:

- `w in {{0.50, 0.7298, 0.90}}`
- `c1 in {{1.20, 1.49618, 1.80}}`
- `c2 in {{1.20, 1.49618, 1.80}}`

for five seeds and for each variant `V0` to `V5`, using `Sphere` in dimension
`10` as the tuning case.

### 4.3 Reproducible Commands

```bash
python scripts/run_pso.py --objective sphere --dim 10 --iters 200 --seed 123
python scripts/run_benchmarks.py --config configs/benchmark_suite_full.yaml
python scripts/run_grid_search.py --config configs/grid_search_protocol.yaml
python scripts/run_full_protocol.py --config configs/protocol_full.yaml
python scripts/run_dashboard.py --results-root results/protocol_full
```

### 4.4 Instrumentation

Each iteration records:

- `best_fitness`
- `mean_fitness`
- `min_fitness_current`
- `eval_time`
- `update_time`
- `worker_time_total`
- `critical_path_time`
- `overhead_time`
- `tasks_submitted`

This allows the implementation to separate useful computation from orchestration
overhead.

## 5. Benchmark Results

![Mean convergence curve]({benchmark_plot})

### 5.1 Aggregate Performance Per Case

{_markdown_table(benchmark_summary[:24], ["variant", "objective", "dimensions", "runs", "mean_best_value", "mean_total_time", "mean_auc", "mean_convergence_iteration"])}

### 5.2 Mean Time And Speedup Against V0

![Speedup by strategy]({speedup_plot})

{_markdown_table(speedups, ["variant", "dimensions", "mean_total_time", "speedup_vs_V0"])}

### 5.3 Mean Final Quality By Variant

![Final fitness distribution]({boxplot})

{_markdown_table(by_variant, ["variant", "mean_best_value"])}

### 5.4 Winners Per Case

Best variant by time:

{_markdown_table(fastest_per_case, ["objective", "dimensions", "winner_variant", "mean_total_time"])}

Best variant by final quality:

{_markdown_table(best_quality_per_case, ["objective", "dimensions", "winner_variant", "mean_best_value"])}

### 5.5 Interpretation

The results show a consistent pattern:

1. `V4` clearly dominates mean runtime in most cases.
2. `V1` does not beat the baseline, which matches expectations under the GIL.
3. `V2` is correct but expensive for these benchmarks because IPC overhead is
   too large.
4. `V3` is methodologically sound, but it is not competitive for CPU-bound
   workloads.
5. `V5` provides a useful comparison against `V2`: similar process-oriented
   logic, but through a higher-level abstraction with its own overhead profile.
6. Optimization quality remains very similar across variants that share the
   same core, confirming that the evaluation strategy does not change the
   essential algorithmic behavior.

## 6. Grid Search Results

The best configuration found for each variant was:

{_markdown_table(best_grid, ["variant", "mean_best_value", "mean_total_time", "mean_auc", "mean_convergence_iteration", "inertia", "cognitive", "social"])}

### 6.1 Tuning Interpretation

The grid search suggests several useful points:

- `w = 0.50` appears consistently in the strongest configurations.
- A high cognitive term (`c1 = 1.80`) helps strong convergence on `Sphere`.
- `V4` favors a slightly different social term, suggesting that the effective
  dynamics shift when the implementation operates on full arrays.

## 7. Critical Discussion

### 7.1 V0 Versus V4

The clearest comparison in the project is `V0` versus `V4`. The vectorized
version shifts work from Python loops to NumPy array operations, reduces
interpreted overhead, and achieves strong speedups across dimensions.

### 7.2 Threads And The GIL

`V1` uses `ThreadPoolExecutor`. For simple numerical evaluation, improvement is
limited because the GIL does not disappear. The variant is still useful as a
didactic contrast: concurrency does not automatically imply speedup.

### 7.3 Processes And IPC

`V2` avoids the GIL by using processes, but it pays for serialization, data
copying, and coordination. In this protocol, that cost dominates, so the
variant remains behind the baseline.

### 7.4 Asyncio With A Valid Use Case

`V3` is not presented as a universal accelerator. It is used for latency-aware
evaluation scenarios, which keeps the methodological interpretation honest and
avoids overstating the value of `asyncio` for CPU-bound numerical work.

### 7.5 Which Strategy Fits Which Case

- If the objective is vectorized with NumPy: favor `V4`
- If evaluation is expensive and not vectorizable: investigate `V2`
- If a higher-level local parallel backend is desirable: investigate `V5`
- If latency or simulated I/O matters: use `V3`
- If a simple reference implementation is needed: use `V0`
- If GIL behavior must be illustrated explicitly: use `V1`

## 8. Persistence, Observability, And Reproducibility

Each run stores:

- `summary.json`: configuration, commit, hardware, strategy, and final metrics
- `history.csv`: per-iteration metrics
- `run.log`: structured logging with timing and events
- `trajectory.npz`: when swarm tracking is enabled

The repository may keep a small subset of representative artifacts under
`results/examples/`, while the rest remains as reproducible local output.

The project also includes:

- a local Gradio dashboard
- aggregate analysis plots
- 2D and 3D swarm visualization
- automated tests for reproducibility, bounds, convergence, and monotonic
  global-best behavior

### 8.1 Dashboard

The dashboard can:

- list stored runs
- inspect `summary.json`
- display the convergence curve
- open graphical artifacts
- show aggregate analysis plots

Command:

```bash
python scripts/run_dashboard.py --results-root results/protocol_full
```

## 9. Limitations And Validity Threats

- The dashboard depends on previously generated results; it does not replace
  the experimental analysis itself.
- Process-based variants can be expensive on Windows or on machines with few
  cores.
- The 3D visualization shows the swarm cloud rather than a full objective
  surface.
- Results depend on the execution environment, which is why hardware and commit
  metadata are recorded.
- The grid search is centered on `Sphere`, so its conclusions may not transfer
  equally well to more rugged functions.

## 10. Final Recommendations

- For vectorizable numerical objectives, prioritize `V4`.
- For expensive, non-vectorizable evaluation, study `V2` with batching.
- Use `V5` when comparing `joblib` ergonomics and behavior against manually
  managed process pools.
- Use `V1` as a didactic comparison for GIL effects.
- Reserve `V3` for objectives with real or simulated latency or I/O.
- Keep `V0` as the conceptual baseline in all comparisons.

## 11. Generated Artifacts

- Benchmarks: `{benchmark_root.as_posix()}`
- Grid search: `{grid_root.as_posix()}`
- Analysis plots: `{analysis_root.as_posix()}`
- Local dashboard: `python scripts/run_dashboard.py --results-root {protocol_root.as_posix()}`

## 12. Conclusion

The project goal has been fully covered:

- a maintainable PSO implementation was designed
- interchangeable variants were implemented
- the system was instrumented
- results were persisted
- visual outputs were generated
- a reproducible experimental protocol was executed
- a local dashboard was added for exploration
- an optional `V5` bonus based on `joblib` was incorporated

The main conclusion is straightforward: for this project and this benchmark
profile, the best balance between simplicity, speed, and quality comes from the
lowest-overhead variants, with `V4` as the strongest reference for vectorizable
objectives and `V0` as the clean conceptual baseline. `V5` adds a valuable
comparison through a higher-level parallel framework.
"""
    output_path = Path(output_path)
    output_path.write_text(report, encoding="utf-8")
    return output_path
