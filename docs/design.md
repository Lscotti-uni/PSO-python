# Design Notes

## Goal

Implement a maintainable Particle Swarm Optimization solution in Python and use
the same core as a testbed to compare:

- `V0` sequential execution
- `V1` `ThreadPoolExecutor` concurrency
- `V2` `ProcessPoolExecutor` parallelism

The delivery prioritizes architecture, reproducibility, observability,
persistence, grid search, and visualization on Linux/WSL.

## Architecture

The full dependency diagram and module overview are documented in
`docs/architecture.md`.

## Design Decisions

1. Keep one shared PSO core instead of multiple separate implementations.
2. Restrict topology to `global-best` to match the required minimum scope.
3. Keep boundary handling decoupled from the core through `clamp` and `reflect`.
4. Centralize configuration in YAML while preserving CLI overrides.
5. Use fixed iteration budgets in benchmarks and grid search so strategy
   comparisons are not distorted by early stopping.
6. Store structured artifacts so results can be analyzed later.
7. Keep results separated into four top-level folders: benchmarks, grid search, runs, and visualizations.

## Interfaces And Abstractions

The design uses explicit abstractions so the optimization logic stays modular:

- `FitnessEvaluator`: switches between sequential, thread-based, and
  process-based execution without changing the PSO core.
- `BoundsPolicy`: isolates how particles are projected back into the feasible
  box when they leave the search space.
- `TopologyStrategy`: isolates the social neighborhood rule. The current
  delivery keeps only `global-best`, but the interface remains available.

## Reproducibility And Observability

- Every execution accepts a `seed`.
- The current `git commit` is recorded when available.
- Basic system information is stored with each run.
- The implementation measures total time, fitness time, update time, and overhead time.
- The global best remains monotonic because it is selected from the personal-best archive.

## Benchmarks And Analysis

Benchmarks included:

- `Sphere`
- `Rosenbrock`
- `Rastrigin`
- `Ackley`

Recommended benchmark dimensions:

- `2`
- `10`
- `30`

Post-run analysis is centralized in
`notebooks/final_report.ipynb`, which:

- builds the Cartesian product of experimental cases
- loads `summary.json` and `history.csv`
- produces summary tables
- plots mean convergence, boxplots, and speedup
- compares grid-search results across `V0`, `V1`, and `V2`

## Limitations And Trade-Offs

- `V1` may not outperform `V0` because of the GIL for CPU-bound objectives.
- `V2` introduces serialization and IPC overhead.
- For small workloads, orchestration overhead can dominate any speedup.
- Full trajectory storage increases disk usage, so it remains optional.

## Main Commands

```bash
python scripts/run_pso.py --config configs/pso.yaml
python scripts/run_benchmarks.py --config configs/benchmark.yaml
python scripts/run_grid_search.py --config configs/grid_search.yaml
python scripts/make_viz.py --run-dir results/runs/<run_id> --gif
pytest
```
