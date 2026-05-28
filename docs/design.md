# Design Notes

## Goal

Implement a maintainable Particle Swarm Optimization solution in Python and use
the same core as a testbed to compare:

- `V0` sequential execution
- `V1` `ThreadPoolExecutor` concurrency
- `V2` `ProcessPoolExecutor` parallelism
- `V3` `asyncio` cooperative concurrency (run-in-executor) with optional simulated latency
- `V4` NumPy vectorisation across the entire swarm
- `V5` `joblib.Parallel` with configurable backend (bonus)

Bonuses on top of the core scope:

- Ring and Von Neumann neighbourhood topologies.
- A real-world use case (inverted-pendulum PID tuning) that exercises
  multimodal, non-separable objectives with actuator saturation.
- SciPy baselines (`differential_evolution`, `dual_annealing`, `L-BFGS-B`) for
  external comparison.
- An interactive Gradio dashboard for ad-hoc exploration.

The delivery prioritizes architecture, reproducibility, observability,
persistence, grid search, and visualization on Linux/WSL.

## Architecture

The full dependency diagram and module overview are documented in
`docs/architecture.md`.

## Design Decisions

1. Keep one shared PSO core instead of multiple separate implementations.
2. Offer three interchangeable topologies (`global`, `ring`, `von_neumann`) so
   the impact of communication patterns can be analysed without forking the
   core algorithm.
3. Keep boundary handling decoupled from the core through `clamp` and `reflect`.
4. Centralize configuration in YAML while preserving CLI overrides.
5. Use fixed iteration budgets in benchmarks and grid search so strategy
   comparisons are not distorted by early stopping.
6. Store structured artifacts so results can be analyzed later.
7. Keep results separated by stage: benchmarks, grid search, runs, baselines,
   use-case results, and visualizations.
8. Make heavy optional dependencies (`joblib`, `scipy`, `gradio`) lazy imports
   so the project keeps working in environments where they are missing.

## Interfaces And Abstractions

The design uses explicit abstractions so the optimization logic stays modular:

- `FitnessEvaluator`: switches between sequential, threaded, process-based,
  asyncio, vectorised, and joblib execution without changing the PSO core.
- `BoundsPolicy`: isolates how particles are projected back into the feasible
  box when they leave the search space. Supports global and per-dimension
  bounds (used by the inverted-pendulum use case).
- `TopologyStrategy`: isolates the social neighborhood rule. Three concrete
  topologies are available: global-best, ring (configurable neighbourhood),
  and Von Neumann (toroidal five-cell stencil).
- `ObjectiveSpec`: registry entry that carries default bounds, known optimum,
  optional `fixed_dimensions`, and descriptive tags so use cases can plug into
  the same orchestration as the benchmark functions.

## Reproducibility And Observability

- Every execution accepts a `seed`.
- The current `git commit` is recorded when available.
- Basic system information is stored with each run.
- The implementation measures total time, fitness time, update time, and overhead time.
- The global best remains monotonic because it is selected from the personal-best archive.

## Persistence Format Choices

The project deliberately mixes three text formats, each chosen for the role
where it fits best:

- **YAML for inputs.** All configuration files in `configs/` (single runs,
  benchmark suite, grid search, use case) use YAML because humans edit them
  by hand. Comments and anchor-style nesting keep the intent visible without
  the punctuation noise of JSON.
- **JSON for per-run metadata.** Each saved run writes `summary.json` with
  the resolved `PSOConfig`, environment info (`platform`, `python_version`,
  `processor`, `cpu_count`), git commit, timing breakdown, and final fitness.
  JSON is a strict, schema-friendly format with nested structures that loads
  back into Python without losing type fidelity, which matters when the
  notebook resolves nested keys such as `summary['metrics']['total_time']`.
- **CSV for tabular outputs.** Per-iteration histories (`history.csv`) and
  aggregated tables (`benchmark_runs.csv`, `benchmark_summary.csv`,
  `grid_search_runs.csv`, `grid_search_summary.csv`) live in CSV because
  pandas reads them in a single line and they survive being opened in a
  spreadsheet for quick sanity checks. The two scopes (per-run history vs.
  cross-run aggregation) are kept separate so neither has to embed the
  other.
- **NPZ for optional trajectories.** When `track_trajectory` is enabled,
  full particle positions are stored in compressed `.npz` so disk usage
  stays bounded on long runs — the design doc flags this as opt-in for
  exactly that reason.

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
- plots mean convergence, boxplots, runtime, speedup, and overhead
- computes parallel efficiency from per-variant worker counts
- compares grid-search results across `V0`–`V5`
- loads the inverted-pendulum runs and compares the tuned cost against the
  untuned baseline gains
- juxtaposes PSO best fitness against the SciPy baselines on shared
  `(objective, dimension)` slices
- demonstrates the `V3` asynchronous evaluator on an I/O-style objective so
  the asymmetric case is actually measured rather than only described

Helper functions for downstream reuse (`compute_speedup_table`, `auc_table`,
`create_speedup_plot`, `plot_mean_convergence`, `create_boxplot`,
`create_efficiency_plot`) live in `pso_lab.experiments.analysis`.

## Limitations And Trade-Offs

- `V1` may not outperform `V0` because of the GIL for CPU-bound objectives.
- `V2` introduces serialization and IPC overhead.
- `V3` shines only when objectives have measurable latency (I/O, network);
  on pure-CPU work the overhead of the event loop dominates.
- `V4` is single-process: gains plateau when the swarm or dimension grows
  beyond the L2/L3 cache footprint.
- `V5` adds a dependency (`joblib`) and the loky backend pays a process-startup
  cost that the small benchmarks barely amortise.
- Ring and Von Neumann topologies trade convergence speed for diversity; on
  unimodal problems they tend to converge later than `global`.
- For small workloads, orchestration overhead can dominate any speedup.
- Full trajectory storage increases disk usage, so it remains optional.

## Main Commands

```bash
python scripts/run_pso.py --config configs/pso.yaml
python scripts/run_benchmarks.py --config configs/benchmark.yaml
python scripts/run_grid_search.py --config configs/grid_search.yaml
python scripts/run_use_case_pendulum.py --config configs/use_case_pendulum.yaml
python scripts/run_scipy_baseline.py --objective sphere --dim 10
python scripts/run_dashboard.py                # bonus
python scripts/make_viz.py --run-dir results/runs/<run_id> --gif
pytest
```
