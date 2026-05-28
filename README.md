# PSO Lab

Particle Swarm Optimization (PSO) project tailored for Linux/WSL, with one
shared core and six interchangeable execution variants:

- `V0`: sequential baseline
- `V1`: thread-based concurrency (`ThreadPoolExecutor`)
- `V2`: process-based parallelism (`ProcessPoolExecutor`)
- `V3`: cooperative concurrency (`asyncio` + `run_in_executor`)
- `V4`: NumPy vectorisation across the full swarm
- `V5`: `joblib.Parallel` with selectable backend (bonus)

Three neighbourhood topologies are available: `global`, `ring`, and
`von_neumann`. Three SciPy baselines (`differential_evolution`,
`dual_annealing`, `L-BFGS-B`) provide an external reference, and a Gradio
dashboard offers an interactive playground.

Repository URL: `https://github.com/Lscotti-uni/PSO-python`

## Repository Layout

| Path | Purpose |
| --- | --- |
| `docs/architecture.md` | Architecture overview and module dependency diagram |
| `configs/` | YAML configuration for single runs, benchmarks, and grid search |
| `docs/design.md` | Short design document with architecture and trade-offs |
| `notebooks/final_report.ipynb` | Narrated experimental report and analysis notebook |
| `results/` | Saved artifacts for runs, benchmarks, grid search, and visualizations |
| `scripts/` | Reproducible entry points required by the assignment |
| `src/pso_lab/` | Project source code |
| `tests/` | Unit tests for correctness and reproducibility |
| `requirements.txt` | Runtime and validation dependencies |
| `pyproject.toml` | Packaging metadata |

## Installation On Linux / WSL

```bash
cd /path/to/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The requirements file also includes the notebook runtime so
`notebooks/final_report.ipynb` can be executed without extra setup.

## Architecture

```mermaid
flowchart TD
    subgraph EntryPoints[CLI Entry Points]
        RP[run_pso.py]
        RB[run_benchmarks.py]
        RG[run_grid_search.py]
        MV[make_viz.py]
    end

    subgraph Experiments[experiments/]
        Runner[runner.py]
        Bench[benchmarks.py]
        Grid[grid_search.py]
    end

    subgraph Core[core/]
        PSO[PSO engine]
        Bounds[Bounds policy]
        Topology[Topology global/ring/von_neumann]
        Stop[Stopping rules]
        Config[PSOConfig]
    end

    subgraph Domain[objectives/ and parallel/]
        Objectives[Objective registry]
        UseCase[Inverted pendulum PID]
        Eval[Evaluators V0/V1/V2/V3/V4/V5]
        Topo[Topologies global/ring/von_neumann]
    end

    subgraph Extras[bonuses]
        Baselines[scipy baselines]
        Dashboard[Gradio dashboard]
    end

    subgraph Persistence[persistence and output]
        IO[io/results.py]
        Results[(results/)]
        Viz[viz/plots.py]
        Report[notebooks/final_report.ipynb]
    end

    RP --> Runner
    RB --> Bench
    RG --> Grid
    MV --> Viz

    Bench --> Runner
    Grid --> Runner

    Runner --> Config
    Runner --> PSO
    Runner --> Objectives
    Runner --> Eval
    Runner --> IO

    PSO --> Bounds
    PSO --> Topology
    PSO --> Stop

    IO --> Results
    Results --> Viz
    Results --> Report
```

Main design decisions:

- One shared PSO core to avoid duplicated logic.
- `clamp` and `reflect` as explicit boundary policies.
- Three pluggable topologies (`global`, `ring`, `von_neumann`).
- YAML-based configuration with CLI overrides.
- Objective bounds are defined in YAML instead of being hardcoded in scripts.
- Per-dimension bounds supported (used by the inverted-pendulum use case).
- Early stopping stays available for single runs, but benchmarks and grid
  search use fixed iteration budgets for fair comparisons.
- Persistence through `JSON`, `CSV`, and `NPZ`.
- Optional dependencies (`joblib`, `scipy`, `gradio`) are imported lazily so
  the core project keeps running without them.

For a dedicated architecture handoff document, see
`docs/architecture.md`.

## Implemented Functionality

- Continuous minimization in arbitrary dimension.
- Per-dimension box constraints through explicit boundary policies.
- Standard benchmarks: `Sphere`, `Rosenbrock`, `Rastrigin`, `Ackley`.
- Reproducible benchmark dimensions such as `2`, `10`, and `30`.
- Structured logging and per-iteration instrumentation.
- Configurable grid search over `w`, `c1`, `c2` across all six strategies
  (`V0`-`V5`); swarm size and iterations are pinned in `defaults` to keep
  the candidates comparable.
- 2D and 3D visualization with convergence plots and swarm animations.
- Structured persistence with seed, commit hash, and system metadata.

## Required Scripts

- `scripts/run_pso.py`: single run.
- `scripts/run_benchmarks.py`: benchmark suite (V0–V5).
- `scripts/run_grid_search.py`: 3×3×3 hyperparameter grid across strategies.
- `scripts/make_viz.py`: plots and animations from saved runs.
- `scripts/run_use_case_pendulum.py`: tune the inverted-pendulum PID controller.
- `scripts/run_scipy_baseline.py`: run SciPy baselines on any registered objective.
- `scripts/run_dashboard.py`: launch the Gradio dashboard (bonus).

## Parallel Strategies

| Variant | Backend | Intended lesson | Expected trade-off |
| --- | --- | --- | --- |
| `V0` | Sequential evaluation | Reference baseline | No parallel overhead, but no concurrency |
| `V1` | `ThreadPoolExecutor` | Show the impact of threads under the GIL | Cheap orchestration, but CPU-bound code may not speed up |
| `V2` | `ProcessPoolExecutor` | Show true parallel evaluation for heavier workloads | Better CPU scaling, but higher IPC and serialization cost |
| `V3` | `asyncio` + `run_in_executor` | Cooperative concurrency for I/O-bound or latency-laden objectives | Good when objectives wait on I/O, near-zero gain on pure CPU work |
| `V4` | NumPy vectorisation (whole-swarm call) | Exploit BLAS/SIMD for objectives that naturally support 2D inputs | Fastest single-process option, but limited by single-core memory bandwidth |
| `V5` | `joblib.Parallel` (loky/threading/mp) | Bonus: convenient parallelism with selectable backend | Trade-offs depend on backend; needs optional `joblib` dependency |

The core PSO implementation is shared across all variants. Only the
fitness-evaluation strategy changes, which keeps optimization quality
comparable while exposing timing differences.

V2 and V5 support batching through `batch_size`; V4 always processes the whole
swarm in a single NumPy call. V3 exposes `async_latency_ms` / `async_jitter_ms`
to model I/O-style objectives without writing an actual async backend.

## Logging And Observability

Each run records:

- total runtime
- fitness-evaluation time
- particle-update time
- estimated parallel overhead
- best fitness per iteration
- convergence-related metrics

Structured logs are written to `run.log`, while machine-readable artifacts are
stored in `summary.json`, `history.csv`, and optionally `trajectory.npz`.

## Reproducing Results

| Goal | Command |
| --- | --- |
| Run unit tests | `python3 -m pytest -q` |
| Single PSO run | `python3 scripts/run_pso.py --config configs/pso.yaml` |
| Benchmark suite (V0–V5) | `python3 scripts/run_benchmarks.py --config configs/benchmark.yaml` |
| Grid search across strategies | `python3 scripts/run_grid_search.py --config configs/grid_search.yaml` |
| Inverted-pendulum PID tuning | `python3 scripts/run_use_case_pendulum.py --config configs/use_case_pendulum.yaml` |
| SciPy baselines | `python3 scripts/run_scipy_baseline.py --objective sphere --dim 10` |
| Gradio dashboard (bonus) | `python3 scripts/run_dashboard.py` |
| Create visualization | `python3 scripts/make_viz.py --run-dir results/runs/<run_id> --gif` |
| Open the narrated report notebook | `jupyter notebook notebooks/final_report.ipynb` |

For fair comparisons, benchmarks and grid search run with fixed iteration
budgets. Single runs can still use early stopping.

## Usage Examples

Single run:

```bash
python scripts/run_pso.py --config configs/pso.yaml --objective sphere --dim 10 --seed 123
```

Benchmark suite:

```bash
python scripts/run_benchmarks.py --config configs/benchmark.yaml
```

Grid search:

```bash
python scripts/run_grid_search.py --config configs/grid_search.yaml --top-n 10
```

Visualization:

```bash
python scripts/run_pso.py --config configs/pso.yaml --objective sphere --dim 2 --track-trajectory
python scripts/make_viz.py --run-dir results/runs/<run_id> --gif
```

If the run dimension is not `2` or `3`, `make_viz.py` still generates
`convergence.png`, but it skips swarm animation frames.

## Persistence

Each individual run stores:

- `summary.json`: configuration, seed, commit, system, and final metrics.
- `history.csv`: per-iteration metrics.
- `trajectory.npz`: compressed trajectory, when enabled.
- `run.log`: structured logging output.

The `results/` directory is organised by stage:

- `results/runs/` — per-run summaries, histories, and optional trajectories
- `results/benchmarks/` — aggregated `benchmark_runs.csv` / `benchmark_summary.csv`
- `results/grid_search/` — aggregated `grid_search_runs.csv` / `grid_search_summary.csv`
- `results/visualizations/` — `convergence.png`, swarm frames, and GIFs
- `results/baselines/` — SciPy baseline reference values (`baseline_*.json`)
- `results/use_case_pendulum/` — runs for the inverted-pendulum PID tuning case

The full `results/` directory is generated automatically by the scripts and is
not versioned by default. This keeps the repository lightweight because full
benchmark and grid-search outputs can become very large.

## Analysis Notebook

Post-run analysis is centralized in:

- `notebooks/final_report.ipynb`

The notebook builds the Cartesian product of experimental cases, loads the
saved artifacts, and produces summary tables, convergence plots, boxplots, and
speedup comparisons.

## Tests

```bash
pytest
```

The current test suite covers:

- seed reproducibility
- boundary handling
- monotonic global-best behavior
- convergence on Sphere
- evaluator consistency across V0, V1, V2, V3, V4, and V5
- topology behaviour for `global`, `ring`, and `von_neumann`
- the inverted-pendulum use case (registry wiring, baseline stability, batched cost)
- SciPy baselines smoke tests (`scipy` is an optional dependency)
