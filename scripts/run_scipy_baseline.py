"""Run SciPy baseline optimisers on a registered objective for comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pso_lab.baselines import (
    run_differential_evolution,
    run_dual_annealing,
    run_lbfgsb,
)
from pso_lab.objectives import get_objective
from pso_lab.utils import build_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SciPy baselines on a registered objective.")
    parser.add_argument("--objective", type=str, default="sphere")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--methods", nargs="*", default=["differential_evolution", "dual_annealing", "lbfgsb"])
    parser.add_argument("--output-dir", type=str, default="results/baselines")
    args = parser.parse_args()

    spec = get_objective(args.objective)
    dimensions = spec.fixed_dimensions or args.dim

    runners = {
        "differential_evolution": run_differential_evolution,
        "dual_annealing": run_dual_annealing,
        "lbfgsb": run_lbfgsb,
    }
    rows: list[dict[str, Any]] = []
    for method in args.methods:
        if method not in runners:
            raise SystemExit(f"Unknown method '{method}'. Choose from {list(runners)}.")
        result = runners[method](spec.fn, spec.default_bounds, dimensions, seed=args.seed)
        rows.append(
            {
                "objective": spec.key,
                "dimensions": dimensions,
                "method": result.method,
                "best_value": result.best_value,
                "nfev": result.nfev,
                "wall_time": result.wall_time,
                "success": result.success,
            }
        )

    table = build_table(rows, ["objective", "dimensions", "method", "best_value", "nfev", "wall_time", "success"])
    print(table)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"baseline_{spec.key}_d{dimensions}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path.resolve()}")


if __name__ == "__main__":
    main()
