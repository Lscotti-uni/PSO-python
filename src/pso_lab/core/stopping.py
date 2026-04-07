"""Stopping rules for tolerance, stagnation, and iteration limits."""

from __future__ import annotations

from pso_lab.core.config import PSOConfig


def reached_tolerance(best_value: float, tolerance: float) -> bool:
    return best_value <= tolerance


def should_stop(best_value: float, no_improve_iters: int, config: PSOConfig) -> str | None:
    if not config.enable_early_stopping:
        return None
    if reached_tolerance(best_value, config.tolerance):
        return "tolerance"
    if config.stagnation_iters and no_improve_iters >= config.stagnation_iters:
        return "stagnation"
    return None
