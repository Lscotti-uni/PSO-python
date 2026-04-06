"""Random-seed helpers for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)
