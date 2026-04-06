"""System and Git metadata collection for experiment traceability."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def get_system_info() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }


def get_git_commit(root: str | Path | None = None) -> str:
    cwd = Path(root) if root is not None else None
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True)
            .strip()
        )
    except Exception:
        return "unknown"
