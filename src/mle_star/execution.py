"""Execution harness: environment setup and working directory management.

Provides functions for setting up the working directory structure,
cleaning output directories, detecting GPU hardware, and building
subprocess environment variables for script execution.

Refs:
    SRS 02a — Execution Environment (REQ-EX-001 through REQ-EX-004).
    IMPLEMENTATION_PLAN.md Task 11.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def setup_working_directory(base_path: str) -> str:
    """Create or verify the working directory structure (REQ-EX-001).

    Creates ``{base_path}/input/`` and ``{base_path}/final/`` if they do
    not already exist. Idempotent — safe to call multiple times.

    Args:
        base_path: Root directory for the competition workspace.

    Returns:
        The absolute path to *base_path*.
    """
    base = Path(base_path)
    (base / "input").mkdir(parents=True, exist_ok=True)
    (base / "final").mkdir(parents=True, exist_ok=True)
    return str(base.resolve())


def clean_output_directory(base_path: str) -> None:
    """Remove all files in ``{base_path}/final/`` without deleting the directory (REQ-EX-002).

    Args:
        base_path: Root directory containing the ``final/`` subdirectory.
    """
    final_dir = Path(base_path) / "final"
    for entry in final_dir.iterdir():
        if entry.is_file():
            entry.unlink()


def detect_gpu_info() -> dict[str, bool | int | list[str]]:
    """Detect available GPUs via ``nvidia-smi`` (REQ-EX-003).

    Returns a dictionary with GPU information. Never raises exceptions;
    returns safe defaults when detection fails.

    Returns:
        Dict with keys ``cuda_available`` (bool), ``gpu_count`` (int),
        and ``gpu_names`` (list[str]).
    """
    _defaults: dict[str, bool | int | list[str]] = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
    }
    try:
        result = subprocess.run(  # nosec B607
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return _defaults

    if result.returncode != 0:
        return _defaults

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        return _defaults

    return {
        "cuda_available": True,
        "gpu_count": len(names),
        "gpu_names": names,
    }


def build_execution_env(
    gpu_indices: list[int] | None = None,
) -> dict[str, str]:
    """Build environment variables for script execution (REQ-EX-004).

    Returns a copy of the current environment with ``PYTHONUNBUFFERED=1``
    and ``PYTHONHASHSEED=0`` set. If *gpu_indices* is provided,
    ``CUDA_VISIBLE_DEVICES`` is set to a comma-separated string of the
    indices; otherwise the variable is inherited from the parent process.

    Args:
        gpu_indices: GPU device indices to expose, or ``None`` to inherit.

    Returns:
        A new dict suitable for passing as ``env`` to ``subprocess.run``.
    """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "0"
    if gpu_indices is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
    return env
