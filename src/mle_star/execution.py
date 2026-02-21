"""Execution harness: environment setup, working directory management, and script operations.

Provides functions for setting up the working directory structure,
cleaning output directories, detecting GPU hardware, building
subprocess environment variables for script execution, writing
validated solution scripts to disk, and running scripts as async
subprocesses with timeout enforcement and output capture.

Refs:
    SRS 02a — Execution Environment (REQ-EX-001 through REQ-EX-004).
    SRS 02b — Script Operations (REQ-EX-005 through REQ-EX-010).
    SRS 02d — Constraints (REQ-EX-037, REQ-EX-043, REQ-EX-044).
    IMPLEMENTATION_PLAN.md Tasks 11, 12, 13.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from mle_star.models import SolutionScript

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Forbidden exit-call patterns (REQ-EX-006, REQ-EX-044)
# ---------------------------------------------------------------------------

_FORBIDDEN_EXIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bexit\s*\("),
    re.compile(r"\bsys\.exit\s*\("),
    re.compile(r"\bos\._exit\s*\("),
    re.compile(r"\bquit\s*\("),
]


def _validate_script_content(content: str) -> None:
    """Validate script content before writing to disk (REQ-EX-006, REQ-EX-044).

    Checks that content is non-empty after stripping and does not contain
    forbidden exit calls (``exit()``, ``sys.exit()``, ``os._exit()``,
    ``quit()``).

    Args:
        content: Python source code to validate.

    Raises:
        ValueError: If content is empty or contains forbidden exit calls.
    """
    if not content.strip():
        msg = "Script content is empty after stripping whitespace"
        raise ValueError(msg)

    for pattern in _FORBIDDEN_EXIT_PATTERNS:
        match = pattern.search(content)
        if match:
            msg = f"Script contains forbidden call: {match.group()!r}"
            raise ValueError(msg)


def write_script(
    solution: SolutionScript,
    working_dir: str,
    filename: str = "solution.py",
) -> str:
    """Write a solution script to disk with pre-validation (REQ-EX-005, REQ-EX-006).

    Validates the script content, then writes it to
    ``{working_dir}/{filename}`` with UTF-8 encoding. Overwrites any
    existing file at the target path.

    Args:
        solution: The solution script to write.
        working_dir: Directory in which to create the script file.
        filename: Name of the script file (default ``"solution.py"``).

    Returns:
        The absolute path to the written file.

    Raises:
        ValueError: If content is empty or contains forbidden exit calls.
    """
    _validate_script_content(solution.content)

    target = Path(working_dir) / filename
    target.write_text(solution.content, encoding="utf-8")
    return str(target.resolve())


# ---------------------------------------------------------------------------
# Async Script Execution (REQ-EX-007 through REQ-EX-010, REQ-EX-037)
# ---------------------------------------------------------------------------

_SIGKILL_GRACE_SECONDS = 5


class ExecutionRawResult(BaseModel):
    """Raw output captured from a subprocess script execution (REQ-EX-008).

    Immutable container for the raw stdout, stderr, exit code, timing,
    and timeout status of a single script execution.

    Attributes:
        stdout: Full standard output from the subprocess.
        stderr: Full standard error from the subprocess.
        exit_code: Process exit code (0 = success, -1 = timeout).
        duration_seconds: Wall-clock execution time in seconds.
        timed_out: Whether execution was killed due to timeout.
    """

    model_config = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Send SIGTERM to the process group, escalating to SIGKILL after grace period.

    Uses ``os.killpg`` to terminate the entire process group, ensuring
    orphan child processes are also cleaned up (REQ-EX-037).

    Args:
        proc: The asyncio subprocess to kill.
    """
    pid = proc.pid
    if pid is None:
        return

    try:
        pgid = os.getpgid(pid)
    except (OSError, ProcessLookupError):
        return

    # SIGTERM the entire process group
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return

    # Wait for graceful shutdown, then escalate to SIGKILL
    try:
        await asyncio.wait_for(proc.wait(), timeout=_SIGKILL_GRACE_SECONDS)
    except TimeoutError:
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()


async def execute_script(
    script_path: str,
    working_dir: str,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> ExecutionRawResult:
    """Execute a Python script as an async subprocess (REQ-EX-007).

    Runs ``python {script_path}`` with ``cwd`` set to *working_dir*,
    captures stdout and stderr separately, records wall-clock duration,
    and enforces timeout via SIGTERM then SIGKILL (REQ-EX-009).

    Each invocation creates a new subprocess with its own process group
    for clean timeout termination (REQ-EX-010, REQ-EX-037).

    Args:
        script_path: Absolute path to the Python script to execute.
        working_dir: Working directory for the subprocess.
        timeout_seconds: Maximum seconds before the script is killed.
        env: Environment variables for the subprocess, or ``None`` to
            inherit the current environment.

    Returns:
        An ``ExecutionRawResult`` containing captured output, exit code,
        duration, and timeout status.
    """
    start = time.monotonic()

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        script_path,
        cwd=working_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        start_new_session=True,  # New process group for killpg (REQ-EX-037)
    )

    timed_out = False
    # Shield communicate() so it isn't cancelled on timeout — we still
    # need it to collect partial output after the process is killed.
    communicate_task = asyncio.ensure_future(proc.communicate())
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            asyncio.shield(communicate_task),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        timed_out = True
        await _kill_process_group(proc)
        # Process is now dead; communicate() will finish collecting
        # whatever was already buffered in the pipes.
        stdout_bytes, stderr_bytes = await communicate_task

    duration = time.monotonic() - start

    # Decode as UTF-8 with replacement for non-UTF-8 bytes (REQ-EX-043)
    stdout_str = stdout_bytes.decode("utf-8", errors="replace")
    stderr_str = stderr_bytes.decode("utf-8", errors="replace")

    if timed_out:
        exit_code = -1
    else:
        exit_code = proc.returncode if proc.returncode is not None else -1

    return ExecutionRawResult(
        stdout=stdout_str,
        stderr=stderr_str,
        exit_code=exit_code,
        duration_seconds=duration,
        timed_out=timed_out,
    )
