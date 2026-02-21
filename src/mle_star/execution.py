"""Execution harness: environment setup, working directory management, and script operations.

Provides functions for setting up the working directory structure,
cleaning output directories, detecting GPU hardware, building
subprocess environment variables for script execution, writing
validated solution scripts to disk, running scripts as async
subprocesses with timeout enforcement and output capture, and
parsing subprocess output into structured evaluation results.

Refs:
    SRS 02a — Execution Environment (REQ-EX-001 through REQ-EX-004).
    SRS 02b — Script Operations (REQ-EX-005 through REQ-EX-014).
    SRS 02d — Constraints (REQ-EX-037, REQ-EX-043, REQ-EX-044).
    IMPLEMENTATION_PLAN.md Tasks 11, 12, 13, 14.
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

from mle_star.models import (
    EvaluationResult,
    MetricDirection,
    PipelineConfig,
    TaskDescription,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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


# ---------------------------------------------------------------------------
# Output Parsing (REQ-EX-011 through REQ-EX-014)
# ---------------------------------------------------------------------------

_TRACEBACK_MARKER = "Traceback (most recent call last):"

_TRACEBACK_PATTERN: re.Pattern[str] = re.compile(
    r"(?P<tb>Traceback \(most recent call last\):\n"
    r"(?:[ \t]+.+\n)*"
    r"\S[^\n]*)",
    re.MULTILINE,
)


def extract_traceback(stderr: str) -> str | None:
    """Extract the last Python traceback from stderr (REQ-EX-012).

    Matches the standard Python traceback pattern starting from
    ``Traceback (most recent call last):`` through the final exception
    line. When multiple tracebacks exist, returns the **last** one.

    Args:
        stderr: Full standard error output from a subprocess.

    Returns:
        The extracted traceback string, or ``None`` if no traceback is found.
    """
    matches = list(_TRACEBACK_PATTERN.finditer(stderr))
    if not matches:
        return None
    return matches[-1].group("tb")


def detect_error(raw: ExecutionRawResult) -> bool:
    """Determine whether a script execution produced an error (REQ-EX-013).

    Returns ``True`` if any of the following conditions hold:

    1. ``raw.exit_code != 0``
    2. ``raw.timed_out is True``
    3. ``raw.stderr`` contains the traceback marker string

    Args:
        raw: Raw execution result to inspect.

    Returns:
        ``True`` if the execution is considered an error.
    """
    if raw.exit_code != 0:
        return True
    if raw.timed_out:
        return True
    return _TRACEBACK_MARKER in raw.stderr


def build_evaluation_result(raw: ExecutionRawResult) -> EvaluationResult:
    """Construct an ``EvaluationResult`` from raw execution output (REQ-EX-014).

    Composes output parsers to build a structured evaluation result:

    1. Calls ``parse_score(raw.stdout)`` to obtain the score.
    2. Calls ``detect_error(raw)`` to set ``is_error``.
    3. Calls ``extract_traceback(raw.stderr)`` to obtain ``error_traceback``
       (only when ``is_error`` is ``True``).
    4. Maps ``stdout``, ``stderr``, ``exit_code``, ``duration_seconds``
       directly from the raw result.

    Args:
        raw: Raw execution result from ``execute_script``.

    Returns:
        A fully constructed ``EvaluationResult`` instance.
    """
    from mle_star.scoring import parse_score

    score = parse_score(raw.stdout)
    is_error = detect_error(raw)
    error_traceback = extract_traceback(raw.stderr) if is_error else None

    return EvaluationResult(
        score=score,
        stdout=raw.stdout,
        stderr=raw.stderr,
        exit_code=raw.exit_code,
        duration_seconds=raw.duration_seconds,
        is_error=is_error,
        error_traceback=error_traceback,
    )


# ---------------------------------------------------------------------------
# End-to-End Evaluation Pipeline (REQ-EX-015 through REQ-EX-023)
# ---------------------------------------------------------------------------


async def evaluate_solution(
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    timeout_override: int | None = None,
) -> EvaluationResult:
    """Evaluate a solution script end-to-end (REQ-EX-015).

    Orchestrates the full evaluation pipeline: set up working directory,
    clean output, write the script to disk, build the execution environment,
    execute the script as a subprocess, and parse the output into an
    ``EvaluationResult``.

    The input ``SolutionScript`` is **never** mutated (REQ-EX-016). The
    caller is responsible for updating ``solution.score`` after evaluation.

    Args:
        solution: The solution script to evaluate.
        task: Task description providing ``data_dir`` and context.
        config: Pipeline configuration with ``time_limit_seconds``.
        timeout_override: If provided, overrides ``config.time_limit_seconds``
            as the execution timeout in seconds.

    Returns:
        An ``EvaluationResult`` with parsed score, error status, and output.

    Raises:
        ValueError: If the solution content is invalid (empty or contains
            forbidden exit calls).
    """
    timeout = (
        timeout_override if timeout_override is not None else config.time_limit_seconds
    )

    working_dir = setup_working_directory(task.data_dir)
    clean_output_directory(working_dir)
    script_path = write_script(solution, working_dir)
    env = build_execution_env()
    raw = await execute_script(script_path, working_dir, timeout, env)
    return build_evaluation_result(raw)


async def evaluate_with_retry(
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    debug_callback: Callable[[SolutionScript, str | None], Awaitable[SolutionScript]],
    max_retries: int | None = None,
) -> tuple[SolutionScript, EvaluationResult]:
    """Evaluate a solution with retry-on-failure via debug callback (REQ-EX-021).

    Calls ``evaluate_solution`` and, if the result has ``is_error=True``,
    invokes *debug_callback* with the current solution and error traceback
    to obtain a fixed solution, then re-evaluates. Repeats up to
    *max_retries* times.

    Args:
        solution: The initial solution script to evaluate.
        task: Task description for evaluation context.
        config: Pipeline configuration (provides ``max_debug_attempts``
            as the default retry limit).
        debug_callback: Async callable ``(solution, traceback) -> fixed_solution``
            invoked on each failure to produce a corrected script.
        max_retries: Maximum number of debug retries. Defaults to
            ``config.max_debug_attempts`` when ``None``.

    Returns:
        A ``(SolutionScript, EvaluationResult)`` tuple. On success the
        result has ``is_error=False``. If all retries are exhausted, returns
        the last attempted pair with ``is_error=True``.
    """
    retries = max_retries if max_retries is not None else config.max_debug_attempts

    current_solution = solution
    result = await evaluate_solution(current_solution, task, config)

    for _ in range(retries):
        if not result.is_error:
            break
        current_solution = await debug_callback(
            current_solution, result.error_traceback
        )
        result = await evaluate_solution(current_solution, task, config)

    return current_solution, result


def is_better_solution(
    new_result: EvaluationResult,
    old_score: float,
    direction: MetricDirection,
) -> bool:
    """Check if a new evaluation result is strictly better than an old score (REQ-EX-023).

    Returns ``False`` immediately if the new result has no score or is an
    error. Otherwise delegates to ``is_improvement`` from the scoring module
    (REQ-EX-022).

    Args:
        new_result: The evaluation result to assess.
        old_score: The previous best score to compare against.
        direction: Whether to maximize or minimize the metric.

    Returns:
        ``True`` if *new_result* represents a strict improvement.
    """
    if new_result.score is None:
        return False
    if new_result.is_error:
        return False
    from mle_star.scoring import is_improvement

    return is_improvement(new_result.score, old_score, direction)


# ---------------------------------------------------------------------------
# Subsampling Utilities (REQ-EX-017 through REQ-EX-020)
# ---------------------------------------------------------------------------

SUBSAMPLE_INSTRUCTION: str = (
    "If there are more than {limit} training samples, "
    "you must subsample to {limit} for a faster run."
)
"""Parameterized subsampling instruction template (REQ-EX-017).

Contains two ``{limit}`` placeholders that are filled by
``get_subsample_instruction`` with the configured subsample limit.
"""


def get_subsample_instruction(config: PipelineConfig) -> str:
    """Return the subsampling instruction with the configured limit (REQ-EX-018).

    Renders ``SUBSAMPLE_INSTRUCTION`` by replacing ``{limit}`` with
    ``config.subsample_limit``.

    Args:
        config: Pipeline configuration providing ``subsample_limit``.

    Returns:
        Rendered instruction string containing the numeric limit.
    """
    return SUBSAMPLE_INSTRUCTION.format(limit=config.subsample_limit)


def request_subsample_removal(solution: SolutionScript) -> str:
    """Build a prompt instructing an agent to remove subsampling code (REQ-EX-019).

    The returned prompt includes the full solution script content and
    instructions to identify and remove all subsampling code while
    preserving all other functionality, returning the full modified script.

    Args:
        solution: The solution script containing subsampling code.

    Returns:
        A prompt string suitable for sending to an agent.
    """
    return (
        "Identify all subsampling code in the following solution script. "
        "Remove the subsampling code while preserving all other functionality. "
        "Return the full modified script.\n\n"
        f"```python\n{solution.content}\n```"
    )


def request_subsample_extraction(solution: SolutionScript) -> str:
    """Build a prompt instructing an agent to extract subsampling code (REQ-EX-020).

    The returned prompt includes the full solution script content and
    instructions to identify and extract the subsampling code block.

    Args:
        solution: The solution script to analyze for subsampling code.

    Returns:
        A prompt string suitable for sending to an agent.
    """
    return (
        "Identify and extract the subsampling code block from the "
        "following solution script.\n\n"
        f"```python\n{solution.content}\n```"
    )


# ---------------------------------------------------------------------------
# Submission Verification (REQ-EX-024, REQ-EX-025)
# ---------------------------------------------------------------------------


def verify_submission(
    working_dir: str,
    expected_filename: str = "submission.csv",
) -> bool:
    """Check that a valid submission file exists (REQ-EX-024).

    Returns ``True`` if ``{working_dir}/final/{expected_filename}`` exists
    and has a file size strictly greater than 0 bytes.

    Args:
        working_dir: Root directory containing the ``final/`` subdirectory.
        expected_filename: Name of the submission file to verify.

    Returns:
        ``True`` if the file exists and is non-empty, ``False`` otherwise.
    """
    submission = Path(working_dir) / "final" / expected_filename
    return submission.is_file() and submission.stat().st_size > 0


def get_submission_info(
    working_dir: str,
    expected_filename: str = "submission.csv",
) -> dict[str, bool | str | int | None]:
    """Return metadata about the submission file (REQ-EX-025).

    Args:
        working_dir: Root directory containing the ``final/`` subdirectory.
        expected_filename: Name of the submission file to inspect.

    Returns:
        A dict with keys:

        - ``exists`` (bool): Whether the submission file exists.
        - ``path`` (str): Absolute path to the submission file.
        - ``size_bytes`` (int): File size in bytes (0 if not exists).
        - ``row_count`` (int | None): Number of data rows (lines minus
          header), or ``None`` if the file does not exist.
    """
    submission = Path(working_dir) / "final" / expected_filename
    abs_path = str(submission.resolve())

    if not submission.is_file():
        return {
            "exists": False,
            "path": abs_path,
            "size_bytes": 0,
            "row_count": None,
        }

    size_bytes = submission.stat().st_size
    text = submission.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line]
    # row_count = data lines, excluding the header (first line)
    row_count = max(0, len(lines) - 1)

    return {
        "exists": True,
        "path": abs_path,
        "size_bytes": size_bytes,
        "row_count": row_count,
    }


# ---------------------------------------------------------------------------
# Batch Evaluation (REQ-EX-026)
# ---------------------------------------------------------------------------


async def evaluate_batch(
    solutions: list[SolutionScript],
    task: TaskDescription,
    config: PipelineConfig,
) -> list[EvaluationResult]:
    """Evaluate multiple solutions sequentially (REQ-EX-026).

    Calls ``evaluate_solution`` for each solution in order. Solutions are
    evaluated **sequentially** — not concurrently — to avoid resource
    contention. The returned list preserves the input order.

    Args:
        solutions: List of solution scripts to evaluate.
        task: Task description for evaluation context.
        config: Pipeline configuration.

    Returns:
        A list of ``EvaluationResult`` instances in the same order as
        the input *solutions* list.
    """
    results: list[EvaluationResult] = []
    for solution in solutions:
        result = await evaluate_solution(solution, task, config)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Solution Ranking (REQ-EX-027)
# ---------------------------------------------------------------------------


def _sort_key(
    pair: tuple[SolutionScript, EvaluationResult],
    direction: MetricDirection,
) -> tuple[int, float]:
    """Produce a sort key for ranking a (solution, result) pair.

    The key is a ``(tier, score_key)`` tuple where:

    - **tier 0**: valid score, no error
    - **tier 1**: ``None`` score, no error
    - **tier 2**: ``is_error=True``

    Within tier 0, *score_key* orders by score according to *direction*.
    Tiers 1 and 2 use ``0.0`` as a neutral sort key.

    Args:
        pair: A ``(SolutionScript, EvaluationResult)`` tuple.
        direction: Whether to maximize or minimize the metric.

    Returns:
        A ``(tier, score_key)`` tuple for use with ``sorted()``.
    """
    result = pair[1]
    if result.is_error:
        return (2, 0.0)
    if result.score is None:
        return (1, 0.0)
    # For maximize: negate so that higher scores sort first (ascending sort)
    # For minimize: use raw score so that lower scores sort first
    if direction == MetricDirection.MAXIMIZE:
        return (0, -result.score)
    return (0, result.score)


def rank_solutions(
    solutions: list[SolutionScript],
    results: list[EvaluationResult],
    direction: MetricDirection,
) -> list[tuple[SolutionScript, EvaluationResult]]:
    """Sort solutions by score, best first (REQ-EX-027).

    Returns a list of ``(solution, result)`` tuples sorted according to
    *direction*. The ordering has three tiers:

    1. Valid scores, sorted best-first per *direction*.
    2. ``None`` scores (no error).
    3. Error results (``is_error=True``).

    Args:
        solutions: List of solution scripts.
        results: Corresponding list of evaluation results (same length).
        direction: Whether to maximize or minimize the metric.

    Returns:
        A sorted list of ``(SolutionScript, EvaluationResult)`` tuples.
    """
    pairs = list(zip(solutions, results, strict=True))
    return sorted(pairs, key=lambda p: _sort_key(p, direction))
