"""Pipeline orchestrator: entry point, CLI client setup, and phase dispatch.

Provides ``run_pipeline()`` (async) and ``run_pipeline_sync()`` (sync wrapper)
as the top-level entry points for the MLE-STAR pipeline. Handles input
validation, Claude Code CLI client lifecycle, system prompt construction,
sequential phase dispatch, time budgeting, and graceful shutdown.

Refs:
    SRS 09a -- Orchestrator Entry Point & SDK Client Setup.
    SRS 09b -- Orchestrator Phase Dispatch & Sequencing.
    SRS 09c -- Orchestrator Budgets & Hooks.
    IMPLEMENTATION_PLAN.md Tasks 42, 43, 44, 45, 46, 47.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess as _subprocess_mod
import time
from typing import Any

from pydantic import BaseModel

from mle_star.execution import detect_gpu_info, setup_working_directory
from mle_star.finalization import run_finalization
from mle_star.models import (
    AgentConfig,
    AgentType,
    FinalResult,
    Phase1Result,
    Phase2Result,
    Phase3Result,
    PhaseTimeBudget,
    PipelineConfig,
    SolutionScript,
    TaskDescription,
    build_default_agent_configs,
)
from mle_star.phase1 import run_phase1
from mle_star.phase2_outer import run_phase2_outer_loop
from mle_star.phase3 import run_phase3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Claude CLI version check (REQ-OR-054)
# ---------------------------------------------------------------------------

_MIN_CLAUDE_CLI_VERSION = "1.0.0"


def check_claude_cli_version() -> None:
    """Verify that ``claude`` CLI meets the minimum version (REQ-OR-054).

    Checks that the ``claude`` executable is on ``PATH`` and that its
    version is at least ``_MIN_CLAUDE_CLI_VERSION``.

    Raises:
        FileNotFoundError: If ``claude`` is not found on PATH.
        ImportError: If the CLI version is below the minimum.
    """
    if shutil.which("claude") is None:
        msg = "Claude Code CLI ('claude') not found on PATH"
        raise FileNotFoundError(msg)

    try:
        result = _subprocess_mod.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_str = result.stdout.strip()
        parts = version_str.split()
        installed = parts[-1] if parts else "0.0.0"
    except (_subprocess_mod.TimeoutExpired, FileNotFoundError) as exc:
        msg = f"Cannot determine claude CLI version: {exc}"
        raise FileNotFoundError(msg) from exc

    if _parse_version_tuple(installed) < _parse_version_tuple(_MIN_CLAUDE_CLI_VERSION):
        msg = f"claude CLI >= {_MIN_CLAUDE_CLI_VERSION} required, but found {installed}"
        raise ImportError(msg)


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Parse a dotted version string into a tuple of ints for comparison.

    Args:
        version: A version string like ``"1.0.0"``.

    Returns:
        Tuple of integer components, e.g. ``(1, 0, 0)``.
    """
    return tuple(int(part) for part in version.split("."))


# ---------------------------------------------------------------------------
# ClaudeCodeClient — subprocess-based CLI wrapper
# ---------------------------------------------------------------------------


class ClaudeCodeClient:
    """Subprocess-based client for Claude Code headless mode.

    Wraps ``claude -p`` invocations as async subprocess calls. Each
    ``send_message()`` call spawns a new subprocess.

    Attributes:
        system_prompt: Shared system prompt for all agents.
        agent_configs: Mapping of AgentType to AgentConfig.
        model: Default Claude model identifier.
        permission_mode: CLI permission mode flag.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        agent_configs: dict[AgentType, AgentConfig],
        model: str = "sonnet",
        permission_mode: str = "dangerously-skip-permissions",
    ) -> None:
        """Initialize the Claude Code client.

        Args:
            system_prompt: Shared system prompt for all agents.
            agent_configs: Mapping of AgentType to AgentConfig.
            model: Default Claude model identifier.
            permission_mode: CLI permission mode flag.
        """
        self._system_prompt = system_prompt
        self._agent_configs = agent_configs
        self._model = model
        self._permission_mode = permission_mode

    async def send_message(
        self,
        agent_type: AgentType,
        message: str,
        *,
        output_schema: type[BaseModel] | None = None,
        use_structured_output: bool = True,
        session_id: str | None = None,
    ) -> str:
        """Send a message to an agent via ``claude -p``.

        Args:
            agent_type: The agent to invoke.
            message: The prompt message.
            output_schema: Per-call override for structured JSON output
                schema. When ``None``, falls back to
                ``AgentConfig.output_schema``.
            use_structured_output: When ``False``, skip structured output
                even if the agent config has an ``output_schema``.
            session_id: Optional session ID for ``--resume`` continuation.

        Returns:
            The agent's text response.

        Raises:
            RuntimeError: If the subprocess exits with non-zero status.
        """
        config = self._agent_configs[agent_type]
        cmd = self._build_command(
            config, message, output_schema, use_structured_output, session_id
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_text = stderr.decode("utf-8", errors="replace")
            msg = f"claude -p failed (exit {proc.returncode}): {error_text[:500]}"
            raise RuntimeError(msg)

        return stdout.decode("utf-8", errors="replace")

    def _build_command(
        self,
        config: AgentConfig,
        message: str,
        output_schema: type[BaseModel] | None,
        use_structured_output: bool,
        session_id: str | None,
    ) -> list[str]:
        """Build the ``claude`` CLI command list.

        Args:
            config: Agent configuration.
            message: The prompt message.
            output_schema: Per-call structured output schema override.
            use_structured_output: Whether to apply structured output.
            session_id: Optional session ID for resume.

        Returns:
            List of command-line arguments.
        """
        cmd = ["claude", "-p", message, "--verbose"]
        self._add_agent_flags(cmd, config)
        self._add_global_flags(
            cmd, output_schema, use_structured_output, config, session_id
        )
        return cmd

    def _add_agent_flags(self, cmd: list[str], config: AgentConfig) -> None:
        """Append agent-specific flags (system prompt, model, tools, max turns)."""
        agent_system = self._system_prompt
        if config.system_prompt:
            agent_system = f"{self._system_prompt}\n\n{config.system_prompt}"
        cmd.extend(["--system-prompt", agent_system])

        model = config.model or self._model
        cmd.extend(["--model", model])

        if config.tools:
            cmd.extend(["--allowedTools", ",".join(config.tools)])
        if config.max_turns is not None:
            cmd.extend(["--max-turns", str(config.max_turns)])

    def _add_global_flags(
        self,
        cmd: list[str],
        output_schema: type[BaseModel] | None,
        use_structured_output: bool,
        config: AgentConfig,
        session_id: str | None,
    ) -> None:
        """Append global flags (permission, budget, structured output, session)."""
        if self._permission_mode:
            cmd.append(f"--{self._permission_mode}")

        effective_schema = output_schema
        if effective_schema is None and use_structured_output:
            effective_schema = config.output_schema
        if effective_schema is not None:
            schema = effective_schema.model_json_schema()
            cmd.extend(["--output-format", "json", "--json-schema", json.dumps(schema)])

        if session_id:
            cmd.extend(["--resume", session_id])


# ---------------------------------------------------------------------------
# Custom exceptions (REQ-OR-042, REQ-OR-030)
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Pipeline failure with diagnostic context (REQ-OR-042).

    Raised when the pipeline encounters an unrecoverable error. The
    ``diagnostics`` dict carries structured context (elapsed time,
    last successful operation) for debugging.

    Attributes:
        diagnostics: Structured diagnostic information about the failure.
    """

    def __init__(self, message: str, *, diagnostics: dict[str, Any]) -> None:
        """Initialize with a message and structured diagnostics.

        Args:
            message: Human-readable error description.
            diagnostics: Structured context (elapsed time, etc.).
        """
        super().__init__(message)
        self.diagnostics = diagnostics


class PipelineTimeoutError(PipelineError):
    """Pipeline timed out before Phase 1 completed (REQ-OR-030).

    Raised when the overall time budget expires before a usable solution
    has been produced. Inherits ``diagnostics`` from ``PipelineError``.
    """


# ---------------------------------------------------------------------------
# Pipeline state (REQ-OR-050)
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    """Mutable runtime state for pipeline introspection (REQ-OR-050).

    Unlike other models in the codebase, this is **not** frozen so that
    the orchestrator can update it as the pipeline progresses.

    Attributes:
        current_phase: Current pipeline phase name.
        elapsed_seconds: Wall-clock seconds since pipeline start.
        phase2_path_statuses: Per-path status strings for Phase 2.
        best_score_so_far: Best score achieved across all phases.
        agent_call_count: Total number of agent calls made.
    """

    current_phase: str = "phase1"
    elapsed_seconds: float = 0.0
    phase2_path_statuses: list[str] = []
    best_score_so_far: float | None = None
    agent_call_count: int = 0


# ---------------------------------------------------------------------------
# Environment variable support (REQ-OR-046)
# ---------------------------------------------------------------------------

_ENV_FIELD_MAP: dict[str, str] = {
    "MLE_STAR_MODEL": "model",
    "MLE_STAR_LOG_LEVEL": "log_level",
    "MLE_STAR_TIME_LIMIT": "time_limit_seconds",
}
"""Maps environment variable names to PipelineConfig field names."""


def validate_api_key() -> None:
    """Check that ``ANTHROPIC_API_KEY`` is set and non-empty (REQ-OR-046).

    Raises:
        EnvironmentError: If the key is missing, empty, or whitespace-only.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key.strip():
        msg = (
            "ANTHROPIC_API_KEY environment variable is required but "
            "not set or empty. Set it to your Anthropic API key."
        )
        raise OSError(msg)


def apply_env_overrides(config: PipelineConfig) -> PipelineConfig:
    """Apply ``MLE_STAR_*`` env var overrides to a config (REQ-OR-046).

    Environment variables override **default** field values but do **not**
    override values explicitly set in the ``PipelineConfig`` constructor.
    A field is considered explicitly set when its value differs from the
    ``PipelineConfig`` default for that field.

    Invalid numeric values (non-parseable or violating validators) are
    silently ignored.

    Args:
        config: The pipeline configuration to apply overrides to.

    Returns:
        A new ``PipelineConfig`` with env var overrides applied.
    """
    defaults = PipelineConfig()
    overrides: dict[str, Any] = {}

    for env_var, field_name in _ENV_FIELD_MAP.items():
        env_value = os.environ.get(env_var)
        if env_value is None:
            continue

        current = getattr(config, field_name)
        default = getattr(defaults, field_name)
        if current != default:
            continue

        parsed = _parse_env_value(field_name, env_value)
        if parsed is not None:
            overrides[field_name] = parsed

    if not overrides:
        return config

    return config.model_copy(update=overrides)


def _parse_env_value(field_name: str, raw: str) -> Any:
    """Parse a raw env var string into the appropriate type for *field_name*.

    Args:
        field_name: The ``PipelineConfig`` field name.
        raw: The raw environment variable string.

    Returns:
        The parsed value, or ``None`` if parsing fails or would violate
        validators.
    """
    if field_name in ("model", "log_level"):
        return raw

    if field_name == "time_limit_seconds":
        try:
            value = int(raw)
        except ValueError:
            return None
        if value < 1:
            return None
        return value

    return None


# ---------------------------------------------------------------------------
# Logging configuration (REQ-OR-047)
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"


def configure_logging(config: PipelineConfig) -> None:
    """Configure Python logging for the pipeline (REQ-OR-047).

    Sets up the ``"mle_star"`` logger with a console handler and an
    optional file handler. Idempotent — repeated calls do not duplicate
    handlers.

    Args:
        config: Pipeline configuration providing ``log_level`` and
            optional ``log_file``.
    """
    mle_logger = logging.getLogger("mle_star")
    mle_logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # Avoid duplicating handlers on repeated calls
    if not any(isinstance(h, logging.StreamHandler) for h in mle_logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_LOG_FORMAT))
        mle_logger.addHandler(console)

    if config.log_file is not None:
        has_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(Path(config.log_file).resolve())
            for h in mle_logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            mle_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Input validation (REQ-OR-002)
# ---------------------------------------------------------------------------


def _validate_inputs(
    task: TaskDescription,
    config: PipelineConfig | None,
) -> PipelineConfig:
    """Validate pipeline inputs before any phase execution (REQ-OR-002).

    Checks that ``task.data_dir`` exists, is a directory, and contains at
    least one file. Returns the resolved ``PipelineConfig`` (defaulting to
    ``PipelineConfig()`` when *config* is ``None``).

    Args:
        task: Task description to validate.
        config: Pipeline configuration, or ``None`` for defaults.

    Returns:
        The validated (or default) ``PipelineConfig``.

    Raises:
        ValueError: If ``task.data_dir`` does not exist, is not a directory,
            or contains no files.
    """
    resolved_config = config if config is not None else PipelineConfig()

    data_path = Path(task.data_dir)
    if not data_path.exists():
        msg = f"data_dir does not exist: {task.data_dir}"
        raise ValueError(msg)
    if not data_path.is_dir():
        msg = f"data_dir is not a directory: {task.data_dir}"
        raise ValueError(msg)

    has_files = any(p.is_file() for p in data_path.iterdir())
    if not has_files:
        msg = f"data_dir contains no files: {task.data_dir}"
        raise ValueError(msg)

    return resolved_config


# ---------------------------------------------------------------------------
# System prompt construction (REQ-OR-007)
# ---------------------------------------------------------------------------

_KAGGLE_PERSONA = (
    "You are a Kaggle grandmaster with expert-level skills in machine learning, "
    "data science, and competitive data analysis. You approach every task "
    "methodically, writing clean, efficient, and well-tested code. You always "
    "validate your solutions against the training data before submission."
)


def _build_system_prompt(
    task: TaskDescription,
    gpu_info: dict[str, Any],
) -> str:
    """Construct the orchestrator system prompt (REQ-OR-007).

    Combines the Kaggle grandmaster persona with task-specific context
    (description, evaluation metric, metric direction) and GPU availability.

    Args:
        task: Task description providing context for the prompt.
        gpu_info: GPU detection results from ``detect_gpu_info()``.

    Returns:
        The fully assembled system prompt string.
    """
    gpu_section = _format_gpu_section(gpu_info)

    return (
        f"{_KAGGLE_PERSONA}\n\n"
        f"## Task\n{task.description}\n\n"
        f"## Evaluation\n"
        f"- Metric: {task.evaluation_metric}\n"
        f"- Direction: {task.metric_direction}\n\n"
        f"## Hardware\n{gpu_section}"
    )


def _format_gpu_section(gpu_info: dict[str, Any]) -> str:
    """Format GPU information for the system prompt.

    Args:
        gpu_info: GPU detection results.

    Returns:
        Human-readable GPU availability summary.
    """
    if gpu_info.get("cuda_available"):
        gpu_count = gpu_info.get("gpu_count", 0)
        gpu_names = gpu_info.get("gpu_names", [])
        names_str = ", ".join(str(n) for n in gpu_names) if gpu_names else "unknown"
        return f"GPU available: {gpu_count} x {names_str}"
    return "No GPU detected. Use CPU-only approaches."


# ---------------------------------------------------------------------------
# Retry with backoff (REQ-OR-052)
# ---------------------------------------------------------------------------


async def retry_with_backoff(
    client: ClaudeCodeClient,
    agent_type: AgentType,
    message: str,
    *,
    max_retries: int = 3,
    session_id: str | None = None,
) -> str:
    """Retry a failed agent invocation with exponential backoff (REQ-OR-052).

    Attempts to invoke the agent up to *max_retries* times. Delays
    between retries follow a ``2^i`` pattern (1 s, 2 s, 4 s, ...).

    Args:
        client: The Claude Code client.
        agent_type: Agent to invoke.
        message: Prompt message.
        max_retries: Maximum number of retry attempts. Defaults to 3.
        session_id: Optional session ID for resume.

    Returns:
        The agent response text.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    last_error: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return await client.send_message(
                agent_type=agent_type,
                message=message,
                session_id=session_id,
            )
        except RuntimeError as exc:
            last_error = exc
            if attempt < max_retries - 1:
                delay = 2**attempt
                logger.warning(
                    "Agent %s attempt %d/%d failed; retrying in %ds",
                    agent_type,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error
    msg = "retry_with_backoff called with max_retries=0"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Time budget computation (REQ-OR-025, REQ-OR-026)
# ---------------------------------------------------------------------------


def _compute_phase_budgets(
    config: PipelineConfig,
    remaining_seconds: float,
) -> dict[str, float]:
    """Compute per-phase time budgets from remaining time (REQ-OR-025).

    After Phase 1 completes, the remaining time is distributed among
    Phase 2, Phase 3, and Finalization proportionally. Per-path Phase 2
    budget is ``phase2_budget / L`` (REQ-OR-026).

    Args:
        config: Pipeline configuration with phase proportions and L.
        remaining_seconds: Seconds remaining after Phase 1.

    Returns:
        Dict with keys ``phase2``, ``phase3``, ``finalization``,
        and ``phase2_per_path``.
    """
    budget = config.phase_time_budget or PhaseTimeBudget()
    total_pct = budget.phase2_pct + budget.phase3_pct + budget.finalization_pct

    if total_pct <= 0 or remaining_seconds <= 0:
        return {
            "phase2": 0.0,
            "phase3": 0.0,
            "finalization": 0.0,
            "phase2_per_path": 0.0,
        }

    phase2 = remaining_seconds * budget.phase2_pct / total_pct
    phase3 = remaining_seconds * budget.phase3_pct / total_pct
    finalization = remaining_seconds * budget.finalization_pct / total_pct

    return {
        "phase2": phase2,
        "phase3": phase3,
        "finalization": finalization,
        "phase2_per_path": phase2 / config.num_parallel_solutions,
    }


# ---------------------------------------------------------------------------
# Phase 1 with deadline enforcement (REQ-OR-024, REQ-OR-030)
# ---------------------------------------------------------------------------


async def _execute_phase1_with_deadline(
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
    deadline: float,
    pipeline_start: float,
) -> Phase1Result:
    """Run Phase 1 with deadline enforcement (REQ-OR-024).

    Wraps ``run_phase1()`` in ``asyncio.wait_for()`` with the remaining
    time until *deadline*. If Phase 1 does not complete in time, raises
    ``PipelineTimeoutError`` (REQ-OR-030).

    Args:
        task: Task description.
        config: Pipeline configuration.
        client: Claude Code client.
        deadline: Absolute monotonic deadline.
        pipeline_start: Pipeline start time (for diagnostics).

    Returns:
        Phase 1 result.

    Raises:
        PipelineTimeoutError: If Phase 1 times out.
    """
    remaining = max(0.01, deadline - time.monotonic())
    logger.info("=== Phase 1: Initial Solution Generation ===")
    p1_start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            run_phase1(task, config, client),
            timeout=remaining,
        )
    except TimeoutError:
        elapsed = time.monotonic() - pipeline_start
        raise PipelineTimeoutError(
            f"Pipeline timed out during Phase 1 after {elapsed:.1f}s",
            diagnostics={"elapsed_time": elapsed},
        ) from None
    logger.info("Phase 1 completed in %.1fs", time.monotonic() - p1_start)
    return result


# ---------------------------------------------------------------------------
# Phase 3 with timeout and skip logic (REQ-OR-015, REQ-OR-024)
# ---------------------------------------------------------------------------


async def _execute_phase3_or_skip(
    client: ClaudeCodeClient,
    task: TaskDescription,
    config: PipelineConfig,
    phase2_solutions: list[SolutionScript],
    current_best: SolutionScript,
    deadline: float,
    budgets: dict[str, float],
) -> tuple[Phase3Result | None, SolutionScript]:
    """Run Phase 3 or skip based on L and deadline (REQ-OR-015, REQ-OR-024).

    Skips Phase 3 when ``L == 1`` (REQ-OR-015) or when the deadline has
    been exceeded. Wraps ``run_phase3()`` in ``asyncio.wait_for()`` with
    the Phase 3 budget.

    Args:
        client: Claude Code client.
        task: Task description.
        config: Pipeline configuration.
        phase2_solutions: Solutions from Phase 2 for ensemble.
        current_best: Best solution found so far (fallback).
        deadline: Absolute monotonic deadline.
        budgets: Phase time budgets from ``_compute_phase_budgets()``.

    Returns:
        Tuple of (phase3_result, best_solution).
    """
    if config.num_parallel_solutions <= 1:
        logger.info("Phase 3 skipped (L=1)")
        return None, current_best

    if time.monotonic() >= deadline:
        logger.warning("Deadline exceeded; skipping Phase 3")
        return None, current_best

    logger.info("=== Phase 3: Ensemble Construction ===")
    p3_start = time.monotonic()
    try:
        phase3_result = await asyncio.wait_for(
            run_phase3(client, task, config, phase2_solutions),
            timeout=max(0.01, budgets.get("phase3", 0.0)),
        )
        logger.info("Phase 3 completed in %.1fs", time.monotonic() - p3_start)
        return phase3_result, phase3_result.best_ensemble
    except TimeoutError:
        logger.warning(
            "Phase 3 timed out after %.1fs; using best Phase 2 solution",
            time.monotonic() - p3_start,
        )
        return None, current_best
    except Exception:
        logger.warning(
            "Phase 3 failed after %.1fs; using best Phase 2 solution",
            time.monotonic() - p3_start,
        )
        return None, current_best


# ---------------------------------------------------------------------------
# Phase 2 dispatch helpers (REQ-OR-013, REQ-OR-022, REQ-OR-040)
# ---------------------------------------------------------------------------


def _create_path_work_directories(
    task: TaskDescription,
    num_paths: int,
) -> list[Path]:
    """Create per-path working subdirectories (REQ-OR-020).

    Creates ``./work/path-{i}/`` directories relative to the parent of
    ``task.data_dir`` for each of the L parallel paths. This ensures each
    path has an isolated filesystem area to avoid conflicts.

    Args:
        task: Task description providing the data directory context.
        num_paths: Number of parallel paths (L).

    Returns:
        List of L ``Path`` objects, one per path's working directory.
    """
    base = Path(task.data_dir).parent / "work"
    path_dirs: list[Path] = []
    for i in range(num_paths):
        path_dir = base / f"path-{i}"
        path_dir.mkdir(parents=True, exist_ok=True)
        path_dirs.append(path_dir)
    return path_dirs


async def _dispatch_phase2(
    client: ClaudeCodeClient,
    task: TaskDescription,
    config: PipelineConfig,
    phase1_result: Phase1Result,
    *,
    phase2_timeout: float | None = None,
) -> list[Phase2Result | BaseException]:
    """Dispatch L parallel Phase 2 paths with isolation (REQ-OR-013, REQ-OR-018).

    Creates one ``run_phase2_outer_loop`` coroutine per path. Each path
    receives a deep copy of the Phase 1 initial solution (REQ-OR-020),
    its own working subdirectory (REQ-OR-020), and a unique session ID
    (REQ-OR-021). Uses ``return_exceptions=True`` so that individual path
    failures do not cancel siblings (REQ-OR-022).

    When *phase2_timeout* is set, paths still running after the timeout
    are cancelled via ``asyncio.Task.cancel()`` (REQ-OR-023).

    Args:
        client: Claude Code client for agent invocations.
        task: Task description.
        config: Pipeline configuration (L = ``num_parallel_solutions``).
        phase1_result: Phase 1 output providing the initial solution.
        phase2_timeout: Maximum seconds to wait for all paths. ``None``
            means no timeout (wait indefinitely).

    Returns:
        List of L results, each either a ``Phase2Result`` or an exception.
        Cancelled paths appear as ``asyncio.CancelledError`` instances.
    """
    num_paths = config.num_parallel_solutions

    # REQ-OR-020: Create per-path working directories
    _create_path_work_directories(task, num_paths)

    # REQ-OR-020: Deep copy the initial solution for each path
    phase2_coros = [
        run_phase2_outer_loop(
            client,
            task,
            config,
            copy.deepcopy(phase1_result.initial_solution),
            phase1_result.initial_score,
            session_id=f"path-{i}",
        )
        for i in range(num_paths)
    ]

    if phase2_timeout is None:
        return await asyncio.gather(*phase2_coros, return_exceptions=True)

    # REQ-OR-023: Bounded wait with cancellation of overtime paths
    tasks = [asyncio.create_task(coro) for coro in phase2_coros]
    _done, pending = await asyncio.wait(tasks, timeout=phase2_timeout)

    for overtime_task in pending:
        overtime_task.cancel()
        logger.warning("Phase 2 path cancelled due to timeout")

    # Wait for cancelled tasks to finish their cancellation
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    # Collect results in original order (preserving path indices)
    results: list[Phase2Result | BaseException] = []
    for t in tasks:
        try:
            results.append(t.result())
        except BaseException as exc:
            results.append(exc)

    return results


async def _dispatch_phase2_with_session_limit(
    client: ClaudeCodeClient,
    task: TaskDescription,
    config: PipelineConfig,
    phase1_result: Phase1Result,
    max_concurrent_sessions: int,
) -> list[Phase2Result | BaseException]:
    """Dispatch Phase 2 paths with a concurrency limit (REQ-OR-056).

    When ``max_concurrent_sessions < L``, excess paths are serialized
    (limited by an ``asyncio.Semaphore``) and a warning is logged.

    Args:
        client: Claude Code client for agent invocations.
        task: Task description.
        config: Pipeline configuration (L = ``num_parallel_solutions``).
        phase1_result: Phase 1 output providing the initial solution.
        max_concurrent_sessions: Maximum number of simultaneously
            running Phase 2 paths.

    Returns:
        List of L results, each either a ``Phase2Result`` or an exception.
    """
    num_paths = config.num_parallel_solutions

    if max_concurrent_sessions < num_paths:
        logger.warning(
            "Concurrent session limit (%d) below L=%d; "
            "serializing excess Phase 2 paths",
            max_concurrent_sessions,
            num_paths,
        )

    _create_path_work_directories(task, num_paths)
    sem = asyncio.Semaphore(max_concurrent_sessions)

    async def _limited_path(idx: int) -> Phase2Result:
        async with sem:
            return await run_phase2_outer_loop(
                client,
                task,
                config,
                copy.deepcopy(phase1_result.initial_solution),
                phase1_result.initial_score,
                session_id=f"path-{idx}",
            )

    coros = [_limited_path(i) for i in range(num_paths)]
    return await asyncio.gather(*coros, return_exceptions=True)


def _make_failed_phase2_result(phase1_result: Phase1Result) -> Phase2Result:
    """Create a synthetic Phase2Result for a failed Phase 2 path (REQ-OR-040).

    Uses the Phase 1 initial solution and score as fallback values.
    The ``step_history`` contains a single entry marking the failure.

    Args:
        phase1_result: Phase 1 output providing fallback solution and score.

    Returns:
        A synthetic ``Phase2Result`` representing a failed path.
    """
    return Phase2Result(
        ablation_summaries=[],
        refined_blocks=[],
        best_solution=phase1_result.initial_solution,
        best_score=phase1_result.initial_score,
        step_history=[{"step": 0, "failed": True}],
    )


def _collect_phase2_results(
    raw_results: list[Phase2Result | BaseException],
    phase1_result: Phase1Result,
) -> tuple[list[Phase2Result], list[SolutionScript]]:
    """Separate Phase 2 results and build Phase 3 input (REQ-OR-040).

    For each path: if the result is a ``Phase2Result``, it is used directly;
    if it is an exception, a synthetic ``Phase2Result`` is created from
    the Phase 1 result as a fallback (REQ-OR-040). Both output lists always
    have the same length as ``raw_results``.

    Args:
        raw_results: Mixed list from ``asyncio.gather(return_exceptions=True)``.
        phase1_result: Fallback source for failed paths.

    Returns:
        Tuple of (phase2_results, solutions_for_phase3) where both
        lists have ``len(raw_results)`` entries.
    """
    phase2_results: list[Phase2Result] = []
    solutions: list[SolutionScript] = []

    for i, result in enumerate(raw_results):
        if isinstance(result, BaseException):
            logger.warning("Phase 2 path %d failed: %s", i, result)
            synthetic = _make_failed_phase2_result(phase1_result)
            phase2_results.append(synthetic)
            solutions.append(phase1_result.initial_solution)
        else:
            phase2_results.append(result)
            solutions.append(result.best_solution)

    return phase2_results, solutions


# ---------------------------------------------------------------------------
# Result assembly and logging (REQ-OR-036 through REQ-OR-039)
# ---------------------------------------------------------------------------


async def _finalize_with_recovery(
    *,
    client: ClaudeCodeClient,
    best_solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    phase1_result: Phase1Result,
    phase2_results: list[Phase2Result],
    phase3_result: Phase3Result | None,
    pipeline_start: float,
) -> FinalResult:
    """Run finalization with error recovery (REQ-OR-043).

    Wraps ``run_finalization()`` in a try/except. On success, updates the
    returned ``FinalResult`` with pipeline-level ``total_duration_seconds``
    (REQ-OR-036). On failure, constructs a best-effort ``FinalResult``
    with ``submission_path=""``.

    Args:
        client: Claude Code client.
        best_solution: Best solution to finalize.
        task: Task description.
        config: Pipeline configuration.
        phase1_result: Phase 1 output.
        phase2_results: All Phase 2 results (including synthetic).
        phase3_result: Phase 3 output (or None).
        pipeline_start: Monotonic time when the pipeline started.

    Returns:
        A ``FinalResult`` with pipeline-level duration.
    """
    try:
        logger.info("=== Finalization ===")
        final_result = await run_finalization(
            client,
            best_solution,
            task,
            config,
            phase1_result,
            phase2_results,
            phase3_result,
        )
        total_duration = time.monotonic() - pipeline_start
        return final_result.model_copy(
            update={
                "total_duration_seconds": total_duration,
            }
        )
    except Exception:
        total_duration = time.monotonic() - pipeline_start
        logger.warning(
            "Finalization failed after %.1fs; returning best-effort result",
            total_duration,
        )
        return FinalResult(
            task=task,
            config=config,
            phase1=phase1_result,
            phase2_results=phase2_results,
            phase3=phase3_result,
            final_solution=best_solution,
            submission_path="",
            total_duration_seconds=total_duration,
        )


def _log_phase_summary(
    phase_durations: dict[str, float],
) -> None:
    """Log per-phase duration breakdown (REQ-OR-037).

    Emits a structured JSON log entry with duration per phase
    for post-pipeline analysis and observability.

    Args:
        phase_durations: Phase name to duration (seconds) mapping.
    """
    summary = {
        "durations": phase_durations,
    }
    logger.info("Phase summary: %s", json.dumps(summary, default=str))


def _log_solution_lineage(
    phase1_result: Phase1Result,
    phase2_results: list[Phase2Result],
    phase3_result: Phase3Result | None,
    final_solution: SolutionScript,
) -> None:
    """Log solution lineage tracing from final to origin (REQ-OR-039).

    Traces the solution evolution through each pipeline phase, recording
    the score at each step for debugging and auditing.

    Args:
        phase1_result: Phase 1 output.
        phase2_results: Phase 2 outputs for all paths.
        phase3_result: Phase 3 output (or None).
        final_solution: The solution sent for submission.
    """
    lineage: dict[str, Any] = {
        "phase1_score": phase1_result.initial_score,
        "phase2_scores": [r.best_score for r in phase2_results],
    }
    if phase3_result is not None:
        lineage["phase3_score"] = phase3_result.best_ensemble_score
    lineage["final_phase"] = str(final_solution.phase)
    logger.info("Solution lineage: %s", json.dumps(lineage, default=str))


# ---------------------------------------------------------------------------
# Client creation helper (REQ-OR-005)
# ---------------------------------------------------------------------------


def _create_client(
    config: PipelineConfig,
    task: TaskDescription,
) -> ClaudeCodeClient:
    """Create a Claude Code headless client (REQ-OR-005).

    Builds the system prompt and agent configurations, and returns
    a configured ``ClaudeCodeClient``.

    Args:
        config: Pipeline configuration.
        task: Task description for system prompt construction.

    Returns:
        A configured Claude Code client.
    """
    gpu_info = detect_gpu_info()
    system_prompt = _build_system_prompt(task, gpu_info)
    agent_configs = build_default_agent_configs()

    return ClaudeCodeClient(
        system_prompt=system_prompt,
        agent_configs=agent_configs,
        model=config.model,
        permission_mode=config.permission_mode,
    )


# ---------------------------------------------------------------------------
# Pipeline entry points (REQ-OR-001, REQ-OR-053)
# ---------------------------------------------------------------------------


async def run_pipeline(
    task: TaskDescription,
    config: PipelineConfig | None = None,
) -> FinalResult:
    """Execute the full MLE-STAR pipeline (REQ-OR-001).

    Validates inputs, creates the Claude Code client with all 14 agents
    configured, and dispatches pipeline phases in sequence. Enforces
    ``config.time_limit_seconds`` via a monotonic deadline (REQ-OR-024).

    Args:
        task: Description of the competition task to solve.
        config: Pipeline configuration. Defaults to paper-specified values
            when ``None``.

    Returns:
        A fully populated ``FinalResult`` with all phase outputs.

    Raises:
        ValueError: If inputs fail validation (REQ-OR-002).
        EnvironmentError: If ``ANTHROPIC_API_KEY`` is not set (REQ-OR-046).
        PipelineError: If the pipeline encounters an unrecoverable error.
        PipelineTimeoutError: If Phase 1 does not complete in time.
    """
    resolved_config = _validate_inputs(task, config)

    # REQ-OR-046: Validate API key and apply env var overrides
    validate_api_key()
    resolved_config = apply_env_overrides(resolved_config)

    # REQ-OR-047: Configure logging
    configure_logging(resolved_config)

    # REQ-OR-054: Verify Claude CLI availability
    check_claude_cli_version()

    # REQ-OR-024: Compute deadline at pipeline start
    pipeline_start = time.monotonic()
    deadline = pipeline_start + resolved_config.time_limit_seconds

    # Create client (no connect/disconnect needed for subprocess-based client)
    client = _create_client(resolved_config, task)

    setup_working_directory(task.data_dir)

    # Phase 1 with deadline enforcement (REQ-OR-024, REQ-OR-030)
    phase1_result = await _execute_phase1_with_deadline(
        task, resolved_config, client, deadline, pipeline_start
    )

    # Compute remaining time budgets (REQ-OR-025)
    remaining = max(0.0, deadline - time.monotonic())
    budgets = _compute_phase_budgets(resolved_config, remaining)

    # Phases 2-Final with graceful shutdown (REQ-OR-030)
    return await _execute_post_phase1(
        client,
        task,
        resolved_config,
        phase1_result,
        deadline,
        budgets,
        pipeline_start=pipeline_start,
    )


async def _execute_post_phase1(
    client: ClaudeCodeClient,
    task: TaskDescription,
    config: PipelineConfig,
    phase1_result: Phase1Result,
    deadline: float,
    budgets: dict[str, float],
    *,
    pipeline_start: float,
) -> FinalResult:
    """Execute Phase 2 through Finalization with graceful shutdown (REQ-OR-030).

    If the deadline is exceeded before Phase 2 starts, skips directly to
    finalization with the Phase 1 solution. Phase 2 receives a computed
    timeout (REQ-OR-026). Phase 3 is handled by ``_execute_phase3_or_skip()``.
    Finalization is wrapped in ``_finalize_with_recovery()`` (REQ-OR-043).

    Args:
        client: Claude Code client.
        task: Task description.
        config: Pipeline configuration.
        phase1_result: Phase 1 output.
        deadline: Absolute monotonic deadline (REQ-OR-024).
        budgets: Phase time budgets from ``_compute_phase_budgets()``.
        pipeline_start: Monotonic time when the pipeline started.

    Returns:
        The assembled ``FinalResult``.
    """
    best_solution = phase1_result.initial_solution
    phase2_results: list[Phase2Result] = []
    phase2_solutions: list[SolutionScript] = [phase1_result.initial_solution]
    phase3_result: Phase3Result | None = None

    # Phase 2 with deadline check and computed timeout (REQ-OR-024, REQ-OR-026)
    if time.monotonic() < deadline:
        logger.info("=== Phase 2: Targeted Refinement ===")
        p2_start = time.monotonic()
        raw_phase2 = await _dispatch_phase2(
            client,
            task,
            config,
            phase1_result,
            phase2_timeout=budgets["phase2"],
        )
        phase2_results, phase2_solutions = _collect_phase2_results(
            raw_phase2, phase1_result
        )
        best_solution = phase2_solutions[0]
        logger.info("Phase 2 completed in %.1fs", time.monotonic() - p2_start)
    else:
        logger.warning("Deadline exceeded; skipping Phase 2")

    # Phase 3 with timeout and skip logic (REQ-OR-015, REQ-OR-024)
    phase3_result, best_solution = await _execute_phase3_or_skip(
        client, task, config, phase2_solutions, best_solution, deadline, budgets
    )

    # Finalization with error recovery (REQ-OR-043)
    final_result = await _finalize_with_recovery(
        client=client,
        best_solution=best_solution,
        task=task,
        config=config,
        phase1_result=phase1_result,
        phase2_results=phase2_results,
        phase3_result=phase3_result,
        pipeline_start=pipeline_start,
    )

    # Log summaries (REQ-OR-037, REQ-OR-038, REQ-OR-039)
    _log_solution_lineage(
        phase1_result, phase2_results, phase3_result, final_result.final_solution
    )

    return final_result


def run_pipeline_sync(
    task: TaskDescription,
    config: PipelineConfig | None = None,
) -> FinalResult:
    """Synchronous wrapper for ``run_pipeline()`` (REQ-OR-053).

    Delegates to :func:`run_pipeline` via ``asyncio.run()``.

    Args:
        task: Description of the competition task to solve.
        config: Pipeline configuration. Defaults to paper-specified values
            when ``None``.

    Returns:
        A fully populated ``FinalResult`` with all phase outputs.

    Raises:
        ValueError: If inputs fail validation.
        PipelineError: If the pipeline encounters an unrecoverable error.
        PipelineTimeoutError: If Phase 1 does not complete in time.
    """
    return asyncio.run(run_pipeline(task, config))
