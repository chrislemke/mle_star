"""Pipeline orchestrator: entry point, SDK client setup, and phase dispatch.

Provides ``run_pipeline()`` (async) and ``run_pipeline_sync()`` (sync wrapper)
as the top-level entry points for the MLE-STAR pipeline. Handles input
validation, SDK client lifecycle, agent registration, system prompt
construction, MCP server registration, sequential phase dispatch, time
budgeting, cost tracking, and graceful shutdown.

Refs:
    SRS 09a -- Orchestrator Entry Point & SDK Client Setup.
    SRS 09b -- Orchestrator Phase Dispatch & Sequencing.
    SRS 09c -- Orchestrator Budgets & Hooks.
    IMPLEMENTATION_PLAN.md Tasks 42, 43, 44, 45, 46.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
)

from mle_star.execution import detect_gpu_info, setup_working_directory
from mle_star.finalization import run_finalization
from mle_star.models import (
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
# Custom exceptions (REQ-OR-042, REQ-OR-030)
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Pipeline failure with diagnostic context (REQ-OR-042).

    Raised when the pipeline encounters an unrecoverable error. The
    ``diagnostics`` dict carries structured context (elapsed time, cost,
    last successful operation) for debugging.

    Attributes:
        diagnostics: Structured diagnostic information about the failure.
    """

    def __init__(self, message: str, *, diagnostics: dict[str, Any]) -> None:
        """Initialize with a message and structured diagnostics.

        Args:
            message: Human-readable error description.
            diagnostics: Structured context (elapsed time, cost, etc.).
        """
        super().__init__(message)
        self.diagnostics = diagnostics


class PipelineTimeoutError(PipelineError):
    """Pipeline timed out before Phase 1 completed (REQ-OR-030).

    Raised when the overall time budget expires before a usable solution
    has been produced. Inherits ``diagnostics`` from ``PipelineError``.
    """


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
# Agent registration (REQ-OR-006)
# ---------------------------------------------------------------------------


def _build_agents_dict() -> dict[str, dict[str, Any]]:
    """Build the agents dictionary for SDK client registration (REQ-OR-006).

    Converts all 14 ``AgentConfig`` instances (from
    ``build_default_agent_configs()``) into agent definition dicts via
    ``to_agent_definition()``, keyed by their ``AgentType`` string value.

    Returns:
        Dict mapping agent type strings to agent definition dicts.
    """
    configs = build_default_agent_configs()
    return {
        str(agent_type): config.to_agent_definition()
        for agent_type, config in configs.items()
    }


# ---------------------------------------------------------------------------
# MCP server registration (REQ-OR-010)
# ---------------------------------------------------------------------------


def _register_mcp_servers(client: ClaudeSDKClient) -> None:
    """Register MCP servers for custom tool capabilities (REQ-OR-010).

    Attempts to register score-parsing and file-listing MCP tools.
    This is a placeholder for future MCP integration.

    Args:
        client: The SDK client to register servers on.
    """


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
# Cost tracking (REQ-OR-027, REQ-OR-029)
# ---------------------------------------------------------------------------


class CostTracker:
    """Thread-safe cost accumulator with budget enforcement (REQ-OR-027).

    Tracks the running total of API costs and logs a warning when 80%
    of the budget is consumed (REQ-OR-029). Thread-safe for concurrent
    Phase 2 paths.

    Attributes:
        total: Current accumulated cost in USD.
        exceeded: Whether the budget has been reached or exceeded.
    """

    def __init__(self, max_budget: float | None = None) -> None:
        """Initialize the cost tracker.

        Args:
            max_budget: Maximum budget in USD. ``None`` means unlimited.
        """
        self._total = 0.0
        self._max_budget = max_budget
        self._warned_80pct = False
        self._lock = threading.Lock()

    def accumulate(self, amount: float) -> None:
        """Add a cost amount to the running total.

        Logs a warning the first time the 80% threshold is crossed
        (REQ-OR-029).

        Args:
            amount: Cost in USD to add.
        """
        with self._lock:
            self._total += amount
            if (
                self._max_budget is not None
                and not self._warned_80pct
                and self._total >= 0.8 * self._max_budget
            ):
                logger.warning(
                    "80%% of budget consumed: $%.2f / $%.2f",
                    self._total,
                    self._max_budget,
                )
                self._warned_80pct = True

    @property
    def total(self) -> float:
        """Current accumulated cost in USD."""
        return self._total

    @property
    def exceeded(self) -> bool:
        """Whether the accumulated cost has reached or exceeded the budget."""
        if self._max_budget is None:
            return False
        return self._total >= self._max_budget


# ---------------------------------------------------------------------------
# Hook system (REQ-OR-031 through REQ-OR-035)
# ---------------------------------------------------------------------------

_DEFAULT_BLOCKED_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{",
]
"""Default dangerous bash command patterns blocked by the safety hook."""


def create_progress_hook(
    pipeline_start: float,
    session_agent_map: dict[str, str],
) -> Any:
    """Create a PostToolUse hook for structured JSON logging (REQ-OR-031).

    Logs a JSON entry with timestamp, agent type, tool name, session ID,
    elapsed time, and success indicator on every tool use completion.

    Args:
        pipeline_start: Monotonic time when the pipeline started.
        session_agent_map: Shared mapping of session_id to agent_type.

    Returns:
        An async hook callback for PostToolUse events.
    """

    async def _hook(
        hook_input: Any,
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        elapsed = time.monotonic() - pipeline_start
        entry = {
            "timestamp": time.time(),
            "agent_type": session_agent_map.get(hook_input["session_id"], "unknown"),
            "tool_name": hook_input["tool_name"],
            "session_id": hook_input["session_id"],
            "elapsed_time": round(elapsed, 3),
            "success": True,
        }
        logger.info(json.dumps(entry))
        return {}

    return _hook


def create_cost_hook(cost_tracker: CostTracker) -> Any:
    """Create a Stop/SubagentStop hook for cost status logging (REQ-OR-032).

    Logs the current accumulated cost on each agent stop event for
    observability and budget monitoring.

    Args:
        cost_tracker: Shared thread-safe cost accumulator.

    Returns:
        An async hook callback for Stop/SubagentStop events.
    """

    async def _hook(
        hook_input: Any,
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        logger.info(
            "Cost status: $%.2f accumulated (session=%s)",
            cost_tracker.total,
            hook_input["session_id"],
        )
        return {}

    return _hook


def create_safety_hook(
    work_dir: str,
    blocked_patterns: list[str] | None = None,
) -> Any:
    """Create a PreToolUse hook that blocks dangerous bash commands (REQ-OR-033).

    Inspects Bash tool inputs against a configurable list of regex patterns.
    Returns a deny decision with an explanation when a match is found.

    Args:
        work_dir: The pipeline working directory.
        blocked_patterns: Regex patterns to block. Defaults to
            ``_DEFAULT_BLOCKED_PATTERNS`` when ``None``.

    Returns:
        An async hook callback for PreToolUse events.
    """
    patterns = (
        blocked_patterns
        if blocked_patterns is not None
        else list(_DEFAULT_BLOCKED_PATTERNS)
    )
    compiled = [re.compile(p) for p in patterns]

    async def _hook(
        hook_input: Any,
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        if hook_input.get("tool_name") != "Bash":
            return {}
        command = hook_input.get("tool_input", {}).get("command", "")
        matched = _check_blocked_command(command, compiled)
        if matched is not None:
            return _make_deny_result(matched, command)
        return {}

    return _hook


def _check_blocked_command(
    command: str,
    compiled_patterns: list[re.Pattern[str]],
) -> str | None:
    """Check a command against compiled blocked patterns.

    Args:
        command: The bash command string to check.
        compiled_patterns: Pre-compiled regex patterns.

    Returns:
        The matched pattern string, or ``None`` if no match.
    """
    for pattern in compiled_patterns:
        if pattern.search(command):
            return pattern.pattern
    return None


def _make_deny_result(matched_pattern: str, command: str) -> dict[str, Any]:
    """Build a deny hook result with explanation.

    Args:
        matched_pattern: The pattern that matched.
        command: The blocked command.

    Returns:
        A SyncHookJSONOutput-compatible dict with deny decision.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"Blocked dangerous command matching /{matched_pattern}/: "
                f"{command[:100]}"
            ),
        },
    }


def create_timeout_hook(
    deadline: float,
    time_limit: float,
    finalize_flag: threading.Event,
) -> Any:
    """Create a PostToolUse hook for deadline monitoring (REQ-OR-034).

    Sets the finalize flag when remaining time drops below the threshold
    ``max(10% of time_limit, 300 seconds)``.

    Args:
        deadline: Absolute monotonic deadline for the pipeline.
        time_limit: Total time budget in seconds.
        finalize_flag: Shared event to signal finalization.

    Returns:
        An async hook callback for PostToolUse events.
    """
    threshold = max(0.10 * time_limit, 300.0)

    async def _hook(
        hook_input: Any,
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining < threshold:
            finalize_flag.set()
        return {}

    return _hook


def create_error_hook(
    failure_counts: dict[str, int],
    failure_lock: threading.Lock,
) -> Any:
    """Create a PostToolUseFailure hook for error tracking (REQ-OR-035).

    Logs failure details and tracks consecutive failure counts per session
    for circuit-breaker logic.

    Args:
        failure_counts: Shared dict mapping session_id to failure count.
        failure_lock: Lock for thread-safe failure count updates.

    Returns:
        An async hook callback for PostToolUseFailure events.
    """

    async def _hook(
        hook_input: Any,
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        session_id = hook_input["session_id"]
        tool_name = hook_input.get("tool_name", "unknown")
        error = hook_input.get("error", "unknown error")
        with failure_lock:
            failure_counts[session_id] = failure_counts.get(session_id, 0) + 1
        logger.warning(
            "Tool failure: tool=%s session=%s error=%s",
            tool_name,
            session_id,
            error,
        )
        return {}

    return _hook


def create_agent_tracking_hook(
    session_agent_map: dict[str, str],
) -> Any:
    """Create a SubagentStart hook to track session-to-agent mapping.

    Populates a shared mapping so other hooks can resolve session IDs
    to agent type names.

    Args:
        session_agent_map: Shared dict to populate with mappings.

    Returns:
        An async hook callback for SubagentStart events.
    """

    async def _hook(
        hook_input: Any,
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        session_agent_map[hook_input["session_id"]] = hook_input["agent_type"]
        return {}

    return _hook


def build_hooks(
    pipeline_start: float,
    deadline: float,
    time_limit: float,
    cost_tracker: CostTracker,
    work_dir: str,
    finalize_flag: threading.Event,
    failure_counts: dict[str, int],
    failure_lock: threading.Lock,
    session_agent_map: dict[str, str],
    blocked_patterns: list[str] | None = None,
) -> dict[str, list[HookMatcher]]:
    """Assemble all SDK hooks into the registration dict (REQ-OR-031..035).

    Creates and groups all hook callbacks into the dict format expected
    by ``ClaudeAgentOptions.hooks``.

    Args:
        pipeline_start: Monotonic time when the pipeline started.
        deadline: Absolute monotonic deadline.
        time_limit: Total time budget in seconds.
        cost_tracker: Shared thread-safe cost accumulator.
        work_dir: Pipeline working directory path.
        finalize_flag: Shared event for finalization signaling.
        failure_counts: Shared dict for consecutive failure tracking.
        failure_lock: Lock for thread-safe failure count updates.
        session_agent_map: Shared session-to-agent-type mapping.
        blocked_patterns: Custom blocked command patterns (optional).

    Returns:
        Dict mapping event name strings to lists of ``HookMatcher``.
    """
    progress = create_progress_hook(pipeline_start, session_agent_map)
    cost = create_cost_hook(cost_tracker)
    safety = create_safety_hook(work_dir, blocked_patterns)
    timeout = create_timeout_hook(deadline, time_limit, finalize_flag)
    error = create_error_hook(failure_counts, failure_lock)
    tracking = create_agent_tracking_hook(session_agent_map)

    return {
        "PreToolUse": [HookMatcher(matcher="Bash", hooks=[safety])],
        "PostToolUse": [HookMatcher(hooks=[progress, timeout])],
        "Stop": [HookMatcher(hooks=[cost])],
        "SubagentStop": [HookMatcher(hooks=[cost])],
        "SubagentStart": [HookMatcher(hooks=[tracking])],
        "PostToolUseFailure": [HookMatcher(hooks=[error])],
    }


# ---------------------------------------------------------------------------
# Phase 1 with deadline enforcement (REQ-OR-024, REQ-OR-030)
# ---------------------------------------------------------------------------


async def _execute_phase1_with_deadline(
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeSDKClient,
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
        client: SDK client.
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
    client: ClaudeSDKClient,
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
        client: SDK client.
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
    client: ClaudeSDKClient,
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
        client: SDK client for agent invocations.
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


def _collect_phase2_results(
    raw_results: list[Phase2Result | BaseException],
    phase1_result: Phase1Result,
) -> tuple[list[Phase2Result], list[SolutionScript]]:
    """Separate successful Phase 2 results and build Phase 3 input (REQ-OR-040).

    For each path: if the result is a ``Phase2Result``, its ``best_solution``
    is used; if it is an exception, the Phase 1 initial solution is
    substituted as a fallback (REQ-OR-040).

    Args:
        raw_results: Mixed list from ``asyncio.gather(return_exceptions=True)``.
        phase1_result: Fallback source for failed paths.

    Returns:
        Tuple of (phase2_results, solutions_for_phase3) where
        ``phase2_results`` contains only successful results and
        ``solutions_for_phase3`` has one entry per path.
    """
    phase2_results: list[Phase2Result] = []
    solutions: list[SolutionScript] = []

    for i, result in enumerate(raw_results):
        if isinstance(result, BaseException):
            logger.warning("Phase 2 path %d failed: %s", i, result)
            solutions.append(phase1_result.initial_solution)
        else:
            phase2_results.append(result)
            solutions.append(result.best_solution)

    return phase2_results, solutions


# ---------------------------------------------------------------------------
# Pipeline entry points (REQ-OR-001, REQ-OR-053)
# ---------------------------------------------------------------------------


async def run_pipeline(
    task: TaskDescription,
    config: PipelineConfig | None = None,
) -> FinalResult:
    """Execute the full MLE-STAR pipeline (REQ-OR-001).

    Validates inputs, creates the SDK client with all 14 agents registered,
    and dispatches pipeline phases in sequence. Enforces
    ``config.time_limit_seconds`` via a monotonic deadline (REQ-OR-024).
    The client is always disconnected on exit via try/finally.

    Args:
        task: Description of the competition task to solve.
        config: Pipeline configuration. Defaults to paper-specified values
            when ``None``.

    Returns:
        A fully populated ``FinalResult`` with all phase outputs.

    Raises:
        ValueError: If inputs fail validation (REQ-OR-002).
        PipelineError: If the pipeline encounters an unrecoverable error.
        PipelineTimeoutError: If Phase 1 does not complete in time.
    """
    resolved_config = _validate_inputs(task, config)

    # REQ-OR-024: Compute deadline at pipeline start
    pipeline_start = time.monotonic()
    deadline = pipeline_start + resolved_config.time_limit_seconds

    # REQ-OR-031..035: Build SDK hooks for observability and safety
    cost_tracker = CostTracker(max_budget=resolved_config.max_budget_usd)
    finalize_flag = threading.Event()
    failure_counts: dict[str, int] = {}
    failure_lock = threading.Lock()
    session_agent_map: dict[str, str] = {}

    hooks = build_hooks(
        pipeline_start=pipeline_start,
        deadline=deadline,
        time_limit=resolved_config.time_limit_seconds,
        cost_tracker=cost_tracker,
        work_dir=task.data_dir,
        finalize_flag=finalize_flag,
        failure_counts=failure_counts,
        failure_lock=failure_lock,
        session_agent_map=session_agent_map,
    )

    client = _create_sdk_client(resolved_config, task, hooks=hooks)

    try:
        await client.connect()
        _try_register_mcp(client)
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
            client, task, resolved_config, phase1_result, deadline, budgets
        )
    finally:
        await client.disconnect()


def _create_sdk_client(
    config: PipelineConfig,
    task: TaskDescription,
    hooks: dict[str, list[HookMatcher]] | None = None,
) -> ClaudeSDKClient:
    """Create and configure the SDK client (REQ-OR-005).

    Builds the system prompt, agent definitions, and SDK options.
    Optionally registers hooks for observability and safety.

    Args:
        config: Pipeline configuration.
        task: Task description for system prompt construction.
        hooks: SDK hook registrations, or ``None`` to skip hooks.

    Returns:
        A configured but not-yet-connected SDK client.
    """
    gpu_info = detect_gpu_info()
    system_prompt = _build_system_prompt(task, gpu_info)
    agents_dict = _build_agents_dict()

    agent_definitions: dict[str, AgentDefinition] = {
        name: AgentDefinition(**defn) for name, defn in agents_dict.items()
    }

    options = ClaudeAgentOptions(
        model=config.model,
        permission_mode=config.permission_mode,  # type: ignore[arg-type]
        max_budget_usd=config.max_budget_usd,
        agents=agent_definitions,
        system_prompt=system_prompt,
        hooks=hooks,  # type: ignore[arg-type]
    )

    return ClaudeSDKClient(options)


def _try_register_mcp(client: ClaudeSDKClient) -> None:
    """Attempt MCP server registration, logging failures (REQ-OR-010).

    Args:
        client: The SDK client.
    """
    try:
        _register_mcp_servers(client)
    except Exception:
        logger.warning("MCP server registration failed; continuing without MCP")


async def _execute_post_phase1(
    client: ClaudeSDKClient,
    task: TaskDescription,
    config: PipelineConfig,
    phase1_result: Phase1Result,
    deadline: float,
    budgets: dict[str, float],
) -> FinalResult:
    """Execute Phase 2 through Finalization with graceful shutdown (REQ-OR-030).

    If the deadline is exceeded before Phase 2 starts, skips directly to
    finalization with the Phase 1 solution. Phase 2 receives a computed
    timeout (REQ-OR-026). Phase 3 is handled by ``_execute_phase3_or_skip()``.

    Args:
        client: SDK client.
        task: Task description.
        config: Pipeline configuration.
        phase1_result: Phase 1 output.
        deadline: Absolute monotonic deadline (REQ-OR-024).
        budgets: Phase time budgets from ``_compute_phase_budgets()``.

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

    # Finalization (REQ-OR-016)
    logger.info("=== Finalization ===")
    p_fin_start = time.monotonic()
    final_result = await run_finalization(
        client,
        best_solution,
        task,
        config,
        phase1_result,
        phase2_results,
        phase3_result,
    )
    logger.info("Finalization completed in %.1fs", time.monotonic() - p_fin_start)

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
