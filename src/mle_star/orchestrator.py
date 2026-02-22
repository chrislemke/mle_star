"""Pipeline orchestrator: entry point, SDK client setup, and phase dispatch.

Provides ``run_pipeline()`` (async) and ``run_pipeline_sync()`` (sync wrapper)
as the top-level entry points for the MLE-STAR pipeline. Handles input
validation, SDK client lifecycle, agent registration, system prompt
construction, MCP server registration, and sequential phase dispatch.

Refs:
    SRS 09a -- Orchestrator Entry Point & SDK Client Setup.
    SRS 09b -- Orchestrator Phase Dispatch & Sequencing.
    IMPLEMENTATION_PLAN.md Tasks 42, 43.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import time
from typing import Any

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKClient,
)

from mle_star.execution import detect_gpu_info, setup_working_directory
from mle_star.finalization import run_finalization
from mle_star.models import (
    FinalResult,
    Phase1Result,
    Phase2Result,
    Phase3Result,
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
# Phase 2 dispatch helpers (REQ-OR-013, REQ-OR-022, REQ-OR-040)
# ---------------------------------------------------------------------------


async def _dispatch_phase2(
    client: ClaudeSDKClient,
    task: TaskDescription,
    config: PipelineConfig,
    phase1_result: Phase1Result,
) -> list[Phase2Result | BaseException]:
    """Dispatch L parallel Phase 2 paths via asyncio.gather (REQ-OR-013).

    Creates one ``run_phase2_outer_loop`` coroutine per path, each receiving
    the Phase 1 initial solution and score.  Uses ``return_exceptions=True``
    so that individual path failures do not cancel siblings (REQ-OR-022).

    Args:
        client: SDK client for agent invocations.
        task: Task description.
        config: Pipeline configuration (L = ``num_parallel_solutions``).
        phase1_result: Phase 1 output providing the initial solution.

    Returns:
        List of L results, each either a ``Phase2Result`` or an exception.
    """
    num_paths = config.num_parallel_solutions
    phase2_coros = [
        run_phase2_outer_loop(
            client,
            task,
            config,
            phase1_result.initial_solution,
            phase1_result.initial_score,
            session_id=f"path-{i}",
        )
        for i in range(num_paths)
    ]
    return await asyncio.gather(*phase2_coros, return_exceptions=True)


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
    and dispatches pipeline phases in sequence. The client is always
    disconnected on exit via try/finally.

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

    gpu_info = detect_gpu_info()
    system_prompt = _build_system_prompt(task, gpu_info)
    agents_dict = _build_agents_dict()

    # Convert plain dicts to AgentDefinition instances for the SDK
    agent_definitions: dict[str, AgentDefinition] = {
        name: AgentDefinition(**defn) for name, defn in agents_dict.items()
    }

    options = ClaudeAgentOptions(
        model=resolved_config.model,
        permission_mode=resolved_config.permission_mode,  # type: ignore[arg-type]
        max_budget_usd=resolved_config.max_budget_usd,
        agents=agent_definitions,
        system_prompt=system_prompt,
    )

    client = ClaudeSDKClient(options)

    try:
        await client.connect()

        try:
            _register_mcp_servers(client)
        except Exception:
            logger.warning("MCP server registration failed; continuing without MCP")

        setup_working_directory(task.data_dir)

        # --- Phase 1 (REQ-OR-012) ---
        logger.info("=== Phase 1: Initial Solution Generation ===")
        p1_start = time.monotonic()
        phase1_result = await run_phase1(task, resolved_config, client)
        p1_duration = time.monotonic() - p1_start
        logger.info("Phase 1 completed in %.1fs", p1_duration)

        # --- Phase 2 (REQ-OR-013, REQ-OR-017) ---
        logger.info("=== Phase 2: Targeted Refinement ===")
        p2_start = time.monotonic()
        raw_phase2 = await _dispatch_phase2(
            client, task, resolved_config, phase1_result
        )
        phase2_results, phase2_solutions = _collect_phase2_results(
            raw_phase2, phase1_result
        )
        p2_duration = time.monotonic() - p2_start
        logger.info("Phase 2 completed in %.1fs", p2_duration)

        # --- Phase 3 (REQ-OR-014, REQ-OR-015) ---
        phase3_result: Phase3Result | None = None
        best_solution: SolutionScript

        if resolved_config.num_parallel_solutions > 1:
            logger.info("=== Phase 3: Ensemble Construction ===")
            p3_start = time.monotonic()
            phase3_result = await run_phase3(
                client, task, resolved_config, phase2_solutions
            )
            p3_duration = time.monotonic() - p3_start
            logger.info("Phase 3 completed in %.1fs", p3_duration)
            best_solution = phase3_result.best_ensemble
        else:
            logger.info("Phase 3 skipped (L=1)")
            best_solution = phase2_solutions[0]

        # --- Finalization (REQ-OR-016) ---
        logger.info("=== Finalization ===")
        p_fin_start = time.monotonic()
        final_result = await run_finalization(
            client,
            best_solution,
            task,
            resolved_config,
            phase1_result,
            phase2_results,
            phase3_result,
        )
        p_fin_duration = time.monotonic() - p_fin_start
        logger.info("Finalization completed in %.1fs", p_fin_duration)

        return final_result
    finally:
        await client.disconnect()


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
