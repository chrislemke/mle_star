"""Phase 2 outer loop: ablation agent invocation and script execution.

Implements ``invoke_ablation``, ``compute_ablation_timeout``,
``execute_ablation_with_retry``, and ``_format_previous_ablations`` for the
Phase 2 outer loop ablation study pipeline (Algorithm 2, lines 5-7).

A_abl receives the current best solution and previous ablation summaries,
producing a self-contained ablation study script that tests 2-3 code
components.  The script is executed with a capped timeout and debug retry
on failure.  Raw output feeds into A_summarize (Task 32).

Refs:
    SRS 05a — Phase 2 Outer Ablation (REQ-P2O-001 through REQ-P2O-007).
    SRS 05c — Ablation Execution (REQ-P2O-020, REQ-P2O-021).
    SRS 05d — Ablation Timeout (REQ-P2O-035).
    IMPLEMENTATION_PLAN.md Task 31.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.execution import (
    build_execution_env,
    execute_script,
    extract_traceback,
    setup_working_directory,
    write_script,
)
from mle_star.models import (
    AgentType,
    PipelineConfig,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
)
from mle_star.prompts import PromptRegistry
from mle_star.safety import extract_code_block, make_debug_callback

logger = logging.getLogger(__name__)

# Maximum ablation script timeout in seconds (REQ-P2O-035).
_ABLATION_TIMEOUT_CAP: int = 600


# ---------------------------------------------------------------------------
# Ablation summary formatting
# ---------------------------------------------------------------------------


def _format_previous_ablations(summaries: list[str]) -> str:
    """Format previous ablation summaries for the A_abl prompt template.

    When *summaries* is empty, returns an empty string so the prompt
    template omits the section entirely (REQ-P2O-002).  When non-empty,
    formats each summary with a numbered header.

    Args:
        summaries: Previous ablation summary texts (T_abl^0 … T_abl^{t-1}).

    Returns:
        Formatted text block or empty string.
    """
    if not summaries:
        return ""

    lines: list[str] = ["# Previous Ablation Study Results"]
    for i, summary in enumerate(summaries, start=1):
        lines.append(f"\n## Ablation Study {i}\n{summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A_abl — Ablation agent invocation (REQ-P2O-001 to REQ-P2O-007)
# ---------------------------------------------------------------------------


async def invoke_ablation(
    solution: SolutionScript,
    previous_summaries: list[str],
    client: Any,
) -> SolutionScript | None:
    """Invoke A_abl to generate an ablation study script (REQ-P2O-003).

    Renders the ablation prompt template with the current solution and
    formatted previous ablation summaries, sends it to the A_abl agent
    via the SDK client, and extracts the code block from the response.

    Args:
        solution: Current best solution (s_t) whose components to ablate.
        previous_summaries: Summaries from prior outer steps (T_abl).
        client: SDK client for agent invocation.

    Returns:
        A ``SolutionScript`` wrapping the ablation study code with
        ``phase=REFINED`` and ``is_executable=True``, or ``None`` if
        the agent response is empty or extraction yields no code.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.ABLATION)

    previous_text = _format_previous_ablations(previous_summaries)

    prompt = template.render(
        solution_script=solution.content,
        previous_ablations=previous_text,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.ABLATION),
        message=prompt,
    )

    extracted = extract_code_block(response)
    if not extracted.strip():
        logger.warning("A_abl returned empty code; treating as failure")
        return None

    return SolutionScript(
        content=extracted,
        phase=SolutionPhase.REFINED,
        is_executable=True,
    )


# ---------------------------------------------------------------------------
# Ablation timeout computation (REQ-P2O-035)
# ---------------------------------------------------------------------------


def compute_ablation_timeout(config: PipelineConfig) -> int:
    """Compute the execution timeout for ablation scripts (REQ-P2O-035).

    Formula: ``min(time_limit_seconds // (outer_loop_steps * 2), 600)``.
    Caps at 600 seconds to prevent a single ablation study from consuming
    an excessive portion of the total time budget.

    Args:
        config: Pipeline configuration providing time limit and outer steps.

    Returns:
        Timeout in seconds (positive integer, at most 600).
    """
    return min(
        config.time_limit_seconds // (config.outer_loop_steps * 2),
        _ABLATION_TIMEOUT_CAP,
    )


# ---------------------------------------------------------------------------
# Ablation script execution with debug retry (REQ-P2O-020, REQ-P2O-021)
# ---------------------------------------------------------------------------


def _is_execution_error(exit_code: int, timed_out: bool, stderr: str) -> bool:
    """Check whether a raw execution result represents an error.

    Args:
        exit_code: Process exit code (0 = success, -1 = timeout).
        timed_out: Whether the process was killed due to timeout.
        stderr: Standard error output from the process.

    Returns:
        ``True`` if the execution should be considered a failure.
    """
    if exit_code != 0:
        return True
    if timed_out:
        return True
    return "Traceback (most recent call last):" in stderr


async def execute_ablation_with_retry(
    ablation_script: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> tuple[str, str]:
    """Execute an ablation study script with debug retry on error (REQ-P2O-020).

    Writes the ablation script to disk, executes it with the ablation
    timeout (REQ-P2O-035), and on failure invokes the debugger agent up
    to ``config.max_debug_attempts`` times (REQ-P2O-021).  If all
    attempts are exhausted, returns empty strings.

    Args:
        ablation_script: The ablation study script to execute.
        task: Task description providing ``data_dir``.
        config: Pipeline configuration (timeout, debug attempts).
        client: SDK client for debugger agent invocation.

    Returns:
        A ``(stdout, stderr)`` tuple.  Returns ``("", "")`` if the
        script fails after all debug attempts.
    """
    timeout = compute_ablation_timeout(config)
    working_dir = setup_working_directory(task.data_dir)
    env = build_execution_env()

    current_script = ablation_script
    script_path = write_script(current_script, working_dir, "ablation_study.py")
    raw = await execute_script(script_path, working_dir, timeout, env)

    if not _is_execution_error(raw.exit_code, raw.timed_out, raw.stderr):
        return raw.stdout, raw.stderr

    # Error path — set up debug callback and retry loop (REQ-P2O-021).
    logger.warning(
        "Ablation script failed (exit_code=%d, timed_out=%s); entering debug retry",
        raw.exit_code,
        raw.timed_out,
    )
    debug_cb = make_debug_callback(task, config, client)

    for attempt in range(config.max_debug_attempts):
        tb = extract_traceback(raw.stderr) or raw.stderr[:500]
        current_script = await debug_cb(current_script, tb)

        script_path = write_script(current_script, working_dir, "ablation_study.py")
        raw = await execute_script(script_path, working_dir, timeout, env)

        if not _is_execution_error(raw.exit_code, raw.timed_out, raw.stderr):
            logger.info(
                "Ablation script fixed on debug attempt %d/%d",
                attempt + 1,
                config.max_debug_attempts,
            )
            return raw.stdout, raw.stderr

    logger.warning(
        "Ablation script failed after %d debug attempts; returning empty output",
        config.max_debug_attempts,
    )
    return "", ""
