"""Safety modules: debugger, leakage, and data agents for solution integrity.

Implements:
- A_debugger: cross-cutting safety agent that fixes broken solution scripts
  by invoking the Claude Agent SDK with the error traceback and original code.
- A_leakage: two-step detection/correction agent that identifies and fixes
  data leakage in preprocessing code.
- A_data: data usage verification agent that ensures solutions incorporate
  all provided data sources.

Shared utility ``extract_code_block`` is used by A_debugger, A_leakage
correction, and A_data.

Refs:
    SRS 03a — Safety Debugger (REQ-SF-001 through REQ-SF-010).
    SRS 03b — Safety Leakage (REQ-SF-011 through REQ-SF-023).
    SRS 03c — Safety Data (REQ-SF-024 through REQ-SF-031).
    IMPLEMENTATION_PLAN.md Tasks 19, 20, 21.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from mle_star.execution import evaluate_solution
from mle_star.models import AgentType, LeakageDetectionOutput, SolutionScript
from mle_star.prompts import PromptRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mle_star.models import (
        EvaluationResult,
        PipelineConfig,
        TaskDescription,
    )

logger = logging.getLogger(__name__)

# Pattern for fenced code blocks (with optional language tag).
# Matches ```python\n...\n``` or ```\n...\n```.
_CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(
    r"```(?:\w+)?\s*\n(.*?)```",
    re.DOTALL,
)

_SCORE_LINE_MARKER = "Final Validation Performance"


def extract_code_block(response: str) -> str:
    """Extract a Python code block from an agent response (REQ-SF-005).

    Extraction rules:
    1. If fenced code blocks (````` ... `````) exist, return the **longest**
       one by character count.
    2. If no fenced code blocks exist, return the entire response stripped
       of leading/trailing whitespace.

    This utility is shared by A_debugger, A_leakage correction, and A_data.

    Args:
        response: Raw text response from an agent.

    Returns:
        Extracted code string.
    """
    matches = _CODE_FENCE_PATTERN.findall(response)
    if not matches:
        return response.strip()
    return max(matches, key=len).strip()


def _ensure_score_line(code: str) -> str:
    """Append the score print line if missing from debugged code (REQ-SF-010).

    If the code does not contain ``"Final Validation Performance"``,
    appends a print statement. When an ``if __name__`` block is present,
    the line is inserted before it; otherwise it is appended at the end.

    Args:
        code: Python source code to check and potentially modify.

    Returns:
        The code with a score print line guaranteed to be present.
    """
    if _SCORE_LINE_MARKER in code:
        return code

    logger.warning(
        "Debugged code missing '%s' print line; appending",
        _SCORE_LINE_MARKER,
    )
    score_line = '\nprint(f"Final Validation Performance: {final_validation_score}")\n'

    # Insert before `if __name__` block if present.
    main_idx = code.rfind("\nif __name__")
    if main_idx != -1:
        return code[:main_idx] + score_line + code[main_idx:]

    return code + score_line


async def _invoke_debugger_agent(
    solution: SolutionScript,
    traceback: str,
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> SolutionScript:
    """Invoke the debugger agent once and return the fixed solution.

    Renders the debugger prompt template from the ``PromptRegistry`` with
    the solution's source code and the error traceback, sends it to the
    A_debugger agent via the SDK client, extracts the code block from the
    response, and ensures the score print line is present.

    The SDK *client* must support::

        await client.send_message(
            agent_type=str,
            message=str,
        ) -> str

    Args:
        solution: The failing solution script.
        traceback: Error traceback from execution.
        task: Task description (unused directly but available for context).
        config: Pipeline configuration (unused directly but available for context).
        client: SDK client for agent invocation.

    Returns:
        A new ``SolutionScript`` with the debugger's fix applied, the same
        ``phase`` as the input, and ``is_executable=True``.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.DEBUGGER)
    prompt = template.render(code=solution.content, bug=traceback)

    response: str = await client.send_message(
        agent_type=str(AgentType.DEBUGGER),
        message=prompt,
    )

    code = extract_code_block(response)
    code = _ensure_score_line(code)

    return SolutionScript(
        content=code,
        phase=solution.phase,
        is_executable=True,
    )


async def debug_solution(
    solution: SolutionScript,
    traceback: str,
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> tuple[SolutionScript, EvaluationResult]:
    """Debug retry loop for fixing execution errors (REQ-SF-006).

    Invokes the A_debugger agent up to ``config.max_debug_attempts`` times.
    Each iteration: invoke debugger with current code and traceback, evaluate
    the fixed solution, check for success. Stops early on the first
    successful evaluation.

    Returns the **final** ``(SolutionScript, EvaluationResult)`` pair. This
    may still have ``is_error=True`` if all attempts failed. The **calling
    code** is responsible for maintaining a fallback reference to the last
    known working solution (REQ-SF-008).

    Args:
        solution: The failing solution script.
        traceback: Error traceback from the initial failure.
        task: Task description for evaluation context.
        config: Pipeline config (provides ``max_debug_attempts``).
        client: SDK client for agent invocation.

    Returns:
        A ``(SolutionScript, EvaluationResult)`` tuple.

    Raises:
        ValueError: If *traceback* is empty or ``None``.
    """
    if not traceback:
        msg = "No traceback provided for debugging"
        raise ValueError(msg)

    current_solution = solution
    current_traceback = traceback
    result: EvaluationResult | None = None

    for _ in range(config.max_debug_attempts):
        current_solution = await _invoke_debugger_agent(
            current_solution, current_traceback, task, config, client
        )
        result = await evaluate_solution(current_solution, task, config)

        if not result.is_error:
            break

        # Use the new traceback for the next attempt (if available).
        if result.error_traceback:
            current_traceback = result.error_traceback

    # max_debug_attempts >= 1 (enforced by PipelineConfig validator),
    # so the loop always runs at least once.
    if result is None:  # pragma: no cover
        msg = "No debug attempts were made"
        raise RuntimeError(msg)

    return current_solution, result


def make_debug_callback(
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> Callable[[SolutionScript, str | None], Awaitable[SolutionScript]]:
    """Factory returning an async callback for ``evaluate_with_retry`` (REQ-SF-007).

    The returned callback wraps a **single** debugger invocation (no retry
    loop). The retry loop is managed by ``evaluate_with_retry`` externally.

    When the traceback is ``None`` or empty, the callback returns the
    original solution unchanged (no debug invocation).

    Args:
        task: Task description for prompt context.
        config: Pipeline configuration.
        client: SDK client for agent invocation.

    Returns:
        An async callback with signature
        ``(SolutionScript, str | None) -> SolutionScript``.
    """

    async def _callback(
        solution: SolutionScript,
        traceback_str: str | None,
    ) -> SolutionScript:
        if not traceback_str:
            return solution
        return await _invoke_debugger_agent(
            solution, traceback_str, task, config, client
        )

    return _callback


async def check_and_fix_leakage(
    solution: SolutionScript,
    task: TaskDescription,
    client: Any,
) -> SolutionScript:
    """Detect and correct data leakage in a solution script (REQ-SF-020).

    Two-step pipeline:

    1. **Detection** — invoke A_leakage with ``variant="detection"`` and
       structured output ``LeakageDetectionOutput``.  Each answer indicates
       whether a preprocessing code block leaks validation/test data.
    2. **Correction** — for every ``LeakageAnswer`` with
       ``leakage_status == "Yes Data Leakage"``, invoke A_leakage with
       ``variant="correction"`` (free-form response), extract the corrected
       code via ``extract_code_block``, and apply it via
       ``SolutionScript.replace_block``.

    If ``replace_block`` raises ``ValueError`` (the detected block is not
    an exact substring of the solution), the replacement is logged as a
    warning and skipped (REQ-SF-021).

    Any exception from the SDK client or response parsing triggers graceful
    degradation: the **original** solution is returned unchanged.

    Args:
        solution: The solution to check for data leakage.
        task: Task description (unused directly but available for context).
        client: SDK client for agent invocation.

    Returns:
        The (potentially corrected) ``SolutionScript``.
    """
    try:
        return await _check_and_fix_leakage_impl(solution, task, client)
    except Exception:
        logger.exception("Leakage check failed; returning original solution")
        return solution


async def _check_and_fix_leakage_impl(
    solution: SolutionScript,
    task: TaskDescription,
    client: Any,
) -> SolutionScript:
    """Inner implementation of leakage detection and correction.

    Separated from ``check_and_fix_leakage`` so that the outer function
    provides a single top-level exception boundary for graceful degradation.

    Args:
        solution: The solution to check for data leakage.
        task: Task description (unused directly but available for context).
        client: SDK client for agent invocation.

    Returns:
        The (potentially corrected) ``SolutionScript``.
    """
    registry = PromptRegistry()

    # Step 1 — Detection.
    detection_template = registry.get(AgentType.LEAKAGE, variant="detection")
    detection_prompt = detection_template.render(code=solution.content)

    detection_response: str = await client.send_message(
        agent_type=str(AgentType.LEAKAGE),
        message=detection_prompt,
    )

    detection_output = LeakageDetectionOutput.model_validate_json(
        detection_response,
    )

    # Step 2 — Correction for each leaky block.
    current_solution = solution
    for answer in detection_output.answers:
        if answer.leakage_status != "Yes Data Leakage":
            continue

        correction_template = registry.get(AgentType.LEAKAGE, variant="correction")
        correction_prompt = correction_template.render(
            code=current_solution.content,
        )

        correction_response: str = await client.send_message(
            agent_type=str(AgentType.LEAKAGE),
            message=correction_prompt,
        )

        corrected_block = extract_code_block(correction_response)

        try:
            current_solution = current_solution.replace_block(
                answer.code_block, corrected_block
            )
        except ValueError:
            logger.warning(
                "Leaky code block not found in solution; skipping "
                "replacement for this answer (REQ-SF-021)",
            )

    return current_solution


# ---------------------------------------------------------------------------
# A_data — Data usage verification agent (REQ-SF-024 to REQ-SF-031)
# ---------------------------------------------------------------------------

_ALL_USED_PHRASE = "all the provided information is used."


def parse_data_agent_response(
    response: str,
    original_solution: SolutionScript,
) -> SolutionScript:
    """Parse the A_data agent response into a SolutionScript (REQ-SF-028).

    Two response formats are supported:

    1. **Confirmation** — the response contains the phrase
       ``"All the provided information is used."`` (case-insensitive).
       Returns ``original_solution`` unchanged.
    2. **Revised code** — otherwise, extract the code block via
       ``extract_code_block()`` and return a new ``SolutionScript`` with the
       extracted code, preserving the original's ``phase``.

    Args:
        response: Raw text response from the A_data agent.
        original_solution: The solution that was checked.

    Returns:
        The original or a new ``SolutionScript`` depending on the response.
    """
    if _ALL_USED_PHRASE in response.lower():
        return original_solution

    code = extract_code_block(response)
    return SolutionScript(
        content=code,
        phase=original_solution.phase,
    )


async def check_data_usage(
    solution: SolutionScript,
    task: TaskDescription,
    client: Any,
) -> SolutionScript:
    """Verify and fix data usage in a solution script (REQ-SF-026).

    Invokes the A_data agent with the solution code and task description.
    The agent checks whether all provided data sources are used and either
    confirms or returns corrected code incorporating unused information.

    This function runs **exactly once** per pipeline execution, after the
    initial solution is generated in Phase 1 (REQ-SF-030).

    On any exception (SDK failure, parsing error), gracefully degrades by
    returning the original solution unchanged.

    Args:
        solution: The initial solution to check for data utilization.
        task: Task description providing context about available data sources.
        client: SDK client for agent invocation.

    Returns:
        The (potentially corrected) ``SolutionScript``.
    """
    try:
        return await _check_data_usage_impl(solution, task, client)
    except Exception:
        logger.exception("Data usage check failed; returning original solution")
        return solution


async def _check_data_usage_impl(
    solution: SolutionScript,
    task: TaskDescription,
    client: Any,
) -> SolutionScript:
    """Inner implementation of data usage verification.

    Separated from ``check_data_usage`` so that the outer function provides
    a single top-level exception boundary for graceful degradation.

    Args:
        solution: The initial solution to check.
        task: Task description for prompt context.
        client: SDK client for agent invocation.

    Returns:
        The (potentially corrected) ``SolutionScript``.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.DATA)
    prompt = template.render(
        initial_solution=solution.content,
        task_description=task.description,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.DATA),
        message=prompt,
    )

    return parse_data_agent_response(response, solution)
