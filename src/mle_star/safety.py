"""Safety modules: debugger agent for fixing execution errors.

Implements the A_debugger cross-cutting safety agent that fixes broken
solution scripts by invoking the Claude Agent SDK with the error traceback
and original code. Shared utility ``extract_code_block`` is also used by
A_leakage correction and A_data (implemented in later tasks).

Refs:
    SRS 03a — Safety Debugger (REQ-SF-001 through REQ-SF-010).
    IMPLEMENTATION_PLAN.md Task 19.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from mle_star.execution import evaluate_solution
from mle_star.models import AgentType, SolutionScript
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
