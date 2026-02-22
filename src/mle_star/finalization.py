"""Finalization: subsampling removal and test submission generation.

Implements the subsampling removal step and the A_test agent invocation
that runs after the main pipeline phases complete.  Uses agent invocations
to strip subsampling code and generate a test submission script.

Refs:
    SRS 08a — Finalization Subsampling (REQ-FN-001 through REQ-FN-009).
    SRS 08b — Finalization Test Submission (REQ-FN-010 through REQ-FN-025).
    SRS 08d — Finalization Constraints (REQ-FN-039, REQ-FN-040, REQ-FN-044).
    IMPLEMENTATION_PLAN.md Tasks 38, 39.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.models import AgentType, SolutionPhase, SolutionScript, TaskDescription
from mle_star.prompts import PromptRegistry
from mle_star.safety import extract_code_block

logger = logging.getLogger(__name__)


async def remove_subsampling(
    client: Any,
    solution: SolutionScript,
    task: TaskDescription,
) -> SolutionScript:
    """Remove subsampling code from a solution script (REQ-FN-009).

    Two-step agent pipeline:

    1. **Extraction** — invoke A_test with ``variant="subsampling_extract"``
       to identify the subsampling code block in the solution.
    2. **Removal** — invoke A_test with ``variant="subsampling_remove"``
       to generate a replacement block without subsampling.

    If the extraction step finds no subsampling (empty block, whitespace-only,
    or block not a substring of the solution), the original solution is
    returned unchanged (REQ-FN-008).

    On any exception (SDK failure, parsing error), gracefully degrades by
    returning the original solution unchanged (REQ-FN-039).

    Args:
        client: SDK client for agent invocation.
        solution: The final solution from which to remove subsampling.
        task: Task description for prompt context.

    Returns:
        The (potentially updated) ``SolutionScript`` with subsampling removed.
    """
    try:
        return await _remove_subsampling_impl(client, solution, task)
    except Exception:
        logger.exception("Subsampling removal failed; returning original solution")
        return solution


async def _remove_subsampling_impl(
    client: Any,
    solution: SolutionScript,
    task: TaskDescription,
) -> SolutionScript:
    """Inner implementation of the subsampling removal pipeline.

    Separated from ``remove_subsampling`` so the outer function provides a
    single top-level exception boundary for graceful degradation.

    Args:
        client: SDK client for agent invocation.
        solution: The final solution from which to remove subsampling.
        task: Task description for prompt context.

    Returns:
        The (potentially updated) ``SolutionScript``.
    """
    registry = PromptRegistry()
    agent_type_str = str(AgentType.TEST)

    # ------------------------------------------------------------------
    # Step 1 — Extract the subsampling code block (REQ-FN-001).
    # ------------------------------------------------------------------
    extraction_template = registry.get(AgentType.TEST, variant="subsampling_extract")
    extraction_prompt = extraction_template.render(
        final_solution=solution.content,
    )

    extraction_response: str = await client.send_message(
        agent_type=agent_type_str,
        message=extraction_prompt,
    )

    extracted_block = extract_code_block(extraction_response)

    # ------------------------------------------------------------------
    # Step 2 — Verify extraction is valid (REQ-FN-003, REQ-FN-008).
    # ------------------------------------------------------------------
    if not extracted_block.strip():
        logger.info("No subsampling code detected (empty extraction)")
        return solution

    if extracted_block not in solution.content:
        logger.info(
            "Extracted block not found in solution content; treating as no subsampling"
        )
        return solution

    # ------------------------------------------------------------------
    # Step 3 — Remove the subsampling from the code block (REQ-FN-004).
    # ------------------------------------------------------------------
    removal_template = registry.get(AgentType.TEST, variant="subsampling_remove")
    removal_prompt = removal_template.render(
        code_block_with_subsampling=extracted_block,
    )

    removal_response: str = await client.send_message(
        agent_type=agent_type_str,
        message=removal_prompt,
    )

    replacement_block = extract_code_block(removal_response)

    # ------------------------------------------------------------------
    # Step 4 — Replace in solution (REQ-FN-007).
    # ------------------------------------------------------------------
    try:
        return solution.replace_block(extracted_block, replacement_block)
    except ValueError:
        logger.warning(
            "replace_block failed: extracted subsampling block not found "
            "in solution; returning original"
        )
        return solution


async def generate_test_submission(
    client: Any,
    task: TaskDescription,
    solution: SolutionScript,
) -> SolutionScript:
    """Invoke A_test to transform a validation solution into a test submission (REQ-FN-019).

    Renders the A_test prompt template (Figure 25) with the task description
    and final solution, sends it to the A_test agent, and parses the response
    into a new ``SolutionScript`` with ``phase=SolutionPhase.FINAL``.

    The A_test agent is instructed to minimally modify the solution to load
    test data, predict for all test samples, and write
    ``./final/submission.csv`` (REQ-FN-014 through REQ-FN-018).

    Exceptions from the SDK client propagate to the caller; the calling
    orchestration code (``run_finalization``) is responsible for fallback
    handling (REQ-FN-025).

    Args:
        client: SDK client for agent invocation.
        task: Task description providing competition context.
        solution: The solution with subsampling removed.

    Returns:
        A new ``SolutionScript`` with ``phase=SolutionPhase.FINAL`` and
        ``is_executable=True``.  Content is the extracted code from the
        agent response, or empty string on empty extraction (REQ-FN-040).
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.TEST)

    prompt = template.render(
        task_description=task.description,
        final_solution=solution.content,
    )

    logger.info(
        "Invoking A_test for test submission: competition=%s, solution_len=%d",
        task.competition_id,
        len(solution.content),
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.TEST),
        message=prompt,
    )

    code = extract_code_block(response)

    if not code.strip():
        logger.warning(
            "A_test returned empty code block; response[:200]=%r",
            response[:200],
        )
        return SolutionScript(
            content="",
            phase=SolutionPhase.FINAL,
            is_executable=True,
        )

    logger.info("A_test generated test submission script: len=%d", len(code))

    return SolutionScript(
        content=code,
        phase=SolutionPhase.FINAL,
        is_executable=True,
    )
