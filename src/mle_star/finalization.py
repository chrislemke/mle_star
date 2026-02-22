"""Finalization: subsampling removal pipeline.

Implements the subsampling removal step that runs after the main pipeline
phases complete.  Uses two sequential A_test agent invocations (extraction
then removal) to identify and strip subsampling code from the final solution
so the model trains on the full dataset.

Refs:
    SRS 08a — Finalization Subsampling (REQ-FN-001 through REQ-FN-009).
    SRS 08d — Finalization Constraints (REQ-FN-039, REQ-FN-044).
    IMPLEMENTATION_PLAN.md Task 38.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.models import AgentType, SolutionScript, TaskDescription
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
