"""Phase 3: ensemble planner and ensembler agent invocations.

Implements ``invoke_ens_planner`` and ``invoke_ensembler`` for Algorithm 3
(ensemble construction).  A_ens_planner proposes ensemble strategies from L
solution scripts and accumulated history.  A_ensembler implements a given
plan into a single self-contained Python script.

Helper functions ``_format_solutions`` and ``_format_ensemble_history``
produce the formatted text blocks inserted into prompt templates.

Refs:
    SRS 07a — Phase 3 Agents (REQ-P3-001 through REQ-P3-016).
    IMPLEMENTATION_PLAN.md Task 35.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.models import (
    AgentType,
    SolutionPhase,
    SolutionScript,
)
from mle_star.prompts import PromptRegistry
from mle_star.safety import extract_code_block

logger = logging.getLogger(__name__)


def _format_solutions(solutions: list[SolutionScript]) -> str:
    """Format solution scripts as numbered sections for prompt templates.

    Each solution is presented as::

        # {n}th Python Solution
        ```
        {content}
        ```

    Solutions are numbered starting from 1.

    Args:
        solutions: List of solution scripts to format.

    Returns:
        Formatted string with all solutions as numbered, fenced sections.
    """
    sections: list[str] = []
    for i, sol in enumerate(solutions, start=1):
        sections.append(f"# {i}th Python Solution\n```\n{sol.content}\n```")
    return "\n\n".join(sections)


def _format_ensemble_history(
    plans: list[str],
    scores: list[float | None],
) -> str:
    """Format plan/score history for the A_ens_planner prompt (REQ-P3-005).

    Each plan-score pair is rendered as::

        ## Plan: <plan text>
        ## Score: <score or "N/A (evaluation failed)">

    Pairs are separated by blank lines and presented in order (r=0 first).
    Returns an empty string when no history exists.

    Args:
        plans: Previous ensemble plan texts.
        scores: Corresponding scores (``None`` for failed evaluations).

    Returns:
        Formatted history string, or empty string if no history.
    """
    if not plans:
        return ""

    entries: list[str] = []
    for plan, score in zip(plans, scores, strict=True):
        score_str = "N/A (evaluation failed)" if score is None else str(score)
        entries.append(f"## Plan: {plan}\n## Score: {score_str}")
    return "\n\n".join(entries)


async def invoke_ens_planner(
    solutions: list[SolutionScript],
    plans: list[str],
    scores: list[float | None],
    client: Any,
) -> str | None:
    """Invoke A_ens_planner to propose an ensemble strategy (REQ-P3-009).

    Validates inputs, formats the solution scripts and plan/score history,
    renders the ens_planner prompt template, and invokes the agent.  Returns
    the agent's full text response (no code block extraction).

    On the first invocation (``plans`` is empty), the history section is
    empty (REQ-P3-004).  On subsequent invocations, the full history of
    previous plans and scores is included (REQ-P3-005).

    Args:
        solutions: L solution scripts to ensemble (must be >= 2).
        plans: Previous ensemble plan texts (empty on first call).
        scores: Corresponding scores; ``None`` for failed evaluations.
        client: SDK client for agent invocation.

    Returns:
        The ensemble plan string, or ``None`` if the response is empty.

    Raises:
        ValueError: If fewer than 2 solutions, or plans/scores length mismatch.
    """
    if len(solutions) < 2:
        msg = "A_ens_planner requires at least 2 solutions for ensembling"
        raise ValueError(msg)
    if len(plans) != len(scores):
        msg = f"plans and scores must have equal length (got {len(plans)} vs {len(scores)})"
        raise ValueError(msg)

    solutions_text = _format_solutions(solutions)
    plan_history = _format_ensemble_history(plans, scores)

    registry = PromptRegistry()
    template = registry.get(AgentType.ENS_PLANNER)
    prompt = template.render(
        L=len(solutions),
        solutions_text=solutions_text,
        plan_history=plan_history,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.ENS_PLANNER),
        message=prompt,
    )

    stripped = response.strip()
    if not stripped:
        logger.warning("A_ens_planner returned empty response; treating as failure")
        return None

    return stripped


async def invoke_ensembler(
    plan: str,
    solutions: list[SolutionScript],
    client: Any,
) -> SolutionScript | None:
    """Invoke A_ensembler to implement an ensemble plan (REQ-P3-016).

    Validates inputs, formats the solution scripts, renders the ensembler
    prompt template, invokes the agent, and extracts the code block from
    the response.  Returns a ``SolutionScript`` with
    ``phase=SolutionPhase.ENSEMBLE``.

    Args:
        plan: The ensemble plan to implement (must be non-empty).
        solutions: L solution scripts to ensemble (must be >= 2).
        client: SDK client for agent invocation.

    Returns:
        A ``SolutionScript`` with the ensemble code, or ``None`` if the
        agent response is empty or extraction yields no code.

    Raises:
        ValueError: If *plan* is empty/whitespace or fewer than 2 solutions.
    """
    if not plan.strip():
        msg = "A_ensembler requires a non-empty ensemble plan"
        raise ValueError(msg)
    if len(solutions) < 2:
        msg = "A_ensembler requires at least 2 solutions for ensembling"
        raise ValueError(msg)

    solutions_text = _format_solutions(solutions)

    registry = PromptRegistry()
    template = registry.get(AgentType.ENSEMBLER)
    prompt = template.render(
        L=len(solutions),
        solutions_text=solutions_text,
        plan=plan,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.ENSEMBLER),
        message=prompt,
    )

    extracted = extract_code_block(response)
    if not extracted.strip():
        logger.warning("A_ensembler returned empty code; treating as failure")
        return None

    return SolutionScript(
        content=extracted,
        phase=SolutionPhase.ENSEMBLE,
    )
