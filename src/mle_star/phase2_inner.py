"""Phase 2 inner loop: coder and planner agent invocations.

Implements ``invoke_coder`` and ``invoke_planner`` for the targeted code
block refinement inner loop (Algorithm 2, lines 9-25).  A_coder receives
a code block and a refinement plan, returning improved code.  A_planner
receives a code block and the history of prior plans/scores, returning a
new refinement strategy.

Refs:
    SRS 06a — Phase 2 Inner Agents (REQ-P2I-001 through REQ-P2I-015).
    IMPLEMENTATION_PLAN.md Task 23.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.models import AgentType
from mle_star.prompts import PromptRegistry
from mle_star.safety import extract_code_block

logger = logging.getLogger(__name__)


def _format_plan_history(
    plans: list[str],
    scores: list[float | None],
) -> str:
    """Format plan/score history for the A_planner prompt (REQ-P2I-010).

    Each plan-score pair is rendered as::

        ## Plan: <plan text>
        ## Score: <score or "N/A (evaluation failed)">

    Pairs are separated by blank lines and presented in order (k=0 first).

    Args:
        plans: Previous refinement plan texts.
        scores: Corresponding scores (``None`` for failed evaluations).

    Returns:
        Formatted history string.
    """
    entries: list[str] = []
    for plan, score in zip(plans, scores, strict=True):
        score_str = "N/A (evaluation failed)" if score is None else str(score)
        entries.append(f"## Plan: {plan}\n## Score: {score_str}")
    return "\n\n".join(entries)


async def invoke_coder(
    code_block: str,
    plan: str,
    client: Any,
) -> str | None:
    """Invoke A_coder to implement a refinement plan on a code block (REQ-P2I-005).

    Renders the coder prompt template with the target code block and plan,
    sends it to the A_coder agent via the SDK client, and extracts the
    improved code block from the response.

    Args:
        code_block: The target code block content to be improved (c_t).
        plan: The natural language refinement plan to implement (p_k).
        client: SDK client for agent invocation.

    Returns:
        The extracted improved code block string, or ``None`` if the
        response is empty or extraction yields an empty result.

    Raises:
        ValueError: If *code_block* or *plan* is empty/whitespace-only.
    """
    if not code_block.strip():
        msg = "code_block must not be empty"
        raise ValueError(msg)
    if not plan.strip():
        msg = "plan must not be empty"
        raise ValueError(msg)

    registry = PromptRegistry()
    template = registry.get(AgentType.CODER)
    prompt = template.render(code_block=code_block, plan=plan)

    response: str = await client.send_message(
        agent_type=str(AgentType.CODER),
        message=prompt,
    )

    extracted = extract_code_block(response)
    if not extracted:
        logger.warning("A_coder returned empty code; treating as failure")
        return None

    return extracted


async def invoke_planner(
    code_block: str,
    plans: list[str],
    scores: list[float | None],
    client: Any,
) -> str | None:
    """Invoke A_planner to propose a new refinement strategy (REQ-P2I-013).

    Validates inputs, formats the history of prior plans and scores,
    renders the planner prompt template, and invokes the A_planner agent.
    Returns the agent's full text response (no code block extraction).

    Args:
        code_block: The original target code block (c_t).
        plans: List of previous refinement plan texts (p_0 … p_{k-1}).
        scores: Corresponding scores; ``None`` for failed evaluations.
        client: SDK client for agent invocation.

    Returns:
        The refinement plan string, or ``None`` if the response is empty.

    Raises:
        ValueError: If *plans* is empty, *plans*/*scores* lengths differ,
            or *code_block* is empty/whitespace-only.
    """
    if not plans:
        msg = "At least one previous plan is required for A_planner"
        raise ValueError(msg)
    if len(plans) != len(scores):
        msg = f"plans and scores must have equal length (got {len(plans)} vs {len(scores)})"
        raise ValueError(msg)
    if not code_block.strip():
        msg = "code_block must not be empty"
        raise ValueError(msg)

    plan_history = _format_plan_history(plans, scores)

    registry = PromptRegistry()
    template = registry.get(AgentType.PLANNER)
    prompt = template.render(code_block=code_block, plan_history=plan_history)

    response: str = await client.send_message(
        agent_type=str(AgentType.PLANNER),
        message=prompt,
    )

    stripped = response.strip()
    if not stripped:
        logger.warning("A_planner returned empty response; treating as failure")
        return None

    return stripped
