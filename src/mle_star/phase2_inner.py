"""Phase 2 inner loop: coder and planner agent invocations and orchestration.

Implements ``invoke_coder`` and ``invoke_planner`` for the targeted code
block refinement inner loop (Algorithm 2, lines 9-25).  A_coder receives
a code block and a refinement plan, returning improved code.  A_planner
receives a code block and the history of prior plans/scores, returning a
new refinement strategy.

``run_phase2_inner_loop`` orchestrates K iterations of the inner loop:
k=0 uses the initial plan from A_extractor (no planner); k>=1 invokes
A_planner with accumulated history.  Each iteration calls A_coder with
the ORIGINAL code block, replaces against the ORIGINAL solution, evaluates,
and tracks the best score using >= semantics.

Refs:
    SRS 06a — Phase 2 Inner Agents (REQ-P2I-001 through REQ-P2I-015).
    SRS 06b — Phase 2 Inner Orchestration (REQ-P2I-016 through REQ-P2I-029).
    IMPLEMENTATION_PLAN.md Tasks 23, 24.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.execution import evaluate_solution
from mle_star.models import (
    AgentType,
    CodeBlock,
    InnerLoopResult,
    PipelineConfig,
    RefinementAttempt,
    SolutionScript,
    TaskDescription,
)
from mle_star.prompts import PromptRegistry
from mle_star.safety import extract_code_block
from mle_star.scoring import is_improvement, is_improvement_or_equal

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


async def run_phase2_inner_loop(
    client: Any,
    solution: SolutionScript,
    code_block: CodeBlock,
    initial_plan: str,
    best_score: float,
    task: TaskDescription,
    config: PipelineConfig,
) -> InnerLoopResult:
    """Execute K inner-loop iterations for targeted code block refinement (REQ-P2I-016).

    Implements Algorithm 2 lines 9-25.  For each iteration k:

    - **k=0**: Uses ``initial_plan`` directly (no A_planner call, REQ-P2I-018).
    - **k>=1**: Invokes A_planner with full accumulated history (REQ-P2I-020).

    A_coder always receives the **original** ``code_block.content`` (REQ-P2I-021).
    ``replace_block`` is always called against the **original** ``solution``
    (REQ-P2I-022/023).  Best score is updated using ``is_improvement_or_equal``
    (>= semantics, REQ-P2I-026).  ``InnerLoopResult.improved`` uses strict
    ``is_improvement`` (REQ-P2I-036).

    Args:
        client: SDK client for agent invocation.
        solution: Current best solution s_t from the outer loop.
        code_block: Target code block c_t to refine.
        initial_plan: Initial refinement plan p_0 from A_extractor.
        best_score: Current h_best from the outer loop.
        task: Task description for evaluation context.
        config: Pipeline configuration (provides ``inner_loop_steps`` = K).

    Returns:
        ``InnerLoopResult`` with the best solution, best score, all K
        ``RefinementAttempt`` records, and the ``improved`` flag.
    """
    k_steps: int = config.inner_loop_steps
    original_code = code_block.content

    # REQ-P2I-024: Initialize tracking from input parameters.
    local_best_score = best_score
    local_best_solution = solution

    accumulated_plans: list[str] = []
    accumulated_scores: list[float | None] = []
    attempts: list[RefinementAttempt] = []

    for k in range(k_steps):
        # ------------------------------------------------------------------
        # Step 1: Determine the plan for this iteration.
        # ------------------------------------------------------------------
        if k == 0:
            # REQ-P2I-018: Use initial_plan directly, no planner call.
            plan = initial_plan
        else:
            # REQ-P2I-019/020: Invoke A_planner with full history.
            # Pass copies so the planner receives a snapshot of history at
            # the time of invocation, not a reference that grows later.
            plan_result = await invoke_planner(
                original_code, list(accumulated_plans), list(accumulated_scores), client
            )
            if plan_result is None:
                # REQ-P2I-034: Planner failure → record and skip.
                logger.warning("A_planner returned None at k=%d; skipping", k)
                failed_plan = "[planner failed]"
                accumulated_plans.append(failed_plan)
                accumulated_scores.append(None)
                attempts.append(
                    RefinementAttempt(
                        plan=failed_plan,
                        score=None,
                        code_block="",
                        was_improvement=False,
                    )
                )
                continue
            plan = plan_result

        # ------------------------------------------------------------------
        # Step 2: Invoke A_coder with ORIGINAL code block (REQ-P2I-021).
        # ------------------------------------------------------------------
        coder_output = await invoke_coder(original_code, plan, client)
        if coder_output is None:
            # REQ-P2I-032: Coder failure → record with code_block="" and skip eval.
            logger.warning("A_coder returned None at k=%d; skipping eval", k)
            accumulated_plans.append(plan)
            accumulated_scores.append(None)
            attempts.append(
                RefinementAttempt(
                    plan=plan,
                    score=None,
                    code_block="",
                    was_improvement=False,
                )
            )
            continue

        # ------------------------------------------------------------------
        # Step 3: Replace block in ORIGINAL solution (REQ-P2I-022/023).
        # ------------------------------------------------------------------
        try:
            candidate = solution.replace_block(original_code, coder_output)
        except ValueError:
            # REQ-P2I-033: Replacement failure → record with coder output.
            logger.warning("replace_block failed at k=%d; code block not found", k)
            accumulated_plans.append(plan)
            accumulated_scores.append(None)
            attempts.append(
                RefinementAttempt(
                    plan=plan,
                    score=None,
                    code_block=coder_output,
                    was_improvement=False,
                )
            )
            continue

        # ------------------------------------------------------------------
        # Step 4: Evaluate the candidate solution.
        # ------------------------------------------------------------------
        eval_result = await evaluate_solution(candidate, task, config)
        score = eval_result.score

        # ------------------------------------------------------------------
        # Step 5: Track score and update best (REQ-P2I-025/026/027).
        # ------------------------------------------------------------------
        was_improvement = False
        if score is not None and is_improvement_or_equal(
            score, local_best_score, task.metric_direction
        ):
            local_best_score = score
            local_best_solution = candidate
            was_improvement = True

        accumulated_plans.append(plan)
        accumulated_scores.append(score)
        attempts.append(
            RefinementAttempt(
                plan=plan,
                score=score,
                code_block=coder_output,
                was_improvement=was_improvement,
            )
        )

    # REQ-P2I-036/037: Construct result. improved uses strict is_improvement.
    improved = is_improvement(local_best_score, best_score, task.metric_direction)

    return InnerLoopResult(
        best_solution=local_best_solution,
        best_score=local_best_score,
        attempts=attempts,
        improved=improved,
    )
