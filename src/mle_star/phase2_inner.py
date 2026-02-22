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
    SRS 06b — Phase 2 Inner Orchestration (REQ-P2I-016 through REQ-P2I-038).
    IMPLEMENTATION_PLAN.md Tasks 23, 24, 25.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.execution import evaluate_with_retry
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
from mle_star.safety import (
    check_and_fix_leakage,
    extract_code_block,
    make_debug_callback,
)
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


async def _execute_coder_step(
    k: int,
    plan: str,
    original_code: str,
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> dict[str, Any]:
    """Execute the coder → replace → leakage → eval portion of one inner step.

    Returns a dict with keys: ``plan``, ``score``, ``code_block``,
    ``was_improvement`` (always ``False``), ``_candidate``,
    ``_eval_result``, and ``_successful``.  The caller uses
    ``_candidate`` and ``_eval_result`` for best-score tracking.

    If the coder fails, replacement fails, or evaluation fails, a
    partial result is returned with ``_successful=False``.

    Args:
        k: Inner step index.
        plan: Refinement plan for this step.
        original_code: Original code block content (c_t).
        solution: Original solution (s_t) used as replacement base.
        task: Task description for evaluation context.
        config: Pipeline configuration.
        client: SDK client for agent invocation.

    Returns:
        Dict with step results and internal tracking keys.
    """
    logger.info("A_coder invocation start: k=%d, plan=%.200s", k, plan)
    coder_output = await invoke_coder(original_code, plan, client)
    if coder_output is None:
        logger.info("A_coder invocation complete: k=%d, result=failed to parse", k)
        logger.warning("A_coder returned None at k=%d; skipping eval", k)
        return {
            "plan": plan,
            "score": None,
            "code_block": "",
            "was_improvement": False,
            "_successful": False,
        }
    logger.info("A_coder invocation complete: k=%d, code_len=%d", k, len(coder_output))

    try:
        candidate = solution.replace_block(original_code, coder_output)
    except ValueError:
        logger.warning("replace_block failed at k=%d; code block not found", k)
        return {
            "plan": plan,
            "score": None,
            "code_block": coder_output,
            "was_improvement": False,
            "_successful": False,
        }
    logger.debug(
        "Code block replacement success: k=%d, old_len=%d, new_len=%d",
        k,
        len(original_code),
        len(coder_output),
    )

    # Leakage check (REQ-P2I-030).
    logger.info("Leakage check start: k=%d, content_len=%d", k, len(candidate.content))
    pre_leakage = candidate
    candidate = await check_and_fix_leakage(candidate, task, client)
    content_changed = candidate is not pre_leakage
    logger.info(
        "Leakage check complete: k=%d, leakage_found=%s, content_changed=%s",
        k,
        "yes" if content_changed else "no",
        "yes" if content_changed else "no",
    )

    # Evaluate with debug retry (REQ-P2I-031).
    logger.info("Evaluation start: k=%d, content_len=%d", k, len(candidate.content))
    debug_callback = make_debug_callback(task, config, client)
    candidate, eval_result = await evaluate_with_retry(
        candidate, task, config, debug_callback
    )
    score = eval_result.score
    score_str = str(score) if score is not None else "failed"
    logger.info(
        "Evaluation complete: k=%d, score=%s, is_error=%s, duration=%.1f",
        k,
        score_str,
        eval_result.is_error,
        eval_result.duration_seconds,
    )

    return {
        "plan": plan,
        "score": score,
        "code_block": coder_output,
        "was_improvement": False,
        "_candidate": candidate,
        "_eval_result": eval_result,
        "_successful": True,
    }


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
    (REQ-P2I-022/023).  Before each evaluation, ``check_and_fix_leakage`` is
    called on the candidate (REQ-P2I-030/REQ-SF-022).  Evaluation uses
    ``evaluate_with_retry`` with ``make_debug_callback`` for error recovery
    (REQ-P2I-031).  Best score is updated using ``is_improvement_or_equal``
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

    # REQ-P2I-043: Log inner loop start.
    logger.info(
        "Inner loop start: code_block_len=%d, initial_plan=%.200s, h_best=%s, K=%d",
        len(original_code),
        initial_plan,
        best_score,
        k_steps,
    )

    # REQ-P2I-024: Initialize tracking from input parameters.
    local_best_score = best_score
    local_best_solution = solution

    accumulated_plans: list[str] = []
    accumulated_scores: list[float | None] = []
    attempts: list[RefinementAttempt] = []
    successful_evals = 0

    for k in range(k_steps):
        step_result = await _run_inner_step(
            k,
            initial_plan,
            original_code,
            solution,
            accumulated_plans,
            accumulated_scores,
            task,
            config,
            client,
        )

        accumulated_plans.append(step_result["plan"])
        accumulated_scores.append(step_result["score"])

        # Update best score if the step was successful (REQ-P2I-025/026/027).
        score = step_result["score"]
        was_improvement = False
        if (
            score is not None
            and step_result.get("_successful")
            and is_improvement_or_equal(score, local_best_score, task.metric_direction)
        ):
            old_best = local_best_score
            local_best_score = score
            local_best_solution = step_result["_candidate"]
            was_improvement = True
            logger.info("Best score updated: k=%d, old=%s, new=%s", k, old_best, score)

        if score is not None and step_result.get("_successful"):
            successful_evals += 1

        attempts.append(
            RefinementAttempt(
                plan=step_result["plan"],
                score=step_result["score"],
                code_block=step_result["code_block"],
                was_improvement=was_improvement,
            )
        )

    # REQ-P2I-036/037: Construct result. improved uses strict is_improvement.
    improved = is_improvement(local_best_score, best_score, task.metric_direction)

    # REQ-P2I-043: Log inner loop completion.
    logger.info(
        "Inner loop complete: attempts=%d, successful_evals=%d, best_score=%s, improved=%s",
        len(attempts),
        successful_evals,
        local_best_score,
        "yes" if improved else "no",
    )

    return InnerLoopResult(
        best_solution=local_best_solution,
        best_score=local_best_score,
        attempts=attempts,
        improved=improved,
    )


async def _run_inner_step(
    k: int,
    initial_plan: str,
    original_code: str,
    solution: SolutionScript,
    accumulated_plans: list[str],
    accumulated_scores: list[float | None],
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> dict[str, Any]:
    """Execute one inner loop step: plan determination + coder step.

    Handles the plan source selection (initial_plan at k=0, A_planner at
    k>=1) and delegates the coder → replace → leakage → eval chain to
    ``_execute_coder_step``.

    Args:
        k: Inner step index.
        initial_plan: Initial plan from A_extractor (used at k=0).
        original_code: Original code block content (c_t).
        solution: Original solution (s_t) used as replacement base.
        accumulated_plans: Plans from previous steps.
        accumulated_scores: Scores from previous steps.
        task: Task description for evaluation context.
        config: Pipeline configuration.
        client: SDK client for agent invocation.

    Returns:
        Dict with step results including ``plan``, ``score``,
        ``code_block``, ``was_improvement``, and internal keys.
    """
    if k == 0:
        plan = initial_plan
    else:
        logger.info(
            "A_planner invocation start: k=%d, history_len=%d",
            k,
            len(accumulated_plans),
        )
        plan_result = await invoke_planner(
            original_code, list(accumulated_plans), list(accumulated_scores), client
        )
        if plan_result is None:
            logger.warning("A_planner returned None at k=%d; skipping", k)
            return {
                "plan": "[planner failed]",
                "score": None,
                "code_block": "",
                "was_improvement": False,
                "_successful": False,
            }
        logger.info(
            "A_planner invocation complete: k=%d, plan=%.200s",
            k,
            plan_result,
        )
        plan = plan_result

    return await _execute_coder_step(
        k, plan, original_code, solution, task, config, client
    )
