"""Phase 3: ensemble planner, ensembler agents, and orchestration.

Implements ``invoke_ens_planner`` and ``invoke_ensembler`` for Algorithm 3
(ensemble construction), plus ``run_phase3`` orchestration.  A_ens_planner
proposes ensemble strategies from L solution scripts and accumulated
history.  A_ensembler implements a given plan into a single self-contained
Python script.  ``run_phase3`` executes R ensemble rounds, selecting the
best ensemble via ``is_improvement_or_equal`` (>= semantics).

Helper functions ``_format_solutions`` and ``_format_ensemble_history``
produce the formatted text blocks inserted into prompt templates.

Refs:
    SRS 07a — Phase 3 Agents (REQ-P3-001 through REQ-P3-016).
    SRS 07b — Phase 3 Orchestration (REQ-P3-017 through REQ-P3-035).
    IMPLEMENTATION_PLAN.md Tasks 35, 36.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mle_star.execution import evaluate_with_retry
from mle_star.models import (
    AgentType,
    MetricDirection,
    Phase3Result,
    PipelineConfig,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
)
from mle_star.prompts import PromptRegistry
from mle_star.safety import (
    check_and_fix_leakage,
    extract_code_block,
    make_debug_callback,
)
from mle_star.scoring import is_improvement_or_equal

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


# ---------------------------------------------------------------------------
# Phase 3 Orchestration (REQ-P3-017 through REQ-P3-035, Task 36)
# ---------------------------------------------------------------------------


def _select_best_input(
    solutions: list[SolutionScript],
    task: TaskDescription,
) -> tuple[SolutionScript, float]:
    """Select the best-scoring input solution for fallback (REQ-P3-026).

    Uses direct comparison (max/min) based on metric direction.
    Only considers solutions with non-None scores.

    Args:
        solutions: Input solution scripts (must be non-empty).
        task: Task description providing metric direction.

    Returns:
        Tuple of (best solution, best score).
    """
    scored = [(s, s.score) for s in solutions if s.score is not None]
    if not scored:
        return solutions[0], 0.0

    if task.metric_direction == MetricDirection.MAXIMIZE:
        return max(scored, key=lambda x: x[1])  # type: ignore[return-value]
    return min(scored, key=lambda x: x[1])  # type: ignore[return-value]


async def _run_ensemble_round(
    solutions: list[SolutionScript],
    plans_snapshot: list[str],
    scores_snapshot: list[float | None],
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
    debug_cb: Any,
    *,
    round_index: int = 0,
) -> tuple[str, float | None, SolutionScript]:
    """Execute a single ensemble round: plan, implement, check, evaluate.

    Handles planner and ensembler failures gracefully by returning
    placeholder values.  Called by ``run_phase3`` for each round r.

    Args:
        solutions: L input solution scripts to ensemble.
        plans_snapshot: Copy of accumulated plan history at this round.
        scores_snapshot: Copy of accumulated score history at this round.
        task: Task description for evaluation context.
        config: Pipeline configuration.
        client: SDK client for agent invocation.
        debug_cb: Debug callback for ``evaluate_with_retry``.
        round_index: Current round index (for logging).

    Returns:
        Tuple of ``(plan_text, score_or_none, solution)``.
    """
    empty_solution = SolutionScript(content="", phase=SolutionPhase.ENSEMBLE)

    # REQ-P3-039: Ensemble round start.
    logger.info(
        "Ensemble round start: r=%d, previous_plans=%d",
        round_index,
        len(plans_snapshot),
    )

    # Step 1: Plan (REQ-P3-019 / REQ-P3-022 step 1).
    logger.info(
        "A_ens_planner invocation start: r=%d, history_size=%d",
        round_index,
        len(plans_snapshot),
    )
    plan = await invoke_ens_planner(solutions, plans_snapshot, scores_snapshot, client)
    if plan is None:
        logger.warning("A_ens_planner empty response: r=%d", round_index)
        return "[ens_planner failed]", None, empty_solution
    logger.info(
        "A_ens_planner invocation complete: r=%d, plan=%.200s",
        round_index,
        plan,
    )

    # Step 2: Implement (REQ-P3-020 / REQ-P3-022 step 3).
    logger.info(
        "A_ensembler invocation start: r=%d, plan=%.200s",
        round_index,
        plan,
    )
    ensemble_sol = await invoke_ensembler(plan, solutions, client)
    if ensemble_sol is None:
        logger.warning("A_ensembler extraction failure: r=%d", round_index)
        return plan, None, empty_solution
    logger.info(
        "A_ensembler invocation complete: r=%d, script_length=%d",
        round_index,
        len(ensemble_sol.content),
    )

    # Step 3: Leakage check (REQ-P3-027).
    logger.info(
        "Leakage check start: r=%d, solution_content_length=%d",
        round_index,
        len(ensemble_sol.content),
    )
    checked = await check_and_fix_leakage(ensemble_sol, task, client)
    content_changed = checked is not ensemble_sol
    logger.info(
        "Leakage check complete: r=%d, leakage_found=%s, content_changed=%s",
        round_index,
        "yes" if content_changed else "no",
        "yes" if content_changed else "no",
    )

    # Step 4: Evaluate with debug retry (REQ-P3-028).
    logger.info(
        "Evaluation start: r=%d, solution_content_length=%d",
        round_index,
        len(checked.content),
    )
    eval_start = time.monotonic()
    evaluated, eval_result = await evaluate_with_retry(checked, task, config, debug_cb)
    eval_duration = time.monotonic() - eval_start

    score = eval_result.score
    if score is not None:
        evaluated.score = score

    score_str = str(score) if score is not None else "failed"
    logger.info(
        "Evaluation complete: r=%d, score=%s, is_error=%s, duration=%.2fs",
        round_index,
        score_str,
        eval_result.is_error,
        eval_duration,
    )

    if eval_result.is_error:
        logger.warning(
            "Round failed (execution error): r=%d, error_summary=%s, plan_summary=%.200s",
            round_index,
            score_str,
            plan,
        )

    return plan, score, evaluated


async def _execute_rounds(
    solutions: list[SolutionScript],
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
    debug_cb: Any,
) -> tuple[
    list[str], list[float | None], SolutionScript | None, float | None, int, int
]:
    """Execute all R ensemble rounds and track best solution.

    Args:
        solutions: L input solution scripts.
        task: Task description for evaluation context.
        config: Pipeline configuration.
        client: SDK client for agent invocation.
        debug_cb: Debug callback for evaluation.

    Returns:
        Tuple of (plans, scores, best_solution, best_score, best_round,
        successful_rounds).
    """
    accumulated_plans: list[str] = []
    accumulated_scores: list[float | None] = []
    best_score: float | None = None
    best_solution: SolutionScript | None = None
    best_round: int = -1
    successful_rounds = 0

    for _r in range(config.ensemble_rounds):
        round_result = await _run_ensemble_round(
            solutions,
            list(accumulated_plans),
            list(accumulated_scores),
            task,
            config,
            client,
            debug_cb,
            round_index=_r,
        )
        plan, round_score, sol = round_result
        accumulated_plans.append(plan)
        accumulated_scores.append(round_score)

        if round_score is not None:
            successful_rounds += 1

        # REQ-P3-025: Update best using >= semantics (last tie wins).
        if round_score is not None and (
            best_score is None
            or is_improvement_or_equal(round_score, best_score, task.metric_direction)
        ):
            best_score = round_score
            best_solution = sol
            best_round = _r

    return (
        accumulated_plans,
        accumulated_scores,
        best_solution,
        best_score,
        best_round,
        successful_rounds,
    )


async def run_phase3(
    client: Any,
    task: TaskDescription,
    config: PipelineConfig,
    solutions: list[SolutionScript],
) -> Phase3Result:
    """Execute Phase 3: ensemble construction via Algorithm 3 (REQ-P3-017).

    Orchestrates R ensemble rounds.  Each round invokes A_ens_planner to
    propose an ensemble strategy, A_ensembler to implement it, applies
    leakage checking, and evaluates with debug retry.  The best ensemble
    solution is selected using ``is_improvement_or_equal`` (>= semantics,
    so the last occurrence of a tied score wins per REQ-P3-025).

    If ``len(solutions) == 1``, skips ensemble entirely (REQ-P3-018).
    If all R rounds fail, falls back to the best input solution without
    raising an exception (REQ-P3-026).

    Args:
        client: SDK client for agent invocation.
        task: Task description providing competition context.
        config: Pipeline configuration (``ensemble_rounds`` = R).
        solutions: L solution scripts to ensemble (must be >= 1).

    Returns:
        A ``Phase3Result`` with the best ensemble solution and scores.

    Raises:
        ValueError: If *solutions* is empty.
    """
    if not solutions:
        msg = "run_phase3 requires at least 1 solution"
        raise ValueError(msg)

    num_rounds = config.ensemble_rounds
    num_solutions = len(solutions)

    # REQ-P3-018: Single solution — skip ensemble entirely.
    if num_solutions == 1:
        sol = solutions[0]
        score = sol.score if sol.score is not None else 0.0
        logger.info(
            "Phase 3 skipped (single solution): score=%s, competition_id=%s",
            score,
            task.competition_id,
        )
        return Phase3Result(
            input_solutions=[sol],
            ensemble_plans=[],
            ensemble_scores=[],
            best_ensemble=sol,
            best_ensemble_score=score,
        )

    # REQ-P3-039: Phase 3 start.
    logger.info(
        "Phase 3 start: L=%d, R=%d, competition_id=%s",
        num_solutions,
        num_rounds,
        task.competition_id,
    )

    phase_start = time.monotonic()
    debug_cb = make_debug_callback(task, config, client)

    (
        accumulated_plans,
        accumulated_scores,
        best_solution,
        best_score,
        best_round,
        successful_rounds,
    ) = await _execute_rounds(solutions, task, config, client, debug_cb)

    # REQ-P3-026: All rounds failed — fallback to best input solution.
    if best_solution is None or best_score is None:
        logger.warning(
            "All rounds failed (fallback): R=%d, falling back to best input solution",
            num_rounds,
        )
        best_solution, best_score = _select_best_input(solutions, task)
    else:
        logger.info(
            "Best selection: best_round=%d, best_score=%s, successful_rounds=%d",
            best_round,
            best_score,
            successful_rounds,
        )

    phase_duration = time.monotonic() - phase_start
    logger.info(
        "Phase 3 complete: best_score=%s, best_round=%d, duration=%.2fs, rounds_attempted=%d",
        best_score,
        best_round,
        phase_duration,
        num_rounds,
    )

    return Phase3Result(
        input_solutions=solutions,
        ensemble_plans=accumulated_plans,
        ensemble_scores=accumulated_scores,
        best_ensemble=best_solution,
        best_ensemble_score=best_score,
    )
