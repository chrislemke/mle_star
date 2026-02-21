"""Phase 1: agent invocations and orchestration for Algorithm 1.

Implements ``retrieve_models``, ``generate_candidate``, ``merge_solutions``,
and ``run_phase1`` for the initial solution generation pipeline (Algorithm 1).
A_retriever searches the web for M effective models, A_init generates a
candidate solution script per model, and A_merger integrates a reference
solution into a base solution via ensemble.  ``run_phase1`` orchestrates the
full pipeline: retrieval, candidate generation/evaluation, sorting, and
merging with break-on-first-failure semantics.

Helper ``parse_retriever_output`` validates and deserializes the structured
JSON response from A_retriever.

Refs:
    SRS 04a — Phase 1 Agents (REQ-P1-001 through REQ-P1-017).
    SRS 04b — Phase 1 Orchestration (REQ-P1-018 through REQ-P1-029).
    IMPLEMENTATION_PLAN.md Tasks 27, 28.
"""

from __future__ import annotations

import logging
from typing import Any

from mle_star.execution import evaluate_with_retry, rank_solutions
from mle_star.models import (
    AgentType,
    EvaluationResult,
    Phase1Result,
    PipelineConfig,
    RetrievedModel,
    RetrieverOutput,
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


# ---------------------------------------------------------------------------
# A_retriever — Model retrieval (REQ-P1-001 to REQ-P1-007)
# ---------------------------------------------------------------------------


def parse_retriever_output(response: str) -> RetrieverOutput:
    """Parse the A_retriever structured JSON response (REQ-P1-004).

    Deserializes *response* into a ``RetrieverOutput`` model using
    ``model_validate_json``.  Raises ``ValueError`` with a descriptive
    message (including the first 500 characters of the raw response)
    on any parsing or validation failure.

    Args:
        response: Raw JSON string from the retriever agent.

    Returns:
        A validated ``RetrieverOutput`` instance.

    Raises:
        ValueError: If the response is not valid JSON or does not
            conform to the ``RetrieverOutput`` schema.
    """
    try:
        return RetrieverOutput.model_validate_json(response)
    except Exception as exc:
        truncated = response[:500]
        msg = f"Failed to parse retriever output: {exc}. Raw response: {truncated}"
        raise ValueError(msg) from exc


def _filter_valid_models(models: list[RetrievedModel]) -> list[RetrievedModel]:
    """Exclude models with empty model_name or example_code (REQ-P1-006).

    Models that fail validation are logged as warnings and excluded.

    Args:
        models: Raw list of retrieved models.

    Returns:
        Filtered list containing only models with non-empty fields.
    """
    valid: list[RetrievedModel] = []
    for model in models:
        if not model.model_name.strip():
            logger.warning(
                "Excluding model with empty model_name: %r",
                model,
            )
            continue
        if not model.example_code.strip():
            logger.warning(
                "Excluding model '%s' with empty example_code",
                model.model_name,
            )
            continue
        valid.append(model)
    return valid


async def retrieve_models(
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> list[RetrievedModel]:
    """Invoke A_retriever to find M effective models for the task (REQ-P1-007).

    Renders the retriever prompt template with the task description and
    requested model count, sends it to the A_retriever agent via the SDK
    client with structured output, parses and validates the response, and
    filters out models with empty fields.

    Args:
        task: Task description providing competition context.
        config: Pipeline configuration (provides ``num_retrieved_models``).
        client: SDK client for agent invocation.

    Returns:
        A list of validated ``RetrievedModel`` instances (1 to M).

    Raises:
        ValueError: If zero valid models remain after filtering.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.RETRIEVER)
    prompt = template.render(
        task_description=task.description,
        M=config.num_retrieved_models,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.RETRIEVER),
        message=prompt,
    )

    output = parse_retriever_output(response)
    valid_models = _filter_valid_models(output.models)

    if len(valid_models) == 0:
        msg = "A_retriever returned zero models"
        raise ValueError(msg)

    if len(valid_models) < config.num_retrieved_models:
        logger.warning(
            "A_retriever returned %d models (requested %d); proceeding with available",
            len(valid_models),
            config.num_retrieved_models,
        )

    return valid_models


# ---------------------------------------------------------------------------
# A_init — Candidate solution generation (REQ-P1-008 to REQ-P1-012)
# ---------------------------------------------------------------------------


async def generate_candidate(
    task: TaskDescription,
    model: RetrievedModel,
    config: PipelineConfig,
    client: Any,
) -> SolutionScript | None:
    """Invoke A_init to generate a candidate solution for a model (REQ-P1-012).

    Renders the init prompt template with the task description and model
    details, sends it to the A_init agent, extracts the code block from
    the response, and constructs a ``SolutionScript`` with
    ``phase=SolutionPhase.INIT``.

    Args:
        task: Task description providing competition context.
        model: The retrieved model to base the solution on.
        config: Pipeline configuration (unused directly but available
            for future extensions).
        client: SDK client for agent invocation.

    Returns:
        A ``SolutionScript`` with the generated code, or ``None`` if the
        agent response is empty or extraction yields no code.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.INIT)
    prompt = template.render(
        task_description=task.description,
        model_name=model.model_name,
        example_code=model.example_code,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.INIT),
        message=prompt,
    )

    extracted = extract_code_block(response)
    if not extracted.strip():
        logger.warning("A_init returned empty code for model '%s'", model.model_name)
        return None

    return SolutionScript(
        content=extracted,
        phase=SolutionPhase.INIT,
        source_model=model.model_name,
    )


# ---------------------------------------------------------------------------
# A_merger — Solution merging (REQ-P1-013 to REQ-P1-017)
# ---------------------------------------------------------------------------


async def merge_solutions(
    base: SolutionScript,
    reference: SolutionScript,
    config: PipelineConfig,
    client: Any,
) -> SolutionScript | None:
    """Invoke A_merger to integrate a reference solution into the base (REQ-P1-017).

    Renders the merger prompt template with the base and reference solution
    source code, sends it to the A_merger agent, extracts the code block
    from the response, and constructs a ``SolutionScript`` with
    ``phase=SolutionPhase.MERGED``.

    Args:
        base: Current best solution (code base).
        reference: Next-ranked candidate solution to integrate.
        config: Pipeline configuration (unused directly but available
            for future extensions).
        client: SDK client for agent invocation.

    Returns:
        A ``SolutionScript`` with the merged code, or ``None`` if the
        agent response is empty or extraction yields no code.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.MERGER)
    prompt = template.render(
        base_code=base.content,
        reference_code=reference.content,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.MERGER),
        message=prompt,
    )

    extracted = extract_code_block(response)
    if not extracted.strip():
        logger.warning("A_merger returned empty code")
        return None

    return SolutionScript(
        content=extracted,
        phase=SolutionPhase.MERGED,
    )


# ---------------------------------------------------------------------------
# Phase 1 orchestration — Algorithm 1 (REQ-P1-018 to REQ-P1-029)
# ---------------------------------------------------------------------------


class _CandidateResults:
    """Mutable accumulator for candidate generation results."""

    def __init__(self) -> None:
        self.solutions: list[SolutionScript] = []
        self.scores: list[float | None] = []
        self.successful_solutions: list[SolutionScript] = []
        self.successful_results: list[EvaluationResult] = []


async def _generate_and_evaluate_candidates(
    models: list[RetrievedModel],
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
    debug_cb: Any,
) -> _CandidateResults:
    """Generate, leakage-check, and evaluate M candidates (REQ-P1-020).

    For each retrieved model: generate a candidate, run the leakage checker,
    evaluate with debug retry, and record the outcome.  Candidates that fail
    generation or evaluation are recorded with ``score=None``.

    Args:
        models: Retrieved models from A_retriever.
        task: Task description providing competition context.
        config: Pipeline configuration with hyperparameters.
        client: SDK client for agent invocation.
        debug_cb: Debug callback for evaluate_with_retry.

    Returns:
        A ``_CandidateResults`` accumulator with all outcomes.
    """
    acc = _CandidateResults()

    for model in models:
        candidate = await generate_candidate(task, model, config, client)

        if candidate is None:
            logger.warning("A_init returned None for model '%s'", model.model_name)
            acc.solutions.append(
                SolutionScript(
                    content="",
                    phase=SolutionPhase.INIT,
                    source_model=model.model_name,
                    is_executable=False,
                )
            )
            acc.scores.append(None)
            continue

        candidate = await check_and_fix_leakage(candidate, task, client)
        candidate, result = await evaluate_with_retry(candidate, task, config, debug_cb)

        if result.is_error or result.score is None:
            logger.warning(
                "Candidate from model '%s' failed: %s",
                model.model_name,
                (result.error_traceback or "no score")[:80],
            )
            acc.solutions.append(candidate)
            acc.scores.append(None)
            continue

        candidate.score = result.score
        acc.solutions.append(candidate)
        acc.scores.append(result.score)
        acc.successful_solutions.append(candidate)
        acc.successful_results.append(result)

    return acc


async def _run_merge_loop(
    ranked: list[tuple[SolutionScript, EvaluationResult]],
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
    debug_cb: Any,
) -> tuple[SolutionScript, float]:
    """Execute the merge loop over sorted candidates (REQ-P1-025 to REQ-P1-028).

    Starting from the best-ranked candidate, iteratively merges each
    remaining candidate and checks for improvement.  Breaks on the first
    non-improvement, execution failure, or ``None`` merge result.

    Args:
        ranked: Sorted ``(solution, result)`` pairs, best first.
        task: Task description providing competition context.
        config: Pipeline configuration with hyperparameters.
        client: SDK client for agent invocation.
        debug_cb: Debug callback for evaluate_with_retry.

    Returns:
        A ``(best_solution, best_score)`` tuple after the merge loop.
    """
    s_0, best_result = ranked[0]
    assert best_result.score is not None
    h_best = best_result.score
    s_0.score = h_best

    for ranked_sol, _ranked_res in ranked[1:]:
        merged = await merge_solutions(s_0, ranked_sol, config, client)
        if merged is None:
            logger.warning("Merge returned None; breaking merge loop")
            break

        merged = await check_and_fix_leakage(merged, task, client)
        merged, merge_result = await evaluate_with_retry(merged, task, config, debug_cb)

        if merge_result.is_error or merge_result.score is None:
            logger.warning("Merged solution failed evaluation; breaking merge loop")
            break

        if is_improvement_or_equal(merge_result.score, h_best, task.metric_direction):
            s_0 = merged
            h_best = merge_result.score
            s_0.score = h_best
        else:
            break

    return s_0, h_best


async def run_phase1(
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> Phase1Result:
    """Execute Phase 1: model retrieval, candidate generation, and merging (REQ-P1-018).

    Implements Algorithm 1 from the MLE-STAR paper:

    1. Retrieve M models via A_retriever.
    2. Generate, leakage-check, and evaluate M candidates.
    3. Sort successful candidates by score (best first).
    4. Merge remaining candidates into the best one, breaking on
       the first non-improvement (>= semantics via
       ``is_improvement_or_equal``).

    Args:
        task: Task description providing competition context.
        config: Pipeline configuration with hyperparameters.
        client: SDK client for agent invocation.

    Returns:
        A ``Phase1Result`` containing retrieved models, candidate solutions
        and scores, and the final merged solution with its score.

    Raises:
        RuntimeError: If all M candidates fail to produce a valid score.
    """
    debug_cb = make_debug_callback(task, config, client)

    # Step 1 — Retrieve M models (REQ-P1-019).
    models = await retrieve_models(task, config, client)

    # Steps 2-5 — Generate and evaluate M candidates (REQ-P1-020).
    acc = await _generate_and_evaluate_candidates(
        models, task, config, client, debug_cb
    )

    # REQ-P1-022 — All candidates failed.
    if not acc.successful_solutions:
        msg = f"Phase 1 failed: all {len(models)} candidates produced execution errors"
        raise RuntimeError(msg)

    # Step 6 — Sort successful candidates by score, best first (REQ-P1-023).
    ranked = rank_solutions(
        acc.successful_solutions, acc.successful_results, task.metric_direction
    )

    # Steps 7-17 — Initialize best + merge loop (REQ-P1-024 to REQ-P1-028).
    s_0, h_best = await _run_merge_loop(ranked, task, config, client, debug_cb)

    # Construct Phase1Result (fields for Task 28; safety checks in Task 29).
    return Phase1Result(
        retrieved_models=models,
        candidate_solutions=acc.solutions,
        candidate_scores=acc.scores,
        initial_solution=s_0,
        initial_score=h_best,
    )
