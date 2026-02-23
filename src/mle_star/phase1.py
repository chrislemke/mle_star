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
    IMPLEMENTATION_PLAN.md Tasks 27, 28, 29.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any

from mle_star.execution import evaluate_with_retry, rank_solutions
from mle_star.models import (
    AgentType,
    EvaluationResult,
    Phase1Result,
    PipelineConfig,
    ResearchFindings,
    RetrievedModel,
    RetrieverOutput,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
)
from mle_star.prompts import PromptRegistry
from mle_star.safety import (
    check_and_fix_leakage,
    check_data_usage,
    extract_code_block,
    make_debug_callback,
)
from mle_star.scoring import is_improvement_or_equal

if TYPE_CHECKING:
    from mle_star.orchestrator import ClaudeCodeClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline — Simple default model generation
# ---------------------------------------------------------------------------


async def generate_baseline(
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
) -> SolutionScript | None:
    """Generate a simple baseline solution with default model settings.

    Creates the simplest possible model (e.g., CatBoost with defaults for
    tabular data) to establish a benchmark score before web retrieval.

    Args:
        task: Task description providing competition context.
        config: Pipeline configuration.
        client: SDK client for agent invocation.

    Returns:
        A ``SolutionScript`` with the baseline code, or ``None`` if the
        agent response is empty or extraction yields no code.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.BASELINE)
    prompt = template.render(
        task_description=task.description,
        target_column=task.target_column or "Not specified",
        task_type=task.task_type,
        data_modality=task.data_modality,
    )

    response: str = await client.send_message(
        agent_type=AgentType.BASELINE,
        message=prompt,
    )

    extracted = extract_code_block(response)

    # Check if the extracted content looks like Python code.
    if not _looks_like_python(extracted):
        workspace_file = Path(task.data_dir) / "solution.py"
        if workspace_file.exists():
            file_content = workspace_file.read_text(encoding="utf-8")
            if _looks_like_python(file_content):
                logger.info(
                    "Using solution.py from workspace for baseline "
                    "(agent wrote to file instead of returning code block)",
                )
                extracted = file_content

    if not extracted.strip():
        logger.warning("Baseline generation returned empty code")
        return None

    return SolutionScript(
        content=extracted,
        phase=SolutionPhase.INIT,
        source_model="baseline",
    )


# ---------------------------------------------------------------------------
# Internet research — Deep web research for techniques and insights
# ---------------------------------------------------------------------------


def _parse_research_output(response: str) -> ResearchFindings | None:
    """Parse the researcher agent response into ResearchFindings.

    Three-strategy parsing:
    1. Direct JSON parse.
    2. Extract embedded JSON from text.
    3. Fallback to raw_summary with empty lists.

    Returns ``None`` for empty responses.
    """
    if not response or not response.strip():
        return None

    # 1. Direct parse
    try:
        return ResearchFindings.model_validate_json(response)
    except Exception:
        pass

    # 2. Extract embedded JSON from text
    extracted = _extract_json_from_text(response)
    if extracted is not None:
        try:
            return ResearchFindings.model_validate_json(extracted)
        except Exception:
            pass

    # 3. Try JSON code blocks
    code_block_match = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
    if code_block_match:
        block = code_block_match.group(1).strip()
        try:
            return ResearchFindings.model_validate_json(block)
        except Exception:
            pass

    # 4. Fallback — use the raw response as a summary
    return ResearchFindings(
        model_recommendations=[],
        feature_engineering_ideas=[],
        preprocessing_ideas=[],
        other_insights=[],
        raw_summary=response.strip()[:2000],
    )


def _format_research_context(findings: ResearchFindings | None) -> str:
    """Format ResearchFindings into a human-readable string for prompt injection.

    Args:
        findings: Research findings to format, or None.

    Returns:
        A formatted string for injection into prompts, or empty string if None.
    """
    if findings is None:
        return ""

    sections: list[str] = []
    sections.append("# Research Findings")

    if findings.model_recommendations:
        sections.append("## Recommended Models")
        for item in findings.model_recommendations:
            sections.append(f"- {item}")

    if findings.feature_engineering_ideas:
        sections.append("## Feature Engineering Ideas")
        for item in findings.feature_engineering_ideas:
            sections.append(f"- {item}")

    if findings.preprocessing_ideas:
        sections.append("## Preprocessing Ideas")
        for item in findings.preprocessing_ideas:
            sections.append(f"- {item}")

    if findings.other_insights:
        sections.append("## Other Insights")
        for item in findings.other_insights:
            sections.append(f"- {item}")

    if findings.raw_summary:
        sections.append("## Summary")
        sections.append(findings.raw_summary)

    return "\n".join(sections)


async def conduct_research(
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
    baseline_score: float | None = None,
) -> ResearchFindings | None:
    """Conduct internet research for model architectures and techniques.

    Invokes the RESEARCHER agent to search the web for insights relevant
    to the competition task. Results are used to inform model retrieval
    and candidate generation.

    Args:
        task: Task description providing competition context.
        config: Pipeline configuration.
        client: SDK client for agent invocation.
        baseline_score: Current baseline score for context (if available).

    Returns:
        A ``ResearchFindings`` instance, or ``None`` on failure.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.RESEARCHER)
    prompt = template.render(
        task_description=task.description,
        target_column=task.target_column or "Not specified",
        task_type=task.task_type,
        data_modality=task.data_modality,
        baseline_score=baseline_score if baseline_score is not None else "N/A",
    )

    response: str = await client.send_message(
        agent_type=AgentType.RESEARCHER,
        message=prompt,
    )

    return _parse_research_output(response)


# ---------------------------------------------------------------------------
# A_retriever — Model retrieval (REQ-P1-001 to REQ-P1-007)
# ---------------------------------------------------------------------------


def _extract_json_from_text(text: str) -> str | None:
    """Try to extract a JSON object or array from free-form text.

    Scans for the outermost ``{...}`` or ``[...]`` block using brace/bracket
    counting.  Returns the extracted JSON string, or ``None`` if nothing
    valid is found.
    """
    # Try to find a JSON object first, then a JSON array
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except (json.JSONDecodeError, ValueError):
                        break
        # If we didn't find a valid match with this bracket type, try next
    return None


def parse_retriever_output(response: str) -> RetrieverOutput:
    """Parse the A_retriever structured JSON response (REQ-P1-004).

    Deserializes *response* into a ``RetrieverOutput`` model.  When the
    response is not directly valid JSON, attempts to extract an embedded
    JSON object or array from the text.  If a bare list is found, wraps
    it as ``{"models": [...]}``.

    Args:
        response: Raw JSON string (or text containing JSON) from the
            retriever agent.

    Returns:
        A validated ``RetrieverOutput`` instance.

    Raises:
        ValueError: If the response cannot be parsed into a valid
            ``RetrieverOutput``.
    """
    # 1. Direct parse (happy path)
    try:
        return RetrieverOutput.model_validate_json(response)
    except Exception:
        pass

    # 2. Try to extract embedded JSON from the text
    extracted = _extract_json_from_text(response)
    if extracted is not None:
        # If it's a bare list, wrap it in {"models": [...]}
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, list):
                wrapped = json.dumps({"models": parsed})
                return RetrieverOutput.model_validate_json(wrapped)
            if isinstance(parsed, dict):
                return RetrieverOutput.model_validate_json(extracted)
        except Exception:
            pass

    # 3. Try to find JSON code blocks (```json ... ```)
    code_block_match = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
    if code_block_match:
        block = code_block_match.group(1).strip()
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                wrapped = json.dumps({"models": parsed})
                return RetrieverOutput.model_validate_json(wrapped)
            if isinstance(parsed, dict):
                return RetrieverOutput.model_validate_json(block)
        except Exception:
            pass

    truncated = response[:500]
    msg = f"Failed to parse retriever output. Raw response: {truncated}"
    raise ValueError(msg)


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
    client: ClaudeCodeClient,
    *,
    max_retries: int = 3,
    research_context: str = "",
) -> list[RetrievedModel]:
    """Invoke A_retriever to find M effective models for the task (REQ-P1-007).

    Renders the retriever prompt template with the task description and
    requested model count, sends it to the A_retriever agent via the SDK
    client with structured output, parses and validates the response, and
    filters out models with empty fields.

    Retries up to *max_retries* times with exponential backoff when
    parsing fails (e.g. if the LLM returns plain text instead of JSON).

    Args:
        task: Task description providing competition context.
        config: Pipeline configuration (provides ``num_retrieved_models``).
        client: SDK client for agent invocation.
        max_retries: Maximum number of retry attempts on parse failure.
        research_context: Formatted research findings for prompt injection.

    Returns:
        A list of validated ``RetrievedModel`` instances (1 to M).

    Raises:
        ValueError: If zero valid models remain after filtering, or if
            all retry attempts are exhausted.
    """
    import asyncio

    registry = PromptRegistry()
    template = registry.get(AgentType.RETRIEVER)
    prompt = template.render(
        task_description=task.description,
        target_column=task.target_column or "Not specified",
        M=config.num_retrieved_models,
        research_context=research_context,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        response: str = await client.send_message(
            agent_type=AgentType.RETRIEVER,
            message=prompt,
        )

        try:
            output = parse_retriever_output(response)
        except ValueError as exc:
            last_error = exc
            if attempt < max_retries - 1:
                delay = 2**attempt
                logger.warning(
                    "Retriever parse attempt %d/%d failed: %s; retrying in %ds",
                    attempt + 1,
                    max_retries,
                    str(exc)[:200],
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            raise

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

    # Should not reach here, but just in case
    if last_error is not None:
        raise last_error
    msg = "retrieve_models: no attempts made"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# A_init — Candidate solution generation (REQ-P1-008 to REQ-P1-012)
# ---------------------------------------------------------------------------


def _looks_like_python(text: str) -> bool:
    """Heuristic check whether text looks like Python source code.

    Returns ``True`` if the text contains common Python keywords or
    constructs (import, def, class, print, assignment with ``=``).
    Returns ``False`` for empty text or text that appears to be
    natural language (e.g., markdown explanations).
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Check for common Python patterns
    python_indicators = [
        r"^\s*import\s+",
        r"^\s*from\s+\w+\s+import\s+",
        r"^\s*def\s+\w+\s*\(",
        r"^\s*class\s+\w+",
        r"^\s*if\s+__name__\s*==",
        r"^\s*print\s*\(",
        r"^\s*\w+\s*=\s*",
    ]
    for pattern in python_indicators:
        if re.search(pattern, stripped, re.MULTILINE):
            return True
    return False


async def generate_candidate(
    task: TaskDescription,
    model: RetrievedModel,
    config: PipelineConfig,
    client: ClaudeCodeClient,
    *,
    research_context: str = "",
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
        research_context: Formatted research findings for prompt injection.

    Returns:
        A ``SolutionScript`` with the generated code, or ``None`` if the
        agent response is empty or extraction yields no code.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.INIT)
    prompt = template.render(
        task_description=task.description,
        target_column=task.target_column or "Not specified",
        model_name=model.model_name,
        example_code=model.example_code,
        research_context=research_context,
    )

    response: str = await client.send_message(
        agent_type=AgentType.INIT,
        message=prompt,
    )

    extracted = extract_code_block(response)

    # Check if the extracted content looks like Python code.
    # If it doesn't (e.g., the agent returned explanatory text instead of a
    # code block), try reading the solution file from the workspace, since
    # agents with execution tools may write code directly to disk.
    if not _looks_like_python(extracted):
        workspace_file = Path(task.data_dir) / "solution.py"
        if workspace_file.exists():
            file_content = workspace_file.read_text(encoding="utf-8")
            if _looks_like_python(file_content):
                logger.info(
                    "Using solution.py from workspace (agent wrote to file "
                    "instead of returning code block) for model '%s'",
                    model.model_name,
                )
                extracted = file_content

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
    client: ClaudeCodeClient,
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
        agent_type=AgentType.MERGER,
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
    client: ClaudeCodeClient,
    debug_cb: Any,
    *,
    research_context: str = "",
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
        research_context: Formatted research findings for prompt injection.

    Returns:
        A ``_CandidateResults`` accumulator with all outcomes.
    """
    acc = _CandidateResults()
    num_models = len(models)

    for i, model in enumerate(models):
        logger.info(
            "Candidate generation start: %d/%d, model=%s",
            i + 1,
            num_models,
            model.model_name,
        )
        candidate = await generate_candidate(
            task, model, config, client, research_context=research_context
        )

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

        logger.info(
            "Candidate generation complete: model=%s, code_length=%d",
            model.model_name,
            len(candidate.content),
        )

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

        logger.info(
            "Candidate evaluation result: model=%s, score=%s, duration=%.2fs",
            model.model_name,
            result.score,
            result.duration_seconds,
        )
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
    client: ClaudeCodeClient,
    debug_cb: Any,
) -> tuple[SolutionScript, float, int]:
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
        A ``(best_solution, best_score, merge_count)`` tuple after the
        merge loop.
    """
    s_0, best_result = ranked[0]
    assert best_result.score is not None
    h_best = best_result.score
    s_0.score = h_best
    merge_count = 0

    for merge_idx, (ranked_sol, _ranked_res) in enumerate(ranked[1:], start=1):
        ref_model = ranked_sol.source_model or "unknown"
        logger.info(
            "Merge attempt start: index=%d, base_score=%s, reference=%s",
            merge_idx,
            h_best,
            ref_model,
        )
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
            logger.info(
                "Merge attempt result: score=%s, accepted (improved from %s)",
                merge_result.score,
                h_best,
            )
            s_0 = merged
            h_best = merge_result.score
            s_0.score = h_best
            merge_count += 1
        else:
            logger.info(
                "Merge attempt result: score=%s, rejected (no improvement over %s)",
                merge_result.score,
                h_best,
            )
            break

    return s_0, h_best, merge_count


async def _apply_safety_check(
    s_0: SolutionScript,
    h_best: float,
    check_fn: Any,
    task: TaskDescription,
    config: PipelineConfig,
    debug_cb: Any,
    client: ClaudeCodeClient,
    label: str,
) -> tuple[SolutionScript, float]:
    """Apply a single safety check with optional re-evaluation (REQ-P1-030/031).

    Calls *check_fn* on the current solution.  If the returned solution
    differs (identity check), re-evaluates it.  If re-evaluation fails
    (``is_error`` or ``score is None``), falls back to the pre-check version.

    Args:
        s_0: Current best solution.
        h_best: Current best score.
        check_fn: Async safety function ``(solution, task, client) -> solution``.
        task: Task description for evaluation context.
        config: Pipeline configuration.
        debug_cb: Debug callback for evaluate_with_retry.
        client: SDK client.
        label: Human-readable label for logging.

    Returns:
        Updated ``(solution, score)`` tuple.
    """
    logger.info("%s safety check start: content_length=%d", label, len(s_0.content))
    pre_check = s_0
    pre_score = h_best
    checked = await check_fn(s_0, task, client)

    modified = checked is not pre_check
    if modified:
        logger.info("%s safety check result: solution modified", label)
        checked, result = await evaluate_with_retry(checked, task, config, debug_cb)
        if result.is_error or result.score is None:
            logger.warning(
                "%s re-evaluation failed; falling back to pre-check solution",
                label,
            )
            s_0 = pre_check
            h_best = pre_score
            s_0.score = h_best
        else:
            s_0 = checked
            h_best = result.score
            s_0.score = h_best
    else:
        logger.info("%s safety check result: solution unchanged", label)
    return s_0, h_best


async def _apply_post_merge_safety(
    s_0: SolutionScript,
    h_best: float,
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
    debug_cb: Any,
) -> tuple[SolutionScript, float]:
    """Run post-merge safety checks: data usage then leakage (REQ-P1-030/031).

    1. ``check_data_usage`` — exactly once (REQ-P1-030).
    2. ``check_and_fix_leakage`` — after data check (REQ-P1-031).

    Each check may modify the solution; if so, re-evaluation occurs.
    On re-evaluation failure, falls back to the pre-check version.

    Args:
        s_0: Best solution after merge loop.
        h_best: Best score after merge loop.
        task: Task description.
        config: Pipeline configuration.
        client: SDK client.
        debug_cb: Debug callback for evaluate_with_retry.

    Returns:
        Final ``(solution, score)`` tuple after all safety checks.
    """
    s_0, h_best = await _apply_safety_check(
        s_0, h_best, check_data_usage, task, config, debug_cb, client, "A_data"
    )
    s_0, h_best = await _apply_safety_check(
        s_0, h_best, check_and_fix_leakage, task, config, debug_cb, client, "A_leakage"
    )
    return s_0, h_best


async def run_phase1(
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
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
    phase1_start = time.monotonic()
    logger.info(
        "Phase 1 start: competition_id=%s, M=%d",
        task.competition_id,
        config.num_retrieved_models,
    )

    debug_cb = make_debug_callback(task, config, client)

    # Step 0a — Generate baseline solution.
    baseline_score: float | None = None
    baseline_solution: SolutionScript | None = None
    try:
        logger.info("Baseline generation start")
        baseline_candidate = await generate_baseline(task, config, client)
        if baseline_candidate is not None:
            baseline_candidate = await check_and_fix_leakage(
                baseline_candidate, task, client
            )
            baseline_candidate, baseline_result = await evaluate_with_retry(
                baseline_candidate, task, config, debug_cb
            )
            if not baseline_result.is_error and baseline_result.score is not None:
                baseline_score = baseline_result.score
                baseline_candidate.score = baseline_score
                baseline_solution = baseline_candidate
                logger.info("Baseline score: %s", baseline_score)
            else:
                logger.warning("Baseline evaluation failed; continuing without baseline")
        else:
            logger.warning("Baseline generation returned None; continuing without baseline")
    except Exception:
        logger.exception("Baseline generation failed; continuing without baseline")

    # Step 0b — Conduct internet research.
    research_findings: ResearchFindings | None = None
    research_context: str = ""
    try:
        logger.info("Internet research start")
        research_findings = await conduct_research(
            task, config, client, baseline_score
        )
        if research_findings is not None:
            research_context = _format_research_context(research_findings)
            total_items = (
                len(research_findings.model_recommendations)
                + len(research_findings.feature_engineering_ideas)
                + len(research_findings.preprocessing_ideas)
                + len(research_findings.other_insights)
            )
            logger.info(
                "Internet research complete: %d findings", total_items
            )
        else:
            logger.warning("Research returned None; continuing without research context")
    except Exception:
        logger.exception("Internet research failed; continuing without research context")

    # Step 1 — Retrieve M models (REQ-P1-019).
    models = await retrieve_models(
        task, config, client, research_context=research_context
    )
    model_names = [m.model_name for m in models]
    logger.info(
        "Retrieval complete: count=%d, models=%s",
        len(models),
        model_names,
    )

    # Steps 2-5 — Generate and evaluate M candidates (REQ-P1-020).
    acc = await _generate_and_evaluate_candidates(
        models, task, config, client, debug_cb, research_context=research_context
    )

    # Inject successful baseline into candidates so it participates in ranking.
    if baseline_solution is not None and baseline_score is not None:
        acc.successful_solutions.append(baseline_solution)
        acc.successful_results.append(
            EvaluationResult(
                score=baseline_score,
                stdout="",
                stderr="",
                exit_code=0,
                duration_seconds=0.0,
                is_error=False,
            )
        )

    # REQ-P1-022 — All candidates failed.
    if not acc.successful_solutions:
        logger.error(
            "All candidates failed: M=%d, models=%s",
            len(models),
            model_names,
        )
        msg = f"Phase 1 failed: all {len(models)} candidates produced execution errors"
        raise RuntimeError(msg)

    # Step 6 — Sort successful candidates by score, best first (REQ-P1-023).
    ranked = rank_solutions(
        acc.successful_solutions, acc.successful_results, task.metric_direction
    )
    ranked_summary = [(sol.source_model or "unknown", res.score) for sol, res in ranked]
    logger.info("Candidates sorted: ranked_order=%s", ranked_summary)

    # Steps 7-17 — Initialize best + merge loop (REQ-P1-024 to REQ-P1-028).
    s_0, h_best, merge_count = await _run_merge_loop(
        ranked, task, config, client, debug_cb
    )

    # Post-merge safety checks (REQ-P1-030 to REQ-P1-033).
    s_0, h_best = await _apply_post_merge_safety(
        s_0, h_best, task, config, client, debug_cb
    )

    duration = time.monotonic() - phase1_start
    logger.info(
        "Phase 1 complete: final_score=%s, duration=%.2fs, merges=%d",
        h_best,
        duration,
        merge_count,
    )

    # Construct Phase1Result.
    return Phase1Result(
        retrieved_models=models,
        candidate_solutions=acc.solutions,
        candidate_scores=acc.scores,
        initial_solution=s_0,
        initial_score=h_best,
        baseline_score=baseline_score,
        baseline_solution=baseline_solution,
        research_findings=research_findings,
    )
