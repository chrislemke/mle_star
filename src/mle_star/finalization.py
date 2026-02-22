"""Finalization: subsampling removal, test submission, contamination check, and orchestration.

Implements the subsampling removal step, A_test agent invocation for test
submission generation, data contamination check against reference discussions,
and the ``run_finalization`` entry point that orchestrates the full finalization
pipeline after the main pipeline phases complete.

Refs:
    SRS 08a — Finalization Subsampling (REQ-FN-001 through REQ-FN-009).
    SRS 08b — Finalization Test Submission (REQ-FN-010 through REQ-FN-025).
    SRS 08c — Finalization Contamination (REQ-FN-026 through REQ-FN-033).
    SRS 08d — Finalization Constraints (REQ-FN-039 through REQ-FN-044).
    IMPLEMENTATION_PLAN.md Tasks 38, 39, 40.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mle_star.execution import (
    evaluate_with_retry,
    get_submission_info,
    verify_submission,
)
from mle_star.models import (
    AgentType,
    DataContaminationResult,
    FinalResult,
    Phase1Result,
    Phase2Result,
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
    logger.info(
        "Subsampling extraction start: solution_content_length=%d",
        len(solution.content),
    )

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
        logger.info("Subsampling extraction result: found=False, block_length=0")
        logger.info("No subsampling code detected (empty extraction)")
        return solution

    if extracted_block not in solution.content:
        logger.info(
            "Subsampling extraction result: found=False, block_length=%d",
            len(extracted_block),
        )
        logger.info(
            "Extracted block not found in solution content; treating as no subsampling"
        )
        return solution

    logger.info(
        "Subsampling extraction result: found=True, block_length=%d",
        len(extracted_block),
    )

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

    logger.info(
        "Subsampling removal result: original_block_length=%d, replacement_block_length=%d",
        len(extracted_block),
        len(replacement_block),
    )

    # ------------------------------------------------------------------
    # Step 4 — Replace in solution (REQ-FN-007).
    # ------------------------------------------------------------------
    original_len = len(solution.content)
    try:
        result = solution.replace_block(extracted_block, replacement_block)
        logger.info(
            "Subsampling replacement result: success=True, content_length_change=%d",
            len(result.content) - original_len,
        )
        return result
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


# ---------------------------------------------------------------------------
# Data Contamination Check (REQ-FN-026 through REQ-FN-033)
# ---------------------------------------------------------------------------


async def check_contamination(
    client: Any,
    solution: SolutionScript,
    reference_discussions: list[str] | None,
) -> DataContaminationResult | None:
    """Check if a solution is contaminated by reference Kaggle discussions (REQ-FN-033).

    For each reference discussion, invokes the contamination check variant of
    A_test with structured output ``DataContaminationResult``.  Aggregates
    verdicts: ANY ``"Same"`` → overall ``"Same"``; ALL ``"Novel"`` → overall
    ``"Novel"`` (REQ-FN-031).

    Skips entirely when no references are provided (REQ-FN-030).

    On any failure (SDK error, parse error), returns ``None`` for graceful
    degradation (REQ-FN-041).

    Args:
        client: SDK client for agent invocation.
        solution: The final solution to check for contamination.
        reference_discussions: List of reference discussion texts, or ``None``.

    Returns:
        Overall ``DataContaminationResult``, or ``None`` if skipped or failed.
    """
    if not reference_discussions:
        logger.info("Contamination check skipped: no reference discussions provided")
        return None

    try:
        return await _check_contamination_impl(client, solution, reference_discussions)
    except Exception:
        logger.warning(
            "Contamination check failed; returning None",
            exc_info=True,
        )
        return None


async def _check_contamination_impl(
    client: Any,
    solution: SolutionScript,
    reference_discussions: list[str],
) -> DataContaminationResult:
    """Inner implementation of the contamination check pipeline.

    Separated from ``check_contamination`` so the outer function provides a
    single top-level exception boundary for graceful degradation (REQ-FN-041).

    Args:
        client: SDK client for agent invocation.
        solution: The final solution to check.
        reference_discussions: Non-empty list of reference discussion texts.

    Returns:
        Overall ``DataContaminationResult`` with aggregated verdict.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.TEST, variant="contamination_check")
    output_format = {
        "type": "json_schema",
        "schema": DataContaminationResult.model_json_schema(),
    }

    verdicts: list[str] = []
    for ref in reference_discussions:
        prompt = template.render(
            reference_discussion=ref,
            final_solution=solution.content,
        )
        response: str = await client.send_message(
            agent_type=str(AgentType.TEST),
            message=prompt,
            output_format=output_format,
        )
        result = DataContaminationResult.model_validate_json(response)
        verdicts.append(result.verdict)

    overall = "Same" if any(v == "Same" for v in verdicts) else "Novel"

    logger.info(
        "Contamination check: %d references, verdicts=%s, overall=%s",
        len(verdicts),
        verdicts,
        overall,
    )

    return DataContaminationResult(verdict=overall)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Finalization Orchestration (REQ-FN-034 through REQ-FN-036)
# ---------------------------------------------------------------------------


async def run_finalization(
    client: Any,
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    phase1_result: Phase1Result,
    phase2_results: list[Phase2Result],
    phase3_result: Phase3Result | None,
    reference_discussions: list[str] | None = None,
) -> FinalResult:
    """Orchestrate the full finalization pipeline (REQ-FN-034).

    Executes the following steps in order (REQ-FN-035):

    1. **Subsampling removal** — strip subsampling code from the solution.
    2. **Test submission generation** — produce a test-submission script.
    3. **Leakage check** — verify the test script for data leakage (REQ-SF-022).
    4. **Evaluation with retry** — execute the test script with debug retry.
    5. **Submission verification** — check that ``./final/submission.csv`` exists.
    6. **Fallback** — if evaluation or verification failed, use original solution.
    7. **Contamination check** — optional, only when references are provided.
    8. **FinalResult construction** — assemble and return the result.

    Args:
        client: SDK client for agent invocation.
        solution: Best solution from pipeline phases.
        task: Task description.
        config: Pipeline configuration.
        phase1_result: Phase 1 output.
        phase2_results: Phase 2 outputs (one per parallel path).
        phase3_result: Phase 3 output, or ``None`` if L=1.
        reference_discussions: Optional reference discussions for contamination check.

    Returns:
        A fully populated ``FinalResult``.
    """
    start = time.monotonic()

    logger.info(
        "Finalization start: solution_phase=%s, content_length=%d, competition_id=%s",
        solution.phase,
        len(solution.content),
        task.competition_id,
    )

    # Step 1 — Remove subsampling (REQ-FN-009).
    logger.info("Finalization: removing subsampling")
    solution_no_subsample = await remove_subsampling(client, solution, task)

    # Step 2 — Generate test submission (REQ-FN-019).
    logger.info("Finalization: generating test submission")
    test_script = await generate_test_submission(client, task, solution_no_subsample)

    # Step 3 — Leakage check on test script (REQ-SF-022).
    logger.info("Finalization: checking leakage on test script")
    test_script_checked = await check_and_fix_leakage(test_script, task, client)

    # Step 4 — Evaluate with retry (REQ-EX-021).
    logger.info("Finalization: evaluating test submission")
    debug_cb = make_debug_callback(task, config, client)
    final_solution, eval_result = await evaluate_with_retry(
        test_script_checked, task, config, debug_cb
    )

    # Step 5 — Verify submission (REQ-EX-024, REQ-EX-025).
    submission_verified = verify_submission(task.data_dir)
    submission_info = get_submission_info(task.data_dir)
    logger.info(
        "Submission verification result: exists=%s, size_bytes=%s, row_count=%s",
        submission_info.get("exists"),
        submission_info.get("size_bytes"),
        submission_info.get("row_count"),
    )

    # Step 6 — Fallback handling (REQ-FN-025).
    final_sol, sub_path = _apply_fallback(
        eval_result, submission_verified, submission_info, final_solution, solution
    )

    # Step 7 — Contamination check (REQ-FN-033).
    await check_contamination(client, final_sol, reference_discussions)

    # Step 8 — Build FinalResult (REQ-FN-036).
    duration = time.monotonic() - start
    logger.info(
        "Finalization complete: solution_phase=%s, submission_path=%s, duration=%.2fs",
        final_sol.phase,
        sub_path,
        duration,
    )

    return FinalResult(
        task=task,
        config=config,
        phase1=phase1_result,
        phase2_results=phase2_results,
        phase3=phase3_result,
        final_solution=final_sol,
        submission_path=sub_path,
        total_duration_seconds=duration,
        total_cost_usd=None,
    )


def _apply_fallback(
    eval_result: Any,
    submission_verified: bool,
    submission_info: dict[str, Any],
    evaluated_solution: SolutionScript,
    original_solution: SolutionScript,
) -> tuple[SolutionScript, str]:
    """Determine the final solution and submission path, applying fallback if needed.

    If the evaluation failed or the submission file was not produced,
    falls back to the original solution with an empty submission path
    (REQ-FN-025).

    Args:
        eval_result: Evaluation result from ``evaluate_with_retry``.
        submission_verified: Whether ``verify_submission`` returned ``True``.
        submission_info: Dict from ``get_submission_info``.
        evaluated_solution: Solution returned by the evaluation pipeline.
        original_solution: Original input solution for fallback.

    Returns:
        A ``(final_solution, submission_path)`` tuple.
    """
    if eval_result.is_error or not submission_verified:
        logger.warning(
            "Finalization fallback: eval_error=%s, verified=%s; using original solution",
            eval_result.is_error,
            submission_verified,
        )
        return original_solution, ""

    return evaluated_solution, str(submission_info.get("path", ""))
