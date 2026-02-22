"""Phase 2 outer loop: ablation, summarize, and extractor agent invocation.

Implements ablation (Task 31) and summarize/extractor (Task 32) agents for
the Phase 2 outer loop pipeline (Algorithm 2, lines 5-8).

A_abl receives the current best solution and previous ablation summaries,
producing a self-contained ablation study script that tests 2-3 code
components.  The script is executed with a capped timeout and debug retry
on failure.  Raw output feeds into A_summarize.

A_summarize receives ablation code + raw output and produces a text summary
(T_abl^t).  Falls back to truncated raw output on empty response.

A_extractor receives the solution + ablation summary and produces structured
``ExtractorOutput`` containing ``list[RefinePlan]`` (code_block + plan).
Retries once on JSON parse failure.

Refs:
    SRS 05a — Phase 2 Outer Ablation (REQ-P2O-001 through REQ-P2O-007).
    SRS 05b — Summarize & Extractor (REQ-P2O-008 through REQ-P2O-018).
    SRS 05c — Ablation Execution (REQ-P2O-020, REQ-P2O-021).
    SRS 05d — Constraints (REQ-P2O-034 through REQ-P2O-036).
    IMPLEMENTATION_PLAN.md Tasks 31, 32.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from mle_star.execution import (
    build_execution_env,
    execute_script,
    extract_traceback,
    setup_working_directory,
    write_script,
)
from mle_star.models import (
    AgentType,
    CodeBlock,
    ExtractorOutput,
    Phase2Result,
    PipelineConfig,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
)
from mle_star.phase2_inner import run_phase2_inner_loop
from mle_star.prompts import PromptRegistry
from mle_star.safety import extract_code_block, make_debug_callback
from mle_star.scoring import is_improvement_or_equal

logger = logging.getLogger(__name__)

# Maximum ablation script timeout in seconds (REQ-P2O-035).
_ABLATION_TIMEOUT_CAP: int = 600


# ---------------------------------------------------------------------------
# Ablation summary formatting
# ---------------------------------------------------------------------------


def _format_previous_ablations(summaries: list[str]) -> str:
    """Format previous ablation summaries for the A_abl prompt template.

    When *summaries* is empty, returns an empty string so the prompt
    template omits the section entirely (REQ-P2O-002).  When non-empty,
    formats each summary with a numbered header.

    Args:
        summaries: Previous ablation summary texts (T_abl^0 … T_abl^{t-1}).

    Returns:
        Formatted text block or empty string.
    """
    if not summaries:
        return ""

    lines: list[str] = ["# Previous Ablation Study Results"]
    for i, summary in enumerate(summaries, start=1):
        lines.append(f"\n## Ablation Study {i}\n{summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A_abl — Ablation agent invocation (REQ-P2O-001 to REQ-P2O-007)
# ---------------------------------------------------------------------------


async def invoke_ablation(
    solution: SolutionScript,
    previous_summaries: list[str],
    client: Any,
) -> SolutionScript | None:
    """Invoke A_abl to generate an ablation study script (REQ-P2O-003).

    Renders the ablation prompt template with the current solution and
    formatted previous ablation summaries, sends it to the A_abl agent
    via the SDK client, and extracts the code block from the response.

    Args:
        solution: Current best solution (s_t) whose components to ablate.
        previous_summaries: Summaries from prior outer steps (T_abl).
        client: SDK client for agent invocation.

    Returns:
        A ``SolutionScript`` wrapping the ablation study code with
        ``phase=REFINED`` and ``is_executable=True``, or ``None`` if
        the agent response is empty or extraction yields no code.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.ABLATION)

    previous_text = _format_previous_ablations(previous_summaries)

    prompt = template.render(
        solution_script=solution.content,
        previous_ablations=previous_text,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.ABLATION),
        message=prompt,
    )

    extracted = extract_code_block(response)
    if not extracted.strip():
        logger.warning("A_abl returned empty code; treating as failure")
        return None

    return SolutionScript(
        content=extracted,
        phase=SolutionPhase.REFINED,
        is_executable=True,
    )


# ---------------------------------------------------------------------------
# Ablation timeout computation (REQ-P2O-035)
# ---------------------------------------------------------------------------


def compute_ablation_timeout(config: PipelineConfig) -> int:
    """Compute the execution timeout for ablation scripts (REQ-P2O-035).

    Formula: ``min(time_limit_seconds // (outer_loop_steps * 2), 600)``.
    Caps at 600 seconds to prevent a single ablation study from consuming
    an excessive portion of the total time budget.

    Args:
        config: Pipeline configuration providing time limit and outer steps.

    Returns:
        Timeout in seconds (positive integer, at most 600).
    """
    return min(
        config.time_limit_seconds // (config.outer_loop_steps * 2),
        _ABLATION_TIMEOUT_CAP,
    )


# ---------------------------------------------------------------------------
# Ablation script execution with debug retry (REQ-P2O-020, REQ-P2O-021)
# ---------------------------------------------------------------------------


def _is_execution_error(exit_code: int, timed_out: bool, stderr: str) -> bool:
    """Check whether a raw execution result represents an error.

    Args:
        exit_code: Process exit code (0 = success, -1 = timeout).
        timed_out: Whether the process was killed due to timeout.
        stderr: Standard error output from the process.

    Returns:
        ``True`` if the execution should be considered a failure.
    """
    if exit_code != 0:
        return True
    if timed_out:
        return True
    return "Traceback (most recent call last):" in stderr


async def execute_ablation_with_retry(
    ablation_script: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> tuple[str, str]:
    """Execute an ablation study script with debug retry on error (REQ-P2O-020).

    Writes the ablation script to disk, executes it with the ablation
    timeout (REQ-P2O-035), and on failure invokes the debugger agent up
    to ``config.max_debug_attempts`` times (REQ-P2O-021).  If all
    attempts are exhausted, returns empty strings.

    Args:
        ablation_script: The ablation study script to execute.
        task: Task description providing ``data_dir``.
        config: Pipeline configuration (timeout, debug attempts).
        client: SDK client for debugger agent invocation.

    Returns:
        A ``(stdout, stderr)`` tuple.  Returns ``("", "")`` if the
        script fails after all debug attempts.
    """
    timeout = compute_ablation_timeout(config)
    working_dir = setup_working_directory(task.data_dir)
    env = build_execution_env()

    current_script = ablation_script
    script_path = write_script(current_script, working_dir, "ablation_study.py")
    raw = await execute_script(script_path, working_dir, timeout, env)

    if not _is_execution_error(raw.exit_code, raw.timed_out, raw.stderr):
        return raw.stdout, raw.stderr

    # Error path — set up debug callback and retry loop (REQ-P2O-021).
    logger.warning(
        "Ablation script failed (exit_code=%d, timed_out=%s); entering debug retry",
        raw.exit_code,
        raw.timed_out,
    )
    debug_cb = make_debug_callback(task, config, client)

    for attempt in range(config.max_debug_attempts):
        tb = extract_traceback(raw.stderr) or raw.stderr[:500]
        current_script = await debug_cb(current_script, tb)

        script_path = write_script(current_script, working_dir, "ablation_study.py")
        raw = await execute_script(script_path, working_dir, timeout, env)

        if not _is_execution_error(raw.exit_code, raw.timed_out, raw.stderr):
            logger.info(
                "Ablation script fixed on debug attempt %d/%d",
                attempt + 1,
                config.max_debug_attempts,
            )
            return raw.stdout, raw.stderr

    logger.warning(
        "Ablation script failed after %d debug attempts; returning empty output",
        config.max_debug_attempts,
    )
    return "", ""


# ---------------------------------------------------------------------------
# Previous code blocks formatting (for A_extractor)
# ---------------------------------------------------------------------------

_SUMMARIZE_FALLBACK_PREFIX: str = "[Auto-summary from raw output] "
_SUMMARIZE_FALLBACK_MAX_CHARS: int = 2000


def _format_previous_blocks(blocks: list[str]) -> str:
    """Format previously improved code blocks for the A_extractor prompt.

    When *blocks* is empty, returns an empty string so the prompt template
    omits the section (REQ-P2O-013).  When non-empty, formats each block
    with a numbered header.

    Args:
        blocks: Code block content strings from prior outer iterations (C).

    Returns:
        Formatted text block or empty string.
    """
    if not blocks:
        return ""

    lines: list[str] = ["# Previously Improved Code Blocks"]
    for i, block in enumerate(blocks, start=1):
        lines.append(f"\n## Code Block {i}\n{block}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A_summarize — Ablation summarization (REQ-P2O-008 to REQ-P2O-011)
# ---------------------------------------------------------------------------


async def invoke_summarize(
    ablation_code: str,
    raw_output: str,
    client: Any,
) -> str:
    """Invoke A_summarize to produce a text summary of ablation results.

    Renders the summarize prompt template with ablation code and raw
    execution output, sends it to the A_summarize agent, and returns the
    full text response.  Falls back to truncated raw output on empty
    response (REQ-P2O-036).

    Args:
        ablation_code: Source code of the executed ablation script (a_t).
        raw_output: Raw stdout from ablation script execution (r_t).
        client: SDK client for agent invocation.

    Returns:
        Summary text (T_abl^t) — either the agent response or a fallback
        constructed from truncated raw output.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.SUMMARIZE)

    prompt = template.render(
        ablation_code=ablation_code,
        raw_result=raw_output,
    )

    response: str = await client.send_message(
        agent_type=str(AgentType.SUMMARIZE),
        message=prompt,
    )

    if not response.strip():
        logger.warning("A_summarize returned empty response; using fallback")
        return _SUMMARIZE_FALLBACK_PREFIX + raw_output[-_SUMMARIZE_FALLBACK_MAX_CHARS:]

    return response


# ---------------------------------------------------------------------------
# Code block validation (REQ-P2O-017)
# ---------------------------------------------------------------------------


def validate_code_block(code_block: str, solution: SolutionScript) -> bool:
    """Check whether *code_block* is an exact substring of the solution.

    Args:
        code_block: Candidate code block extracted by A_extractor.
        solution: Current best solution to validate against.

    Returns:
        ``True`` if *code_block* appears verbatim in ``solution.content``.
    """
    return code_block in solution.content


# ---------------------------------------------------------------------------
# A_extractor — Code block extraction (REQ-P2O-012 to REQ-P2O-018)
# ---------------------------------------------------------------------------


async def invoke_extractor(
    summary: str,
    solution: SolutionScript,
    previous_blocks: list[str],
    client: Any,
) -> ExtractorOutput | None:
    """Invoke A_extractor to identify a code block and refinement plan.

    Renders the extractor prompt template with the solution, ablation
    summary, and previous code blocks, sends it to A_extractor using
    structured output (``ExtractorOutput`` schema), and parses the JSON
    response.  Retries once on JSON parse failure (REQ-P2O-034).

    Args:
        summary: Ablation summary for the current outer step (T_abl^t).
        solution: Current best solution (s_t).
        previous_blocks: Code block strings from prior iterations (C).
        client: SDK client for agent invocation.

    Returns:
        Parsed ``ExtractorOutput`` containing one or more ``RefinePlan``
        objects, or ``None`` if both parsing attempts fail.
    """
    registry = PromptRegistry()
    template = registry.get(AgentType.EXTRACTOR)

    blocks_text = _format_previous_blocks(previous_blocks)

    prompt = template.render(
        solution_script=solution.content,
        ablation_summary=summary,
        previous_code_blocks=blocks_text,
    )

    output_format = {
        "type": "json_schema",
        "schema": ExtractorOutput.model_json_schema(),
    }

    for attempt in range(2):
        response: str = await client.send_message(
            agent_type=str(AgentType.EXTRACTOR),
            message=prompt,
            output_format=output_format,
        )
        try:
            return ExtractorOutput.model_validate_json(response)
        except (ValidationError, ValueError):
            logger.warning(
                "A_extractor response parse failure (attempt %d/2): %.200s",
                attempt + 1,
                response,
            )

    logger.warning("A_extractor failed after 2 attempts; returning None")
    return None


# ---------------------------------------------------------------------------
# Outer loop — single iteration helper (REQ-P2O-019 through REQ-P2O-030)
# ---------------------------------------------------------------------------


async def _run_outer_step(
    t: int,
    s_t: SolutionScript,
    h_best: float,
    ablation_summaries: list[str],
    code_block_strings: list[str],
    task: TaskDescription,
    config: PipelineConfig,
    client: Any,
) -> dict[str, Any]:
    """Execute a single outer loop iteration t (REQ-P2O-025).

    Runs the ablation → summarize → extract → inner loop pipeline for one
    outer step.  Returns a step record dict for ``step_history``.

    Args:
        t: Outer loop step index (0-based).
        s_t: Current best solution entering this step.
        h_best: Current best score entering this step.
        ablation_summaries: Accumulated summaries from prior steps (read-only snapshot).
        code_block_strings: Accumulated code block strings from prior steps.
        task: Task description for agent context.
        config: Pipeline configuration.
        client: SDK client for agent invocation.

    Returns:
        A dict with keys matching REQ-P2O-030: ``outer_step``, ``ablation_summary``,
        ``code_block``, ``plan``, ``inner_loop_attempts``, ``best_score_after_step``,
        ``was_skipped``, plus internal keys ``_new_best_solution`` and ``_new_h_best``
        for the caller to apply updates.
    """
    # Step 1: Invoke A_abl (REQ-P2O-001).
    ablation_script = await invoke_ablation(s_t, list(ablation_summaries), client)

    # Step 2: Execute ablation script (REQ-P2O-020).
    if ablation_script is not None:
        stdout, _stderr = await execute_ablation_with_retry(
            ablation_script, task, config, client
        )
        ablation_code = ablation_script.content
    else:
        logger.warning("Ablation agent returned None at t=%d; using empty output", t)
        stdout = ""
        ablation_code = ""

    # Step 3: Summarize ablation results (REQ-P2O-008).
    summary = await invoke_summarize(ablation_code, stdout, client)

    # Step 4: Extract code block and plan (REQ-P2O-012).
    extractor_output = await invoke_extractor(
        summary, s_t, list(code_block_strings), client
    )

    if extractor_output is None:
        logger.warning("Extractor returned None at t=%d; skipping iteration", t)
        return _make_skipped_step(t, summary, h_best)

    # Use FIRST plan (REQ-P2O-026).
    c_t = extractor_output.plans[0].code_block
    p_0 = extractor_output.plans[0].plan

    # Validate code block (REQ-P2O-017).
    if not validate_code_block(c_t, s_t):
        logger.warning("Code block validation failed at t=%d; skipping iteration", t)
        return _make_skipped_step(t, summary, h_best)

    # Step 5: Run inner loop (REQ-P2O-026).
    code_block = CodeBlock(content=c_t, outer_step=t)
    inner_result = await run_phase2_inner_loop(
        client=client,
        solution=s_t,
        code_block=code_block,
        initial_plan=p_0,
        best_score=h_best,
        task=task,
        config=config,
    )

    # Step 6: Update best (REQ-P2O-027) using >= semantics.
    new_h_best = h_best
    new_best_solution: SolutionScript | None = None
    if is_improvement_or_equal(inner_result.best_score, h_best, task.metric_direction):
        new_h_best = inner_result.best_score
        new_best_solution = inner_result.best_solution

    return {
        "outer_step": t,
        "ablation_summary": summary,
        "code_block": c_t,
        "plan": p_0,
        "inner_loop_attempts": list(inner_result.attempts),
        "best_score_after_step": new_h_best,
        "was_skipped": False,
        "_new_h_best": new_h_best,
        "_new_best_solution": new_best_solution,
    }


def _make_skipped_step(
    t: int,
    summary: str,
    h_best: float,
) -> dict[str, Any]:
    """Build a step history record for a skipped outer iteration (REQ-P2O-030).

    Args:
        t: Outer step index.
        summary: Ablation summary for this step (may still have been computed).
        h_best: Current best score (unchanged).

    Returns:
        Step record dict with ``was_skipped=True``.
    """
    return {
        "outer_step": t,
        "ablation_summary": summary,
        "code_block": "",
        "plan": "",
        "inner_loop_attempts": [],
        "best_score_after_step": h_best,
        "was_skipped": True,
        "_new_h_best": h_best,
        "_new_best_solution": None,
    }


# ---------------------------------------------------------------------------
# Outer loop orchestration entry point (REQ-P2O-019)
# ---------------------------------------------------------------------------


async def run_phase2_outer_loop(
    client: Any,
    task: TaskDescription,
    config: PipelineConfig,
    initial_solution: SolutionScript,
    initial_score: float,
    session_id: str,
) -> Phase2Result:
    """Execute the Phase 2 outer loop for T iterations (REQ-P2O-019).

    Implements Algorithm 2 outer loop: for each of T iterations, runs
    ablation → summarize → extract → inner loop, tracking the best solution
    using ``is_improvement_or_equal`` (>= semantics, REQ-P2O-027).

    Args:
        client: SDK client for agent invocation.
        task: Task description providing metric direction and context.
        config: Pipeline configuration providing ``outer_loop_steps`` (T).
        initial_solution: Starting solution s_0 from Phase 1.
        initial_score: Starting best score h_best (explicit ``float``).
        session_id: Identifier for this solution path.

    Returns:
        ``Phase2Result`` with accumulated ablation summaries, refined code
        blocks, best solution, best score, and per-step history.
    """
    s_final = initial_solution
    h_best = initial_score
    ablation_summaries: list[str] = []
    refined_blocks: list[CodeBlock] = []
    code_block_strings: list[str] = []
    step_history: list[dict[str, Any]] = []

    s_t = initial_solution

    for t in range(config.outer_loop_steps):
        logger.info(
            "Outer loop step %d/%d (h_best=%.6f, session=%s)",
            t + 1,
            config.outer_loop_steps,
            h_best,
            session_id,
        )

        step_record = await _run_outer_step(
            t=t,
            s_t=s_t,
            h_best=h_best,
            ablation_summaries=ablation_summaries,
            code_block_strings=code_block_strings,
            task=task,
            config=config,
            client=client,
        )

        # Apply updates from step result.
        new_h_best: float = step_record.pop("_new_h_best")
        new_best_solution: SolutionScript | None = step_record.pop("_new_best_solution")

        if new_best_solution is not None:
            s_final = new_best_solution
            s_t = new_best_solution
        h_best = new_h_best

        # Accumulate summaries and blocks (REQ-P2O-022, REQ-P2O-023).
        ablation_summaries.append(step_record["ablation_summary"])
        code_block_content = step_record["code_block"]
        code_block_strings.append(code_block_content)
        refined_blocks.append(CodeBlock(content=code_block_content, outer_step=t))

        step_history.append(step_record)

    return Phase2Result(
        ablation_summaries=ablation_summaries,
        refined_blocks=refined_blocks,
        best_solution=s_final,
        best_score=h_best,
        step_history=step_history,
    )
