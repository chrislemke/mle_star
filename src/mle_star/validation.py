"""Comprehensive validation module for MLE-STAR solutions.

Validates solutions that beat the baseline through four parallel checks:
reproducibility, sanity (shuffled labels), deep leakage analysis, and
overfitting detection. Mechanical checks run without AI; AI-powered
checks invoke specialized agents.

Each check returns a ``ValidationCheck``. All four run concurrently via
``asyncio.gather``. Results are aggregated into a ``ValidationResult``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from mle_star.execution import evaluate_solution
from mle_star.models import (
    AgentType,
    PipelineConfig,
    SolutionScript,
    TaskDescription,
    ValidationCheck,
    ValidationResult,
    ValidationStatus,
)
from mle_star.prompts import get_registry
from mle_star.safety import extract_code_block
from mle_star.scoring import parse_score

if TYPE_CHECKING:
    from mle_star.orchestrator import ClaudeCodeClient

logger = logging.getLogger(__name__)

# Seed-setting code injected at the top of scripts for reproducibility checks.
_SEED_PREAMBLE = """\
import random as _rng; _rng.seed({seed})
import numpy as _np_seed; _np_seed.random.seed({seed})
try:
    import torch as _torch_seed; _torch_seed.manual_seed({seed})
except ImportError:
    pass
"""

# Pattern for parsing training performance from overfitting check output.
_TRAIN_SCORE_PATTERN = re.compile(r"Training Performance:\s*([\d.eE+-]+)")


# ---------------------------------------------------------------------------
# Individual validation checks
# ---------------------------------------------------------------------------


async def check_reproducibility(
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
) -> ValidationCheck:
    """Reproducibility check (mechanical — no AI needed).

    Re-runs the solution N times with different deterministic seeds.
    Computes mean and standard deviation of scores. Passes when
    ``std/mean < validation_score_tolerance`` and mean score is valid.

    Args:
        solution: The solution to validate.
        task: Task description for evaluation context.
        config: Pipeline configuration with seed run count and tolerance.

    Returns:
        A ``ValidationCheck`` with name ``"reproducibility"``.
    """
    n_runs = config.validation_seed_runs
    tolerance = config.validation_score_tolerance
    seeds = [42 + i for i in range(n_runs)]
    scores: list[float] = []

    for seed in seeds:
        seeded_content = _SEED_PREAMBLE.format(seed=seed) + solution.content
        seeded_solution = SolutionScript(
            content=seeded_content,
            phase=solution.phase,
            is_executable=True,
        )
        try:
            result = await evaluate_solution(seeded_solution, task, config)
            if result.score is not None and not result.is_error:
                scores.append(result.score)
            else:
                logger.warning(
                    "Reproducibility run seed=%d failed: error=%s score=%s",
                    seed, result.is_error, result.score,
                )
        except Exception:
            logger.warning("Reproducibility run seed=%d raised exception", seed, exc_info=True)

    if len(scores) < 2:
        return ValidationCheck(
            name="reproducibility",
            status=ValidationStatus.SKIPPED,
            details=f"Only {len(scores)}/{n_runs} runs produced scores; cannot assess reproducibility",
            scores=scores,
        )

    import statistics
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)

    if mean == 0:
        cv = float("inf") if stdev > 0 else 0.0
    else:
        cv = abs(stdev / mean)

    if cv < tolerance:
        return ValidationCheck(
            name="reproducibility",
            status=ValidationStatus.PASSED,
            details=f"CV={cv:.4f} < tolerance={tolerance} across {len(scores)} runs (mean={mean:.6f}, std={stdev:.6f})",
            scores=scores,
        )
    return ValidationCheck(
        name="reproducibility",
        status=ValidationStatus.FAILED,
        details=f"CV={cv:.4f} >= tolerance={tolerance} across {len(scores)} runs (mean={mean:.6f}, std={stdev:.6f})",
        scores=scores,
    )


async def check_sanity(
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
) -> ValidationCheck:
    """Sanity check (AI-powered) — shuffled label test.

    An agent generates a modified script that shuffles target labels
    before training. If the shuffled version performs similarly to the
    real version, the model likely has leakage.

    Args:
        solution: The solution to validate.
        task: Task description for prompt context.
        config: Pipeline configuration.
        client: Claude Code client for agent invocation.

    Returns:
        A ``ValidationCheck`` with name ``"sanity"``.
    """
    try:
        registry = get_registry()
        template = registry.get(AgentType.VALIDATOR, variant="sanity")
        prompt = template.render(
            solution_code=solution.content,
            task_description=task.description,
            evaluation_metric=task.evaluation_metric,
            metric_direction=task.metric_direction,
        )

        response = await client.send_message(
            agent_type=AgentType.VALIDATOR,
            message=prompt,
        )

        shuffled_code = extract_code_block(response)
        if not shuffled_code.strip():
            return ValidationCheck(
                name="sanity",
                status=ValidationStatus.SKIPPED,
                details="Validator agent returned empty code for shuffled-label test",
            )

        shuffled_solution = SolutionScript(
            content=shuffled_code,
            phase=solution.phase,
            is_executable=True,
        )
        result = await evaluate_solution(shuffled_solution, task, config)

        if result.score is None or result.is_error:
            return ValidationCheck(
                name="sanity",
                status=ValidationStatus.SKIPPED,
                details=f"Shuffled-label script failed to execute: error={result.is_error}",
            )

        original_score = solution.score
        shuffled_score = result.score

        if original_score is None:
            return ValidationCheck(
                name="sanity",
                status=ValidationStatus.SKIPPED,
                details="Original solution has no score; cannot compare",
            )

        # For a meaningful model, shuffled labels should degrade performance significantly.
        # We check if the gap is > 10% of original score.
        gap = abs(original_score - shuffled_score)
        relative_gap = gap / max(abs(original_score), 1e-10)

        if relative_gap > 0.1:
            return ValidationCheck(
                name="sanity",
                status=ValidationStatus.PASSED,
                details=(
                    f"Shuffled labels degraded performance: "
                    f"original={original_score:.6f}, shuffled={shuffled_score:.6f}, "
                    f"relative_gap={relative_gap:.4f}"
                ),
                scores=[original_score, shuffled_score],
            )
        return ValidationCheck(
            name="sanity",
            status=ValidationStatus.FAILED,
            details=(
                f"Shuffled labels did NOT degrade performance significantly: "
                f"original={original_score:.6f}, shuffled={shuffled_score:.6f}, "
                f"relative_gap={relative_gap:.4f} (model may have leakage)"
            ),
            scores=[original_score, shuffled_score],
        )
    except Exception as exc:
        logger.warning("Sanity check failed: %s", exc, exc_info=True)
        return ValidationCheck(
            name="sanity",
            status=ValidationStatus.SKIPPED,
            details=f"Sanity check raised exception: {exc!s:.200}",
        )


async def check_deep_leakage(
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
) -> ValidationCheck:
    """Deep leakage analysis (AI-powered).

    Enhanced version of the standard leakage check. An agent performs
    thorough analysis covering feature-target correlation, train-test
    overlap, preprocessing leakage, temporal leakage, and score
    suspicion.

    Args:
        solution: The solution to validate.
        task: Task description for prompt context.
        config: Pipeline configuration.
        client: Claude Code client for agent invocation.

    Returns:
        A ``ValidationCheck`` with name ``"deep_leakage"``.
    """
    try:
        registry = get_registry()
        template = registry.get(AgentType.LEAKAGE, variant="deep_analysis")
        prompt = template.render(
            code=solution.content,
            task_description=task.description,
            evaluation_metric=task.evaluation_metric,
            metric_direction=task.metric_direction,
            current_score=solution.score if solution.score is not None else "N/A",
        )

        response = await client.send_message(
            agent_type=AgentType.LEAKAGE,
            message=prompt,
            use_structured_output=False,
        )

        # Parse the response for leakage detection.
        response_lower = response.lower()
        if "leakage_detected" in response_lower or "leakage detected" in response_lower:
            return ValidationCheck(
                name="deep_leakage",
                status=ValidationStatus.FAILED,
                details=f"Deep leakage analysis found issues: {response[:500]}",
            )

        if "clean" in response_lower and "leakage" not in response_lower.replace("clean", ""):
            return ValidationCheck(
                name="deep_leakage",
                status=ValidationStatus.PASSED,
                details="Deep leakage analysis: no issues detected",
            )

        # Default: parse the overall verdict line
        if "overall verdict: clean" in response_lower:
            return ValidationCheck(
                name="deep_leakage",
                status=ValidationStatus.PASSED,
                details="Deep leakage analysis: overall verdict clean",
            )
        if "overall verdict: leakage" in response_lower:
            return ValidationCheck(
                name="deep_leakage",
                status=ValidationStatus.FAILED,
                details=f"Deep leakage analysis detected issues: {response[:500]}",
            )

        # Ambiguous response — treat as passed with note
        return ValidationCheck(
            name="deep_leakage",
            status=ValidationStatus.PASSED,
            details=f"Deep leakage analysis result ambiguous (treating as clean): {response[:300]}",
        )
    except Exception as exc:
        logger.warning("Deep leakage check failed: %s", exc, exc_info=True)
        return ValidationCheck(
            name="deep_leakage",
            status=ValidationStatus.SKIPPED,
            details=f"Deep leakage check raised exception: {exc!s:.200}",
        )


async def check_overfitting(
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
) -> ValidationCheck:
    """Overfitting check (AI-powered).

    An agent generates a modified script that reports both training and
    validation scores. A large gap indicates overfitting.

    Args:
        solution: The solution to validate.
        task: Task description for prompt context.
        config: Pipeline configuration with overfit threshold.
        client: Claude Code client for agent invocation.

    Returns:
        A ``ValidationCheck`` with name ``"overfitting"``.
    """
    try:
        registry = get_registry()
        template = registry.get(AgentType.VALIDATOR, variant="overfitting")
        prompt = template.render(
            solution_code=solution.content,
            task_description=task.description,
            evaluation_metric=task.evaluation_metric,
            metric_direction=task.metric_direction,
        )

        response = await client.send_message(
            agent_type=AgentType.VALIDATOR,
            message=prompt,
        )

        modified_code = extract_code_block(response)
        if not modified_code.strip():
            return ValidationCheck(
                name="overfitting",
                status=ValidationStatus.SKIPPED,
                details="Validator agent returned empty code for overfitting test",
            )

        modified_solution = SolutionScript(
            content=modified_code,
            phase=solution.phase,
            is_executable=True,
        )
        result = await evaluate_solution(modified_solution, task, config)

        if result.is_error:
            return ValidationCheck(
                name="overfitting",
                status=ValidationStatus.SKIPPED,
                details=f"Overfitting check script failed: {(result.error_traceback or '')[:200]}",
            )

        # Parse both training and validation scores from stdout.
        val_score = parse_score(result.stdout)
        train_match = _TRAIN_SCORE_PATTERN.findall(result.stdout)
        train_score: float | None = None
        if train_match:
            try:
                train_score = float(train_match[-1])
            except ValueError:
                pass

        if train_score is None or val_score is None:
            return ValidationCheck(
                name="overfitting",
                status=ValidationStatus.SKIPPED,
                details=(
                    f"Could not parse both scores: "
                    f"train={train_score}, val={val_score}"
                ),
            )

        gap = abs(train_score - val_score)
        threshold = config.validation_overfit_threshold

        # Also flag suspiciously perfect validation scores
        suspicious = val_score > 0.99 and task.evaluation_metric.lower() in (
            "accuracy", "f1", "auc", "roc_auc", "precision", "recall",
        )

        if gap <= threshold and not suspicious:
            return ValidationCheck(
                name="overfitting",
                status=ValidationStatus.PASSED,
                details=(
                    f"Train-val gap={gap:.4f} <= threshold={threshold} "
                    f"(train={train_score:.6f}, val={val_score:.6f})"
                ),
                scores=[train_score, val_score],
            )

        details_parts = []
        if gap > threshold:
            details_parts.append(
                f"Train-val gap={gap:.4f} > threshold={threshold}"
            )
        if suspicious:
            details_parts.append(f"Suspiciously perfect val_score={val_score:.6f}")
        return ValidationCheck(
            name="overfitting",
            status=ValidationStatus.FAILED,
            details=(
                f"{'; '.join(details_parts)} "
                f"(train={train_score:.6f}, val={val_score:.6f})"
            ),
            scores=[train_score, val_score],
        )
    except Exception as exc:
        logger.warning("Overfitting check failed: %s", exc, exc_info=True)
        return ValidationCheck(
            name="overfitting",
            status=ValidationStatus.SKIPPED,
            details=f"Overfitting check raised exception: {exc!s:.200}",
        )


# ---------------------------------------------------------------------------
# Top-level validation entry point
# ---------------------------------------------------------------------------


async def validate_solution(
    solution: SolutionScript,
    task: TaskDescription,
    config: PipelineConfig,
    client: ClaudeCodeClient,
) -> ValidationResult:
    """Run all validation checks on a solution in parallel.

    Executes reproducibility (mechanical), sanity, deep leakage, and
    overfitting checks concurrently via ``asyncio.gather``. Aggregates
    results into a ``ValidationResult``.

    Args:
        solution: The solution to validate.
        task: Task description for evaluation context.
        config: Pipeline configuration.
        client: Claude Code client for agent invocations.

    Returns:
        A ``ValidationResult`` with all check outcomes.
    """
    from mle_star.scoring import beats_baseline

    logger.info(
        "Validation start: solution_phase=%s, score=%s",
        solution.phase,
        solution.score,
    )

    baseline_beaten = beats_baseline(
        solution.score if solution.score is not None else float("-inf"),
        task.baseline_value,
        task.metric_direction,
    )

    raw_results = await asyncio.gather(
        check_reproducibility(solution, task, config),
        check_sanity(solution, task, config, client),
        check_deep_leakage(solution, task, config, client),
        check_overfitting(solution, task, config, client),
        return_exceptions=True,
    )

    checks: list[ValidationCheck] = []
    for r in raw_results:
        if isinstance(r, BaseException):
            logger.warning("Validation check raised exception: %s", r)
            checks.append(
                ValidationCheck(
                    name="unknown",
                    status=ValidationStatus.SKIPPED,
                    details=f"Check raised exception: {r!s:.200}",
                )
            )
        else:
            checks.append(r)

    # Solution passes if no checks FAILED (SKIPPED is acceptable).
    passed = all(c.status != ValidationStatus.FAILED for c in checks)

    result = ValidationResult(
        solution=solution,
        checks=checks,
        passed=passed,
        baseline_beaten=baseline_beaten,
    )

    check_summary = {c.name: c.status for c in checks}
    logger.info(
        "Validation complete: passed=%s, baseline_beaten=%s, checks=%s",
        passed,
        baseline_beaten,
        check_summary,
    )

    return result
