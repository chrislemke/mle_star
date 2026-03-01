"""Score parsing and comparison functions for the MLE-STAR pipeline.

Provides the ``ScoreFunction`` protocol, ``parse_score`` regex extraction,
and two comparison functions (``is_improvement``, ``is_improvement_or_equal``)
that respect ``MetricDirection``.

Refs:
    SRS 01c — Score Function Interface (REQ-DM-026 through REQ-DM-029).
    REQ-EX-011 — parse_score returns the LAST match.
    IMPLEMENTATION_PLAN.md Task 07.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from mle_star.models import (
    EvaluationResult,
    MetricDirection,
    SolutionScript,
    TaskDescription,
)

_SCORE_PATTERN: re.Pattern[str] = re.compile(
    r"Final Validation Performance:\s*([\d.eE+-]+)"
)


@runtime_checkable
class ScoreFunction(Protocol):
    """Protocol for scoring a solution against a task (REQ-DM-026).

    Any callable with the signature
    ``(solution: SolutionScript, task: TaskDescription) -> EvaluationResult``
    satisfies this protocol.
    """

    def __call__(  # noqa: D102
        self, solution: SolutionScript, task: TaskDescription
    ) -> EvaluationResult: ...


def parse_score(stdout: str) -> float | None:
    r"""Extract the validation score from script stdout (REQ-DM-027).

    Matches the regex ``Final Validation Performance:\s*([\d.eE+-]+)`` and
    returns the **last** match as a float (REQ-EX-011). Returns ``None`` when
    no match is found or float conversion fails.

    Args:
        stdout: Full standard output from a solution script execution.

    Returns:
        The parsed score as a float, or None if the pattern is not found.
    """
    matches = _SCORE_PATTERN.findall(stdout)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def is_improvement(
    new_score: float,
    old_score: float,
    direction: MetricDirection,
) -> bool:
    """Check if *new_score* is strictly better than *old_score* (REQ-DM-028).

    Args:
        new_score: The candidate score to evaluate.
        old_score: The baseline score to compare against.
        direction: Whether to maximize or minimize the metric.

    Returns:
        True if *new_score* is strictly better than *old_score*.
    """
    if direction == MetricDirection.MAXIMIZE:
        return new_score > old_score
    return new_score < old_score


def beats_baseline(
    score: float,
    baseline_value: float | None,
    direction: MetricDirection,
) -> bool:
    """Check if *score* beats the external baseline threshold.

    Returns ``True`` when *baseline_value* is ``None`` (no baseline
    configured = always passes). Otherwise delegates to
    ``is_improvement`` for a strict comparison.

    Args:
        score: The candidate score to evaluate.
        baseline_value: The external baseline threshold, or ``None``.
        direction: Whether to maximize or minimize the metric.

    Returns:
        True if the score beats the baseline (or no baseline is set).
    """
    if baseline_value is None:
        return True
    return is_improvement(score, baseline_value, direction)


def is_improvement_or_equal(
    new_score: float,
    old_score: float,
    direction: MetricDirection,
) -> bool:
    """Check if *new_score* is better than or equal to *old_score* (REQ-DM-029).

    Args:
        new_score: The candidate score to evaluate.
        old_score: The baseline score to compare against.
        direction: Whether to maximize or minimize the metric.

    Returns:
        True if *new_score* is at least as good as *old_score*.
    """
    if direction == MetricDirection.MAXIMIZE:
        return new_score >= old_score
    return new_score <= old_score
