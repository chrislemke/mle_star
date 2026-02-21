"""Tests for MLE-STAR score interface and comparison functions (Task 07).

Validates the ``ScoreFunction`` protocol, ``parse_score`` regex extraction,
``is_improvement`` strict comparison, and ``is_improvement_or_equal``
non-strict comparison defined in ``src/mle_star/scoring.py``.  These tests
are written TDD-first -- the implementation does not yet exist.  They serve
as the executable specification for REQ-DM-026 through REQ-DM-029.

Refs:
    SRS 01a (Score Interface & Comparison), IMPLEMENTATION_PLAN.md Task 07.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    DataModality,
    EvaluationResult,
    MetricDirection,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
from mle_star.scoring import (
    ScoreFunction,
    is_improvement,
    is_improvement_or_equal,
    parse_score,
)
import pytest

# ---------------------------------------------------------------------------
# Helpers -- factory functions for building valid model instances
# ---------------------------------------------------------------------------


def _make_task_description(**overrides: Any) -> TaskDescription:
    """Build a valid TaskDescription with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed TaskDescription instance.
    """
    defaults: dict[str, Any] = {
        "competition_id": "spaceship-titanic",
        "task_type": TaskType.CLASSIFICATION,
        "data_modality": DataModality.TABULAR,
        "evaluation_metric": "accuracy",
        "metric_direction": MetricDirection.MAXIMIZE,
        "description": "Predict which passengers were transported.",
    }
    defaults.update(overrides)
    return TaskDescription(**defaults)


def _make_solution_script(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\ndf = pd.read_csv('train.csv')\n",
        "phase": SolutionPhase.INIT,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_evaluation_result(**overrides: Any) -> EvaluationResult:
    """Build a valid EvaluationResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed EvaluationResult instance.
    """
    defaults: dict[str, Any] = {
        "score": 0.85,
        "stdout": "Training complete.",
        "stderr": "",
        "exit_code": 0,
        "duration_seconds": 42.5,
        "is_error": False,
        "error_traceback": None,
    }
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ===========================================================================
# REQ-DM-026: ScoreFunction Protocol
# ===========================================================================


@pytest.mark.unit
class TestScoreFunctionProtocol:
    """ScoreFunction is a runtime_checkable Protocol with __call__ (REQ-DM-026)."""

    def test_score_function_is_a_protocol(self) -> None:
        """ScoreFunction is defined as a Protocol class."""
        from typing import Protocol

        assert issubclass(type(ScoreFunction), type(Protocol))

    def test_score_function_is_runtime_checkable(self) -> None:
        """ScoreFunction is decorated with @runtime_checkable."""

        # runtime_checkable protocols support isinstance() checks
        class _Conforming:
            def __call__(
                self, solution: SolutionScript, task: TaskDescription
            ) -> EvaluationResult:
                return _make_evaluation_result()

        instance = _Conforming()
        assert isinstance(instance, ScoreFunction)

    def test_conforming_callable_satisfies_protocol(self) -> None:
        """A class with the correct __call__ signature satisfies ScoreFunction."""

        class _ValidScorer:
            def __call__(
                self, solution: SolutionScript, task: TaskDescription
            ) -> EvaluationResult:
                return _make_evaluation_result(score=0.99)

        scorer = _ValidScorer()
        assert isinstance(scorer, ScoreFunction)

        # Also verify it can be called and returns EvaluationResult
        result = scorer(
            solution=_make_solution_script(),
            task=_make_task_description(),
        )
        assert isinstance(result, EvaluationResult)
        assert result.score == 0.99

    def test_non_conforming_class_does_not_satisfy_protocol(self) -> None:
        """A class without __call__ does not satisfy ScoreFunction."""

        class _NoCall:
            pass

        instance = _NoCall()
        assert not isinstance(instance, ScoreFunction)

    def test_wrong_signature_callable_does_not_satisfy_protocol(self) -> None:
        """A callable with wrong parameter names does not satisfy at runtime.

        Note: runtime_checkable only checks for the existence of __call__,
        not parameter names/types. This test documents that any callable
        passes the isinstance check -- full type safety comes from mypy.
        """

        class _WrongParams:
            def __call__(self) -> None:
                pass

        instance = _WrongParams()
        # runtime_checkable only checks __call__ exists, not signature
        assert isinstance(instance, ScoreFunction)

    def test_plain_function_satisfies_protocol(self) -> None:
        """A plain function with matching signature satisfies ScoreFunction.

        Functions have __call__ and are thus detected by runtime_checkable.
        """

        def score_fn(
            solution: SolutionScript, task: TaskDescription
        ) -> EvaluationResult:
            return _make_evaluation_result()

        assert isinstance(score_fn, ScoreFunction)


# ===========================================================================
# REQ-DM-027: parse_score -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestParseScoreHappyPath:
    """parse_score extracts float from 'Final Validation Performance:' pattern (REQ-DM-027)."""

    def test_parse_score_basic_decimal(self) -> None:
        """parse_score extracts 0.8196 from standard output."""
        result = parse_score("Final Validation Performance: 0.8196")
        assert result is not None
        assert result == pytest.approx(0.8196)

    def test_parse_score_integer_value(self) -> None:
        """parse_score extracts integer-like values (e.g., 1.0 as '1')."""
        result = parse_score("Final Validation Performance: 1")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_parse_score_zero(self) -> None:
        """parse_score extracts zero."""
        result = parse_score("Final Validation Performance: 0.0")
        assert result is not None
        assert result == pytest.approx(0.0)

    def test_parse_score_negative_value(self) -> None:
        """parse_score extracts negative values (some metrics like neg_MSE)."""
        result = parse_score("Final Validation Performance: -0.5432")
        assert result is not None
        assert result == pytest.approx(-0.5432)

    def test_parse_score_large_positive_value(self) -> None:
        """parse_score extracts large float values."""
        result = parse_score("Final Validation Performance: 99.9999")
        assert result is not None
        assert result == pytest.approx(99.9999)

    def test_parse_score_with_surrounding_text(self) -> None:
        """parse_score finds the pattern inside longer output."""
        stdout = (
            "Epoch 1/10 - loss: 0.5423\n"
            "Epoch 10/10 - loss: 0.1234\n"
            "Final Validation Performance: 0.8196\n"
            "Done.\n"
        )
        result = parse_score(stdout)
        assert result is not None
        assert result == pytest.approx(0.8196)

    def test_parse_score_with_extra_spaces(self) -> None:
        """parse_score handles extra whitespace after the colon."""
        result = parse_score("Final Validation Performance:   0.75")
        assert result is not None
        assert result == pytest.approx(0.75)

    def test_parse_score_with_positive_sign(self) -> None:
        """parse_score handles explicit positive sign."""
        result = parse_score("Final Validation Performance: +0.85")
        assert result is not None
        assert result == pytest.approx(0.85)


# ===========================================================================
# REQ-DM-027: parse_score -- Scientific Notation
# ===========================================================================


@pytest.mark.unit
class TestParseScoreScientificNotation:
    r"""parse_score handles scientific notation per regex r'[\d.eE+-]+' (REQ-DM-027)."""

    def test_parse_score_scientific_lowercase_e(self) -> None:
        """parse_score extracts values in 1.5e-3 notation."""
        result = parse_score("Final Validation Performance: 1.5e-3")
        assert result is not None
        assert result == pytest.approx(1.5e-3)

    def test_parse_score_scientific_uppercase_e(self) -> None:
        """parse_score extracts values in 2.0E+2 notation."""
        result = parse_score("Final Validation Performance: 2.0E+2")
        assert result is not None
        assert result == pytest.approx(200.0)

    def test_parse_score_scientific_negative_exponent(self) -> None:
        """parse_score extracts values with negative exponent."""
        result = parse_score("Final Validation Performance: 5e-10")
        assert result is not None
        assert result == pytest.approx(5e-10)

    def test_parse_score_scientific_positive_exponent(self) -> None:
        """parse_score extracts values with explicit positive exponent."""
        result = parse_score("Final Validation Performance: 3.14e+0")
        assert result is not None
        assert result == pytest.approx(3.14)


# ===========================================================================
# REQ-DM-027: parse_score -- No Match (returns None)
# ===========================================================================


@pytest.mark.unit
class TestParseScoreNoMatch:
    """parse_score returns None when the pattern is not found (REQ-DM-027)."""

    def test_parse_score_empty_string(self) -> None:
        """parse_score returns None for empty string."""
        result = parse_score("")
        assert result is None

    def test_parse_score_no_pattern(self) -> None:
        """parse_score returns None when pattern is absent from output."""
        result = parse_score("Training complete. Score: 0.85")
        assert result is None

    def test_parse_score_partial_pattern(self) -> None:
        """parse_score returns None for partial pattern match."""
        result = parse_score("Final Validation: 0.85")
        assert result is None

    def test_parse_score_wrong_case(self) -> None:
        """parse_score returns None for wrong case (case-sensitive regex)."""
        result = parse_score("final validation performance: 0.85")
        assert result is None

    def test_parse_score_pattern_without_number(self) -> None:
        """parse_score returns None when pattern exists but has no number."""
        result = parse_score("Final Validation Performance: NaN")
        assert result is None

    def test_parse_score_only_whitespace(self) -> None:
        """parse_score returns None for whitespace-only string."""
        result = parse_score("   \n\t  ")
        assert result is None

    def test_parse_score_similar_but_different_prefix(self) -> None:
        """parse_score returns None for similar but non-matching prefix."""
        result = parse_score("Final Test Performance: 0.85")
        assert result is None

    def test_parse_score_returns_none_type(self) -> None:
        """parse_score return type is exactly None (not 0.0 or other falsy)."""
        result = parse_score("no match here")
        assert result is None
        assert type(result) is type(None)


# ===========================================================================
# REQ-EX-011: parse_score -- Multiple Matches (returns LAST)
# ===========================================================================


@pytest.mark.unit
class TestParseScoreMultipleMatches:
    """parse_score returns the LAST match when multiple exist (REQ-EX-011)."""

    def test_parse_score_two_matches_returns_last(self) -> None:
        """Two pattern occurrences: returns the second (last) value."""
        stdout = (
            "Final Validation Performance: 0.5\nFinal Validation Performance: 0.8196\n"
        )
        result = parse_score(stdout)
        assert result is not None
        assert result == pytest.approx(0.8196)

    def test_parse_score_three_matches_returns_last(self) -> None:
        """Three pattern occurrences: returns the third (last) value."""
        stdout = (
            "Final Validation Performance: 0.5\n"
            "Final Validation Performance: 0.6\n"
            "Final Validation Performance: 0.7\n"
        )
        result = parse_score(stdout)
        assert result is not None
        assert result == pytest.approx(0.7)

    def test_parse_score_last_match_is_lower_than_first(self) -> None:
        """Returns last match even if its value is lower than earlier matches."""
        stdout = (
            "Final Validation Performance: 0.95\nFinal Validation Performance: 0.40\n"
        )
        result = parse_score(stdout)
        assert result is not None
        assert result == pytest.approx(0.40)

    def test_parse_score_multiple_matches_with_interleaved_text(self) -> None:
        """Returns last match even with interleaved non-matching text."""
        stdout = (
            "Epoch 1 done.\n"
            "Final Validation Performance: 0.60\n"
            "Epoch 2 done.\n"
            "Final Validation Performance: 0.75\n"
            "Training complete.\n"
        )
        result = parse_score(stdout)
        assert result is not None
        assert result == pytest.approx(0.75)

    def test_parse_score_acceptance_criterion_exact(self) -> None:
        """Exact acceptance criterion from Task 07 spec."""
        stdout = (
            "...Performance: 0.5\n"
            "Final Validation Performance: 0.5\n"
            "Final Validation Performance: 0.8196\n"
        )
        result = parse_score(stdout)
        assert result is not None
        assert result == pytest.approx(0.8196)


# ===========================================================================
# REQ-DM-027: parse_score -- Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestParseScoreEdgeCases:
    """parse_score edge cases for boundary inputs (REQ-DM-027)."""

    def test_parse_score_value_with_trailing_whitespace(self) -> None:
        """parse_score handles trailing whitespace after the number."""
        result = parse_score("Final Validation Performance: 0.85   \n")
        assert result is not None
        assert result == pytest.approx(0.85)

    def test_parse_score_value_with_trailing_text(self) -> None:
        """parse_score extracts value even when followed by text on same line."""
        result = parse_score("Final Validation Performance: 0.85 (accuracy)")
        assert result is not None
        assert result == pytest.approx(0.85)

    def test_parse_score_very_small_float(self) -> None:
        """parse_score extracts very small float values."""
        result = parse_score("Final Validation Performance: 0.000001")
        assert result is not None
        assert result == pytest.approx(1e-6)

    def test_parse_score_no_leading_zero(self) -> None:
        """parse_score extracts value without leading zero."""
        result = parse_score("Final Validation Performance: .5")
        assert result is not None
        assert result == pytest.approx(0.5)

    def test_parse_score_only_digits_no_decimal(self) -> None:
        """parse_score extracts integer-only value (no decimal point)."""
        result = parse_score("Final Validation Performance: 42")
        assert result is not None
        assert result == pytest.approx(42.0)

    def test_parse_score_returns_float_type(self) -> None:
        """parse_score return type is float when match is found."""
        result = parse_score("Final Validation Performance: 0.5")
        assert result is not None
        assert isinstance(result, float)


# ===========================================================================
# REQ-DM-027: parse_score -- Property-based with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestParseScorePropertyBased:
    """Property-based tests for parse_score using Hypothesis (REQ-DM-027)."""

    @given(
        score=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_formatted_score_round_trips_through_parse(self, score: float) -> None:
        """Property: a float formatted into the pattern is extracted correctly."""
        stdout = f"Final Validation Performance: {score}"
        result = parse_score(stdout)
        assert result is not None
        assert result == pytest.approx(score, rel=1e-6, abs=1e-12)

    @given(
        text=st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),  # type: ignore[arg-type]
            ),
            min_size=0,
            max_size=200,
        ),
    )
    @settings(max_examples=50)
    def test_no_pattern_always_returns_none(self, text: str) -> None:
        """Property: text without 'Final Validation Performance:' returns None.

        We filter out any text that accidentally contains the marker.
        """
        if "Final Validation Performance:" in text:
            return
        result = parse_score(text)
        assert result is None

    @given(
        first=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        last=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50)
    def test_multiple_matches_always_returns_last(
        self, first: float, last: float
    ) -> None:
        """Property: with two matches, parse_score always returns the last one."""
        stdout = (
            f"Final Validation Performance: {first}\n"
            f"Final Validation Performance: {last}\n"
        )
        result = parse_score(stdout)
        assert result is not None
        assert result == pytest.approx(last, rel=1e-6, abs=1e-12)


# ===========================================================================
# REQ-DM-028: is_improvement -- Strict Comparison
# ===========================================================================


@pytest.mark.unit
class TestIsImprovement:
    """is_improvement performs strict comparison respecting MetricDirection (REQ-DM-028)."""

    # -- Maximize direction --

    def test_maximize_higher_is_improvement(self) -> None:
        """For maximize: new > old is an improvement."""
        assert is_improvement(0.9, 0.8, MetricDirection.MAXIMIZE) is True

    def test_maximize_lower_is_not_improvement(self) -> None:
        """For maximize: new < old is not an improvement."""
        assert is_improvement(0.7, 0.8, MetricDirection.MAXIMIZE) is False

    def test_maximize_equal_is_not_improvement(self) -> None:
        """For maximize: new == old is NOT an improvement (strict)."""
        assert is_improvement(0.8, 0.8, MetricDirection.MAXIMIZE) is False

    def test_maximize_slightly_higher(self) -> None:
        """For maximize: even a tiny improvement counts."""
        assert is_improvement(0.80001, 0.8, MetricDirection.MAXIMIZE) is True

    def test_maximize_slightly_lower(self) -> None:
        """For maximize: even a tiny decrease is not an improvement."""
        assert is_improvement(0.79999, 0.8, MetricDirection.MAXIMIZE) is False

    # -- Minimize direction --

    def test_minimize_lower_is_improvement(self) -> None:
        """For minimize: new < old is an improvement."""
        assert is_improvement(0.7, 0.8, MetricDirection.MINIMIZE) is True

    def test_minimize_higher_is_not_improvement(self) -> None:
        """For minimize: new > old is not an improvement."""
        assert is_improvement(0.9, 0.8, MetricDirection.MINIMIZE) is False

    def test_minimize_equal_is_not_improvement(self) -> None:
        """For minimize: new == old is NOT an improvement (strict)."""
        assert is_improvement(0.8, 0.8, MetricDirection.MINIMIZE) is False

    def test_minimize_slightly_lower(self) -> None:
        """For minimize: even a tiny decrease counts as improvement."""
        assert is_improvement(0.79999, 0.8, MetricDirection.MINIMIZE) is True

    def test_minimize_slightly_higher(self) -> None:
        """For minimize: even a tiny increase is not an improvement."""
        assert is_improvement(0.80001, 0.8, MetricDirection.MINIMIZE) is False

    # -- Acceptance criteria from spec --

    def test_acceptance_criterion_maximize_true(self) -> None:
        """Exact acceptance criterion: is_improvement(0.9, 0.8, 'maximize') is True."""
        assert is_improvement(0.9, 0.8, MetricDirection.MAXIMIZE) is True

    def test_acceptance_criterion_minimize_false(self) -> None:
        """Exact acceptance criterion: is_improvement(0.9, 0.8, 'minimize') is False."""
        assert is_improvement(0.9, 0.8, MetricDirection.MINIMIZE) is False

    # -- Return type --

    def test_returns_bool(self) -> None:
        """is_improvement returns a bool, not an int or other truthy/falsy type."""
        result = is_improvement(0.9, 0.8, MetricDirection.MAXIMIZE)
        assert isinstance(result, bool)

    # -- Negative values --

    def test_maximize_negative_scores(self) -> None:
        """For maximize: -0.3 > -0.5, so new=-0.3, old=-0.5 is improvement."""
        assert is_improvement(-0.3, -0.5, MetricDirection.MAXIMIZE) is True

    def test_minimize_negative_scores(self) -> None:
        """For minimize: -0.5 < -0.3, so new=-0.5, old=-0.3 is improvement."""
        assert is_improvement(-0.5, -0.3, MetricDirection.MINIMIZE) is True

    # -- Zero crossing --

    def test_maximize_zero_to_positive(self) -> None:
        """For maximize: moving from 0 to positive is improvement."""
        assert is_improvement(0.1, 0.0, MetricDirection.MAXIMIZE) is True

    def test_minimize_zero_to_negative(self) -> None:
        """For minimize: moving from 0 to negative is improvement."""
        assert is_improvement(-0.1, 0.0, MetricDirection.MINIMIZE) is True

    # -- String direction values --

    def test_accepts_string_maximize(self) -> None:
        """is_improvement accepts the string 'maximize' for direction."""
        assert is_improvement(0.9, 0.8, "maximize") is True  # type: ignore[arg-type]

    def test_accepts_string_minimize(self) -> None:
        """is_improvement accepts the string 'minimize' for direction."""
        assert is_improvement(0.7, 0.8, "minimize") is True  # type: ignore[arg-type]


# ===========================================================================
# REQ-DM-029: is_improvement_or_equal -- Non-strict Comparison
# ===========================================================================


@pytest.mark.unit
class TestIsImprovementOrEqual:
    """is_improvement_or_equal performs non-strict comparison (REQ-DM-029)."""

    # -- Maximize direction --

    def test_maximize_higher_is_improvement(self) -> None:
        """For maximize: new > old is an improvement."""
        assert is_improvement_or_equal(0.9, 0.8, MetricDirection.MAXIMIZE) is True

    def test_maximize_lower_is_not_improvement(self) -> None:
        """For maximize: new < old is not an improvement."""
        assert is_improvement_or_equal(0.7, 0.8, MetricDirection.MAXIMIZE) is False

    def test_maximize_equal_is_true(self) -> None:
        """For maximize: new == old returns True (non-strict)."""
        assert is_improvement_or_equal(0.8, 0.8, MetricDirection.MAXIMIZE) is True

    # -- Minimize direction --

    def test_minimize_lower_is_improvement(self) -> None:
        """For minimize: new < old is an improvement."""
        assert is_improvement_or_equal(0.7, 0.8, MetricDirection.MINIMIZE) is True

    def test_minimize_higher_is_not_improvement(self) -> None:
        """For minimize: new > old is not an improvement."""
        assert is_improvement_or_equal(0.9, 0.8, MetricDirection.MINIMIZE) is False

    def test_minimize_equal_is_true(self) -> None:
        """For minimize: new == old returns True (non-strict)."""
        assert is_improvement_or_equal(0.8, 0.8, MetricDirection.MINIMIZE) is True

    # -- Acceptance criterion from spec --

    def test_acceptance_criterion_equal_maximize(self) -> None:
        """Exact acceptance criterion: is_improvement_or_equal(0.8, 0.8, 'maximize') is True."""
        assert is_improvement_or_equal(0.8, 0.8, MetricDirection.MAXIMIZE) is True

    # -- Return type --

    def test_returns_bool(self) -> None:
        """is_improvement_or_equal returns a bool."""
        result = is_improvement_or_equal(0.9, 0.8, MetricDirection.MAXIMIZE)
        assert isinstance(result, bool)

    # -- Negative values --

    def test_maximize_equal_negative_scores(self) -> None:
        """For maximize: equal negative scores returns True."""
        assert is_improvement_or_equal(-0.5, -0.5, MetricDirection.MAXIMIZE) is True

    def test_minimize_equal_negative_scores(self) -> None:
        """For minimize: equal negative scores returns True."""
        assert is_improvement_or_equal(-0.5, -0.5, MetricDirection.MINIMIZE) is True

    # -- String direction values --

    def test_accepts_string_maximize(self) -> None:
        """is_improvement_or_equal accepts the string 'maximize' for direction."""
        assert is_improvement_or_equal(0.8, 0.8, "maximize") is True  # type: ignore[arg-type]

    def test_accepts_string_minimize(self) -> None:
        """is_improvement_or_equal accepts the string 'minimize' for direction."""
        assert is_improvement_or_equal(0.8, 0.8, "minimize") is True  # type: ignore[arg-type]


# ===========================================================================
# Consistency: is_improvement vs is_improvement_or_equal
# ===========================================================================


@pytest.mark.unit
class TestComparisonConsistency:
    """is_improvement and is_improvement_or_equal are consistent with each other."""

    @pytest.mark.parametrize(
        "direction", [MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]
    )
    def test_strict_true_implies_nonstrict_true(
        self, direction: MetricDirection
    ) -> None:
        """If is_improvement is True, is_improvement_or_equal must also be True."""
        # Pick values where strict is True
        if direction == MetricDirection.MAXIMIZE:
            new, old = 0.9, 0.8
        else:
            new, old = 0.7, 0.8
        assert is_improvement(new, old, direction) is True
        assert is_improvement_or_equal(new, old, direction) is True

    @pytest.mark.parametrize(
        "direction", [MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]
    )
    def test_equal_values_differ_between_strict_and_nonstrict(
        self, direction: MetricDirection
    ) -> None:
        """Equal values: strict returns False, non-strict returns True."""
        assert is_improvement(0.5, 0.5, direction) is False
        assert is_improvement_or_equal(0.5, 0.5, direction) is True

    @pytest.mark.parametrize(
        "direction", [MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]
    )
    def test_nonstrict_false_implies_strict_false(
        self, direction: MetricDirection
    ) -> None:
        """If is_improvement_or_equal is False, is_improvement must also be False."""
        # Pick values where non-strict is False
        if direction == MetricDirection.MAXIMIZE:
            new, old = 0.7, 0.8
        else:
            new, old = 0.9, 0.8
        assert is_improvement_or_equal(new, old, direction) is False
        assert is_improvement(new, old, direction) is False


# ===========================================================================
# Property-based tests: Comparison functions with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestComparisonPropertyBased:
    """Property-based tests for is_improvement and is_improvement_or_equal."""

    @given(
        a=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        b=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50)
    def test_maximize_a_greater_than_b_is_improvement(self, a: float, b: float) -> None:
        """Property: for maximize, a > b implies is_improvement(a, b) is True."""
        if a > b:
            assert is_improvement(a, b, MetricDirection.MAXIMIZE) is True

    @given(
        a=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        b=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50)
    def test_minimize_a_less_than_b_is_improvement(self, a: float, b: float) -> None:
        """Property: for minimize, a < b implies is_improvement(a, b) is True."""
        if a < b:
            assert is_improvement(a, b, MetricDirection.MINIMIZE) is True

    @given(
        x=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_equal_values_strict_always_false(
        self, x: float, direction: MetricDirection
    ) -> None:
        """Property: is_improvement(x, x, direction) is always False."""
        assert is_improvement(x, x, direction) is False

    @given(
        x=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_equal_values_nonstrict_always_true(
        self, x: float, direction: MetricDirection
    ) -> None:
        """Property: is_improvement_or_equal(x, x, direction) is always True."""
        assert is_improvement_or_equal(x, x, direction) is True

    @given(
        a=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        b=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_strict_implies_nonstrict(
        self, a: float, b: float, direction: MetricDirection
    ) -> None:
        """Property: is_improvement(a, b, d) => is_improvement_or_equal(a, b, d)."""
        if is_improvement(a, b, direction):
            assert is_improvement_or_equal(a, b, direction) is True

    @given(
        a=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        b=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_not_nonstrict_implies_not_strict(
        self, a: float, b: float, direction: MetricDirection
    ) -> None:
        """Property: NOT is_improvement_or_equal(a, b, d) => NOT is_improvement(a, b, d)."""
        if not is_improvement_or_equal(a, b, direction):
            assert is_improvement(a, b, direction) is False

    @given(
        a=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        b=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50)
    def test_maximize_and_minimize_are_inverses_for_strict_inequality(
        self, a: float, b: float
    ) -> None:
        """Property: for a != b, maximize and minimize give opposite results.

        When a != b: is_improvement(a, b, maximize) != is_improvement(a, b, minimize).
        """
        if a != b:
            max_result = is_improvement(a, b, MetricDirection.MAXIMIZE)
            min_result = is_improvement(a, b, MetricDirection.MINIMIZE)
            assert max_result != min_result

    @given(
        a=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        b=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_is_improvement_returns_bool_type(
        self, a: float, b: float, direction: MetricDirection
    ) -> None:
        """Property: is_improvement always returns a bool instance."""
        result = is_improvement(a, b, direction)
        assert isinstance(result, bool)

    @given(
        a=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        b=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        direction=st.sampled_from([MetricDirection.MAXIMIZE, MetricDirection.MINIMIZE]),
    )
    @settings(max_examples=50)
    def test_is_improvement_or_equal_returns_bool_type(
        self, a: float, b: float, direction: MetricDirection
    ) -> None:
        """Property: is_improvement_or_equal always returns a bool instance."""
        result = is_improvement_or_equal(a, b, direction)
        assert isinstance(result, bool)
