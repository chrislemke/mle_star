"""Tests for MLE-STAR iteration loop data models (Task 10).

Validates RefinementAttempt, EnsembleAttempt, and InnerLoopResult Pydantic
models defined in ``src/mle_star/models.py``.  These tests are written
TDD-first -- the implementation does not yet exist.  They serve as the
executable specification for REQ-DM-039 and REQ-DM-041 (iteration models).

Refs:
    SRS 01b (Iteration Loop Models), IMPLEMENTATION_PLAN.md Task 10.
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    EnsembleAttempt,
    InnerLoopResult,
    RefinementAttempt,
    SolutionPhase,
    SolutionScript,
)
from pydantic import ValidationError
import pytest

# ---------------------------------------------------------------------------
# Helpers -- factory functions for building valid model instances
# ---------------------------------------------------------------------------


def _make_solution_script(**overrides: Any) -> SolutionScript:
    """Build a valid SolutionScript with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed SolutionScript instance.
    """
    defaults: dict[str, Any] = {
        "content": "import pandas as pd\ndf = pd.read_csv('train.csv')\n",
        "phase": SolutionPhase.REFINED,
    }
    defaults.update(overrides)
    return SolutionScript(**defaults)


def _make_refinement_attempt(**overrides: Any) -> RefinementAttempt:
    """Build a valid RefinementAttempt with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed RefinementAttempt instance.
    """
    defaults: dict[str, Any] = {
        "plan": "Increase regularization strength to reduce overfitting.",
        "score": 0.87,
        "code_block": "model = LogisticRegression(C=0.1)",
        "was_improvement": True,
    }
    defaults.update(overrides)
    return RefinementAttempt(**defaults)


def _make_ensemble_attempt(**overrides: Any) -> EnsembleAttempt:
    """Build a valid EnsembleAttempt with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed EnsembleAttempt instance.
    """
    defaults: dict[str, Any] = {
        "plan": "Weighted average of top-2 solutions.",
        "score": 0.92,
        "solution": _make_solution_script(phase=SolutionPhase.ENSEMBLE),
    }
    defaults.update(overrides)
    return EnsembleAttempt(**defaults)


def _make_inner_loop_result(**overrides: Any) -> InnerLoopResult:
    """Build a valid InnerLoopResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed InnerLoopResult instance.
    """
    defaults: dict[str, Any] = {
        "best_solution": _make_solution_script(phase=SolutionPhase.REFINED),
        "best_score": 0.90,
        "attempts": [_make_refinement_attempt()],
        "improved": True,
    }
    defaults.update(overrides)
    return InnerLoopResult(**defaults)


# ===========================================================================
# RefinementAttempt -- construction, fields, frozen, JSON schema
# ===========================================================================


@pytest.mark.unit
class TestRefinementAttemptConstruction:
    """RefinementAttempt has correct required fields (REQ-DM-039)."""

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all fields succeeds and stores correct values."""
        attempt = _make_refinement_attempt()
        assert attempt.plan == "Increase regularization strength to reduce overfitting."
        assert attempt.score == 0.87
        assert attempt.code_block == "model = LogisticRegression(C=0.1)"
        assert attempt.was_improvement is True

    def test_plan_is_required_string(self) -> None:
        """Plan field holds a string value describing the refinement strategy."""
        attempt = _make_refinement_attempt(plan="Try XGBoost instead of RF.")
        assert isinstance(attempt.plan, str)
        assert attempt.plan == "Try XGBoost instead of RF."

    def test_score_accepts_float(self) -> None:
        """Score field accepts a float value."""
        attempt = _make_refinement_attempt(score=0.95)
        assert attempt.score == 0.95

    def test_score_accepts_none(self) -> None:
        """Score field accepts None (evaluation may have failed)."""
        attempt = _make_refinement_attempt(score=None)
        assert attempt.score is None

    def test_score_accepts_negative_float(self) -> None:
        """Score field accepts negative floats (some metrics are negative)."""
        attempt = _make_refinement_attempt(score=-1.5)
        assert attempt.score == -1.5

    def test_score_accepts_zero(self) -> None:
        """Score field accepts zero."""
        attempt = _make_refinement_attempt(score=0.0)
        assert attempt.score == 0.0

    def test_code_block_is_required_string(self) -> None:
        """Code block field holds the exact code string."""
        attempt = _make_refinement_attempt(code_block="x = train(model, data)")
        assert isinstance(attempt.code_block, str)
        assert attempt.code_block == "x = train(model, data)"

    def test_code_block_accepts_multiline(self) -> None:
        """Code block field accepts multiline code strings."""
        block = (
            "model = XGBClassifier()\nmodel.fit(X, y)\npreds = model.predict(X_test)"
        )
        attempt = _make_refinement_attempt(code_block=block)
        assert attempt.code_block == block
        assert "\n" in attempt.code_block

    def test_was_improvement_is_required_bool(self) -> None:
        """was_improvement field holds a boolean value."""
        attempt_true = _make_refinement_attempt(was_improvement=True)
        assert isinstance(attempt_true.was_improvement, bool)
        assert attempt_true.was_improvement is True

        attempt_false = _make_refinement_attempt(was_improvement=False)
        assert isinstance(attempt_false.was_improvement, bool)
        assert attempt_false.was_improvement is False

    def test_was_improvement_true_with_good_score(self) -> None:
        """An attempt that improved the score records was_improvement=True."""
        attempt = _make_refinement_attempt(score=0.95, was_improvement=True)
        assert attempt.was_improvement is True
        assert attempt.score == 0.95

    def test_was_improvement_false_with_bad_score(self) -> None:
        """An attempt that worsened the score records was_improvement=False."""
        attempt = _make_refinement_attempt(score=0.70, was_improvement=False)
        assert attempt.was_improvement is False
        assert attempt.score == 0.70

    def test_was_improvement_false_with_none_score(self) -> None:
        """A failed evaluation records was_improvement=False."""
        attempt = _make_refinement_attempt(score=None, was_improvement=False)
        assert attempt.was_improvement is False
        assert attempt.score is None

    def test_empty_plan_string_accepted(self) -> None:
        """Empty string is a valid plan value."""
        attempt = _make_refinement_attempt(plan="")
        assert attempt.plan == ""

    def test_empty_code_block_string_accepted(self) -> None:
        """Empty string is a valid code_block value."""
        attempt = _make_refinement_attempt(code_block="")
        assert attempt.code_block == ""


@pytest.mark.unit
class TestRefinementAttemptRequiredFields:
    """RefinementAttempt raises ValidationError when required fields are missing."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "plan",
            "code_block",
            "was_improvement",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "plan": "Refine model",
            "score": 0.85,
            "code_block": "model.fit(X, y)",
            "was_improvement": True,
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            RefinementAttempt(**all_fields)

    def test_score_can_be_omitted_defaults_to_none(self) -> None:
        """Omitting score should default to None (optional field).

        Score is typed as ``float | None`` and defaults to None when
        the evaluation failed or was not run.
        """
        attempt = RefinementAttempt(
            plan="Refine model",
            code_block="model.fit(X, y)",
            was_improvement=False,
        )
        assert attempt.score is None


@pytest.mark.unit
class TestRefinementAttemptFrozen:
    """RefinementAttempt is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_plan(self) -> None:
        """Assignment to plan raises an error."""
        attempt = _make_refinement_attempt()
        with pytest.raises(ValidationError):
            attempt.plan = "changed"  # type: ignore[misc]

    def test_cannot_mutate_score(self) -> None:
        """Assignment to score raises an error."""
        attempt = _make_refinement_attempt()
        with pytest.raises(ValidationError):
            attempt.score = 0.99  # type: ignore[misc]

    def test_cannot_mutate_code_block(self) -> None:
        """Assignment to code_block raises an error."""
        attempt = _make_refinement_attempt()
        with pytest.raises(ValidationError):
            attempt.code_block = "new code"  # type: ignore[misc]

    def test_cannot_mutate_was_improvement(self) -> None:
        """Assignment to was_improvement raises an error."""
        attempt = _make_refinement_attempt()
        with pytest.raises(ValidationError):
            attempt.was_improvement = False  # type: ignore[misc]


@pytest.mark.unit
class TestRefinementAttemptJsonSchema:
    """RefinementAttempt produces valid JSON via model_json_schema() (REQ-DM-041)."""

    def test_model_json_schema_returns_dict(self) -> None:
        """model_json_schema() returns a dict."""
        schema = RefinementAttempt.model_json_schema()
        assert isinstance(schema, dict)

    def test_model_json_schema_is_valid_json(self) -> None:
        """model_json_schema() can be serialized to valid JSON."""
        schema = RefinementAttempt.model_json_schema()
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_model_json_schema_contains_expected_properties(self) -> None:
        """model_json_schema() includes all field names in properties."""
        schema = RefinementAttempt.model_json_schema()
        props = schema.get("properties", {})
        assert "plan" in props
        assert "score" in props
        assert "code_block" in props
        assert "was_improvement" in props

    def test_model_json_schema_has_title(self) -> None:
        """model_json_schema() has a title matching the model name."""
        schema = RefinementAttempt.model_json_schema()
        assert schema.get("title") == "RefinementAttempt"

    def test_model_json_schema_has_type_object(self) -> None:
        """model_json_schema() declares type as object."""
        schema = RefinementAttempt.model_json_schema()
        assert schema.get("type") == "object"


@pytest.mark.unit
class TestRefinementAttemptSerialization:
    """RefinementAttempt supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_refinement_attempt()
        json_str = original.model_dump_json()
        restored = RefinementAttempt.model_validate_json(json_str)

        assert restored.plan == original.plan
        assert restored.score == original.score
        assert restored.code_block == original.code_block
        assert restored.was_improvement == original.was_improvement

    def test_round_trip_with_none_score(self) -> None:
        """Round-trip preserves None score."""
        original = _make_refinement_attempt(score=None)
        json_str = original.model_dump_json()
        restored = RefinementAttempt.model_validate_json(json_str)
        assert restored.score is None

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_refinement_attempt()
        restored = RefinementAttempt.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        attempt = _make_refinement_attempt()
        parsed = json.loads(attempt.model_dump_json())
        assert isinstance(parsed, dict)
        assert "plan" in parsed
        assert "score" in parsed
        assert "code_block" in parsed
        assert "was_improvement" in parsed


# ===========================================================================
# EnsembleAttempt -- construction, fields, frozen, JSON schema
# ===========================================================================


@pytest.mark.unit
class TestEnsembleAttemptConstruction:
    """EnsembleAttempt has correct required fields (REQ-DM-039)."""

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all fields succeeds and stores correct values."""
        attempt = _make_ensemble_attempt()
        assert attempt.plan == "Weighted average of top-2 solutions."
        assert attempt.score == 0.92
        assert isinstance(attempt.solution, SolutionScript)

    def test_plan_is_required_string(self) -> None:
        """Plan field holds a string value describing the ensemble strategy."""
        attempt = _make_ensemble_attempt(plan="Stacking with logistic meta-learner.")
        assert isinstance(attempt.plan, str)
        assert attempt.plan == "Stacking with logistic meta-learner."

    def test_score_accepts_float(self) -> None:
        """Score field accepts a float value."""
        attempt = _make_ensemble_attempt(score=0.95)
        assert attempt.score == 0.95

    def test_score_accepts_none(self) -> None:
        """Score field accepts None (ensemble evaluation may have failed)."""
        attempt = _make_ensemble_attempt(score=None)
        assert attempt.score is None

    def test_score_accepts_negative_float(self) -> None:
        """Score field accepts negative floats."""
        attempt = _make_ensemble_attempt(score=-2.5)
        assert attempt.score == -2.5

    def test_score_accepts_zero(self) -> None:
        """Score field accepts zero."""
        attempt = _make_ensemble_attempt(score=0.0)
        assert attempt.score == 0.0

    def test_solution_is_solution_script(self) -> None:
        """Solution field holds a SolutionScript instance."""
        sol = _make_solution_script(
            content="ensemble code", phase=SolutionPhase.ENSEMBLE
        )
        attempt = _make_ensemble_attempt(solution=sol)
        assert isinstance(attempt.solution, SolutionScript)
        assert attempt.solution.content == "ensemble code"

    def test_solution_with_different_phases(self) -> None:
        """Solution field accepts SolutionScript with any phase."""
        for phase in SolutionPhase:
            sol = _make_solution_script(content=f"code_{phase}", phase=phase)
            attempt = _make_ensemble_attempt(solution=sol)
            assert attempt.solution.phase == phase

    def test_empty_plan_string_accepted(self) -> None:
        """Empty string is a valid plan value."""
        attempt = _make_ensemble_attempt(plan="")
        assert attempt.plan == ""


@pytest.mark.unit
class TestEnsembleAttemptRequiredFields:
    """EnsembleAttempt raises ValidationError when required fields are missing."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "plan",
            "solution",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "plan": "Ensemble strategy",
            "score": 0.92,
            "solution": _make_solution_script(phase=SolutionPhase.ENSEMBLE),
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            EnsembleAttempt(**all_fields)

    def test_score_can_be_omitted_defaults_to_none(self) -> None:
        """Omitting score should default to None (optional field)."""
        attempt = EnsembleAttempt(
            plan="Ensemble strategy",
            solution=_make_solution_script(phase=SolutionPhase.ENSEMBLE),
        )
        assert attempt.score is None


@pytest.mark.unit
class TestEnsembleAttemptFrozen:
    """EnsembleAttempt is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_plan(self) -> None:
        """Assignment to plan raises an error."""
        attempt = _make_ensemble_attempt()
        with pytest.raises(ValidationError):
            attempt.plan = "changed"  # type: ignore[misc]

    def test_cannot_mutate_score(self) -> None:
        """Assignment to score raises an error."""
        attempt = _make_ensemble_attempt()
        with pytest.raises(ValidationError):
            attempt.score = 0.99  # type: ignore[misc]

    def test_cannot_mutate_solution(self) -> None:
        """Assignment to solution raises an error."""
        attempt = _make_ensemble_attempt()
        with pytest.raises(ValidationError):
            attempt.solution = _make_solution_script()  # type: ignore[misc]


@pytest.mark.unit
class TestEnsembleAttemptJsonSchema:
    """EnsembleAttempt produces valid JSON via model_json_schema() (REQ-DM-041)."""

    def test_model_json_schema_returns_dict(self) -> None:
        """model_json_schema() returns a dict."""
        schema = EnsembleAttempt.model_json_schema()
        assert isinstance(schema, dict)

    def test_model_json_schema_is_valid_json(self) -> None:
        """model_json_schema() can be serialized to valid JSON."""
        schema = EnsembleAttempt.model_json_schema()
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_model_json_schema_contains_expected_properties(self) -> None:
        """model_json_schema() includes all field names in properties."""
        schema = EnsembleAttempt.model_json_schema()
        props = schema.get("properties", {})
        assert "plan" in props
        assert "score" in props
        assert "solution" in props

    def test_model_json_schema_has_title(self) -> None:
        """model_json_schema() has a title matching the model name."""
        schema = EnsembleAttempt.model_json_schema()
        assert schema.get("title") == "EnsembleAttempt"

    def test_model_json_schema_has_type_object(self) -> None:
        """model_json_schema() declares type as object."""
        schema = EnsembleAttempt.model_json_schema()
        assert schema.get("type") == "object"


@pytest.mark.unit
class TestEnsembleAttemptSerialization:
    """EnsembleAttempt supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_ensemble_attempt()
        json_str = original.model_dump_json()
        restored = EnsembleAttempt.model_validate_json(json_str)

        assert restored.plan == original.plan
        assert restored.score == original.score
        assert restored.solution.content == original.solution.content
        assert restored.solution.phase == original.solution.phase

    def test_round_trip_with_none_score(self) -> None:
        """Round-trip preserves None score."""
        original = _make_ensemble_attempt(score=None)
        json_str = original.model_dump_json()
        restored = EnsembleAttempt.model_validate_json(json_str)
        assert restored.score is None

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_ensemble_attempt()
        restored = EnsembleAttempt.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        attempt = _make_ensemble_attempt()
        parsed = json.loads(attempt.model_dump_json())
        assert isinstance(parsed, dict)
        assert "plan" in parsed
        assert "score" in parsed
        assert "solution" in parsed


# ===========================================================================
# InnerLoopResult -- construction, fields, frozen, JSON schema
# ===========================================================================


@pytest.mark.unit
class TestInnerLoopResultConstruction:
    """InnerLoopResult has correct required fields (REQ-DM-039)."""

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all fields succeeds and stores correct values."""
        result = _make_inner_loop_result()
        assert isinstance(result.best_solution, SolutionScript)
        assert result.best_score == 0.90
        assert len(result.attempts) == 1
        assert result.improved is True

    def test_best_solution_is_solution_script(self) -> None:
        """best_solution holds a SolutionScript instance."""
        sol = _make_solution_script(content="best code", phase=SolutionPhase.REFINED)
        result = _make_inner_loop_result(best_solution=sol)
        assert isinstance(result.best_solution, SolutionScript)
        assert result.best_solution.content == "best code"
        assert result.best_solution.phase == SolutionPhase.REFINED

    def test_best_score_is_float(self) -> None:
        """best_score holds a float value."""
        result = _make_inner_loop_result(best_score=0.95)
        assert isinstance(result.best_score, float)
        assert result.best_score == 0.95

    def test_best_score_accepts_negative(self) -> None:
        """best_score accepts negative values (some metrics are negative)."""
        result = _make_inner_loop_result(best_score=-0.5)
        assert result.best_score == -0.5

    def test_best_score_accepts_zero(self) -> None:
        """best_score accepts zero."""
        result = _make_inner_loop_result(best_score=0.0)
        assert result.best_score == 0.0

    def test_attempts_is_list_of_refinement_attempts(self) -> None:
        """Attempts holds a list of RefinementAttempt instances."""
        attempts = [
            _make_refinement_attempt(plan="Plan A", score=0.85, was_improvement=False),
            _make_refinement_attempt(plan="Plan B", score=0.90, was_improvement=True),
            _make_refinement_attempt(plan="Plan C", score=0.88, was_improvement=False),
        ]
        result = _make_inner_loop_result(attempts=attempts)
        assert len(result.attempts) == 3
        assert result.attempts[0].plan == "Plan A"
        assert result.attempts[1].plan == "Plan B"
        assert result.attempts[2].plan == "Plan C"

    def test_attempts_contains_refinement_attempt_instances(self) -> None:
        """Each element in attempts is a RefinementAttempt instance."""
        result = _make_inner_loop_result()
        for attempt in result.attempts:
            assert isinstance(attempt, RefinementAttempt)

    def test_attempts_empty_list_accepted(self) -> None:
        """Empty list for attempts is valid (no attempts were made)."""
        result = _make_inner_loop_result(attempts=[])
        assert len(result.attempts) == 0

    def test_improved_is_required_bool(self) -> None:
        """Improved field holds a boolean value."""
        result_true = _make_inner_loop_result(improved=True)
        assert isinstance(result_true.improved, bool)
        assert result_true.improved is True

        result_false = _make_inner_loop_result(improved=False)
        assert isinstance(result_false.improved, bool)
        assert result_false.improved is False

    def test_improved_true_indicates_strict_improvement(self) -> None:
        """improved=True means best_score is strictly better than input.

        Uses is_improvement (strict), not is_improvement_or_equal.
        """
        result = _make_inner_loop_result(best_score=0.92, improved=True)
        assert result.improved is True
        assert result.best_score == 0.92

    def test_improved_false_indicates_no_strict_improvement(self) -> None:
        """improved=False means best_score was not strictly better than input."""
        result = _make_inner_loop_result(best_score=0.85, improved=False)
        assert result.improved is False
        assert result.best_score == 0.85

    def test_multiple_attempts_with_mixed_improvement(self) -> None:
        """Result with multiple attempts having mixed improvement flags."""
        attempts = [
            _make_refinement_attempt(score=0.86, was_improvement=True),
            _make_refinement_attempt(score=0.84, was_improvement=False),
            _make_refinement_attempt(score=0.90, was_improvement=True),
            _make_refinement_attempt(score=None, was_improvement=False),
        ]
        result = _make_inner_loop_result(
            best_score=0.90, attempts=attempts, improved=True
        )
        assert result.best_score == 0.90
        assert len(result.attempts) == 4
        assert result.improved is True


@pytest.mark.unit
class TestInnerLoopResultRequiredFields:
    """InnerLoopResult raises ValidationError when required fields are missing."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "best_solution",
            "best_score",
            "attempts",
            "improved",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "best_solution": _make_solution_script(phase=SolutionPhase.REFINED),
            "best_score": 0.90,
            "attempts": [_make_refinement_attempt()],
            "improved": True,
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            InnerLoopResult(**all_fields)


@pytest.mark.unit
class TestInnerLoopResultFrozen:
    """InnerLoopResult is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_best_solution(self) -> None:
        """Assignment to best_solution raises an error."""
        result = _make_inner_loop_result()
        with pytest.raises(ValidationError):
            result.best_solution = _make_solution_script()  # type: ignore[misc]

    def test_cannot_mutate_best_score(self) -> None:
        """Assignment to best_score raises an error."""
        result = _make_inner_loop_result()
        with pytest.raises(ValidationError):
            result.best_score = 0.99  # type: ignore[misc]

    def test_cannot_mutate_attempts(self) -> None:
        """Assignment to attempts raises an error."""
        result = _make_inner_loop_result()
        with pytest.raises(ValidationError):
            result.attempts = []  # type: ignore[misc]

    def test_cannot_mutate_improved(self) -> None:
        """Assignment to improved raises an error."""
        result = _make_inner_loop_result()
        with pytest.raises(ValidationError):
            result.improved = False  # type: ignore[misc]


@pytest.mark.unit
class TestInnerLoopResultJsonSchema:
    """InnerLoopResult produces valid JSON via model_json_schema() (REQ-DM-041)."""

    def test_model_json_schema_returns_dict(self) -> None:
        """model_json_schema() returns a dict."""
        schema = InnerLoopResult.model_json_schema()
        assert isinstance(schema, dict)

    def test_model_json_schema_is_valid_json(self) -> None:
        """model_json_schema() can be serialized to valid JSON."""
        schema = InnerLoopResult.model_json_schema()
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_model_json_schema_contains_expected_properties(self) -> None:
        """model_json_schema() includes all field names in properties."""
        schema = InnerLoopResult.model_json_schema()
        props = schema.get("properties", {})
        assert "best_solution" in props
        assert "best_score" in props
        assert "attempts" in props
        assert "improved" in props

    def test_model_json_schema_has_title(self) -> None:
        """model_json_schema() has a title matching the model name."""
        schema = InnerLoopResult.model_json_schema()
        assert schema.get("title") == "InnerLoopResult"

    def test_model_json_schema_has_type_object(self) -> None:
        """model_json_schema() declares type as object."""
        schema = InnerLoopResult.model_json_schema()
        assert schema.get("type") == "object"


@pytest.mark.unit
class TestInnerLoopResultSerialization:
    """InnerLoopResult supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_inner_loop_result()
        json_str = original.model_dump_json()
        restored = InnerLoopResult.model_validate_json(json_str)

        assert restored.best_solution.content == original.best_solution.content
        assert restored.best_score == original.best_score
        assert len(restored.attempts) == len(original.attempts)
        assert restored.attempts[0].plan == original.attempts[0].plan
        assert restored.improved == original.improved

    def test_round_trip_with_empty_attempts(self) -> None:
        """Round-trip preserves empty attempts list."""
        original = _make_inner_loop_result(attempts=[])
        json_str = original.model_dump_json()
        restored = InnerLoopResult.model_validate_json(json_str)
        assert len(restored.attempts) == 0

    def test_round_trip_with_multiple_attempts(self) -> None:
        """Round-trip preserves multiple RefinementAttempt entries."""
        attempts = [
            _make_refinement_attempt(plan=f"Plan {i}", score=0.80 + i * 0.03)
            for i in range(4)
        ]
        original = _make_inner_loop_result(attempts=attempts)
        json_str = original.model_dump_json()
        restored = InnerLoopResult.model_validate_json(json_str)
        assert len(restored.attempts) == 4
        for i in range(4):
            assert restored.attempts[i].plan == f"Plan {i}"
            assert restored.attempts[i].score == pytest.approx(0.80 + i * 0.03)

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_inner_loop_result()
        restored = InnerLoopResult.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        result = _make_inner_loop_result()
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed, dict)
        assert "best_solution" in parsed
        assert "best_score" in parsed
        assert "attempts" in parsed
        assert "improved" in parsed

    def test_round_trip_with_none_scores_in_attempts(self) -> None:
        """Round-trip preserves None scores within RefinementAttempt entries."""
        attempts = [
            _make_refinement_attempt(score=None, was_improvement=False),
            _make_refinement_attempt(score=0.85, was_improvement=True),
            _make_refinement_attempt(score=None, was_improvement=False),
        ]
        original = _make_inner_loop_result(attempts=attempts)
        json_str = original.model_dump_json()
        restored = InnerLoopResult.model_validate_json(json_str)
        assert restored.attempts[0].score is None
        assert restored.attempts[1].score == 0.85
        assert restored.attempts[2].score is None


# ===========================================================================
# Cross-Cutting Constraints
# ===========================================================================


@pytest.mark.unit
class TestCrossCuttingConstraints:
    """Cross-cutting constraints for iteration models and existing models."""

    # -- All frozen models raise on attribute assignment --

    def test_refinement_attempt_is_frozen(self) -> None:
        """RefinementAttempt raises ValidationError on attribute mutation."""
        attempt = _make_refinement_attempt()
        with pytest.raises(ValidationError):
            attempt.plan = "changed"  # type: ignore[misc]

    def test_ensemble_attempt_is_frozen(self) -> None:
        """EnsembleAttempt raises ValidationError on attribute mutation."""
        attempt = _make_ensemble_attempt()
        with pytest.raises(ValidationError):
            attempt.plan = "changed"  # type: ignore[misc]

    def test_inner_loop_result_is_frozen(self) -> None:
        """InnerLoopResult raises ValidationError on attribute mutation."""
        result = _make_inner_loop_result()
        with pytest.raises(ValidationError):
            result.best_score = 0.99  # type: ignore[misc]

    # -- SolutionScript is mutable (not frozen) --

    def test_solution_script_is_mutable(self) -> None:
        """SolutionScript allows attribute mutation (frozen=False)."""
        sol = _make_solution_script()
        sol.score = 0.99
        assert sol.score == 0.99

    def test_solution_script_score_can_be_updated_after_construction(self) -> None:
        """SolutionScript.score can be set after creation (evaluation workflow)."""
        sol = _make_solution_script(score=None)
        assert sol.score is None
        sol.score = 0.85
        assert sol.score == 0.85

    # -- All models defined in mle_star/models.py --

    def test_refinement_attempt_defined_in_models_module(self) -> None:
        """RefinementAttempt is defined in mle_star.models."""
        assert RefinementAttempt.__module__ == "mle_star.models"

    def test_ensemble_attempt_defined_in_models_module(self) -> None:
        """EnsembleAttempt is defined in mle_star.models."""
        assert EnsembleAttempt.__module__ == "mle_star.models"

    def test_inner_loop_result_defined_in_models_module(self) -> None:
        """InnerLoopResult is defined in mle_star.models."""
        assert InnerLoopResult.__module__ == "mle_star.models"

    # -- Public types re-exported from mle_star/__init__.py --

    def test_refinement_attempt_reexported_from_init(self) -> None:
        """RefinementAttempt is accessible from mle_star package root."""
        import mle_star

        assert hasattr(mle_star, "RefinementAttempt")
        assert mle_star.RefinementAttempt is RefinementAttempt  # type: ignore[attr-defined]

    def test_ensemble_attempt_reexported_from_init(self) -> None:
        """EnsembleAttempt is accessible from mle_star package root."""
        import mle_star

        assert hasattr(mle_star, "EnsembleAttempt")
        assert mle_star.EnsembleAttempt is EnsembleAttempt  # type: ignore[attr-defined]

    def test_inner_loop_result_reexported_from_init(self) -> None:
        """InnerLoopResult is accessible from mle_star package root."""
        import mle_star

        assert hasattr(mle_star, "InnerLoopResult")
        assert mle_star.InnerLoopResult is InnerLoopResult  # type: ignore[attr-defined]

    # -- All structured output schemas produce valid JSON via model_json_schema() --

    @pytest.mark.parametrize(
        "model_cls",
        [RefinementAttempt, EnsembleAttempt, InnerLoopResult],
        ids=["RefinementAttempt", "EnsembleAttempt", "InnerLoopResult"],
    )
    def test_model_json_schema_produces_valid_json(
        self,
        model_cls: type,
    ) -> None:
        """Every iteration model produces valid JSON schema (REQ-DM-041)."""
        schema = model_cls.model_json_schema()  # type: ignore[attr-defined]
        json_str = json.dumps(schema)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "properties" in parsed
        assert "title" in parsed


# ===========================================================================
# Model Composition Tests
# ===========================================================================


@pytest.mark.unit
class TestIterationModelComposition:
    """Iteration models correctly compose nested Pydantic models."""

    def test_inner_loop_result_contains_refinement_attempts(self) -> None:
        """InnerLoopResult.attempts are RefinementAttempt instances."""
        result = _make_inner_loop_result()
        for attempt in result.attempts:
            assert isinstance(attempt, RefinementAttempt)

    def test_inner_loop_result_contains_solution_script(self) -> None:
        """InnerLoopResult.best_solution is a SolutionScript instance."""
        result = _make_inner_loop_result()
        assert isinstance(result.best_solution, SolutionScript)

    def test_ensemble_attempt_contains_solution_script(self) -> None:
        """EnsembleAttempt.solution is a SolutionScript instance."""
        attempt = _make_ensemble_attempt()
        assert isinstance(attempt.solution, SolutionScript)

    def test_inner_loop_result_nested_field_access(self) -> None:
        """InnerLoopResult allows deep field access through nested models."""
        result = _make_inner_loop_result()
        # Access through best_solution
        assert isinstance(result.best_solution.content, str)
        assert result.best_solution.phase == SolutionPhase.REFINED
        # Access through attempts
        assert isinstance(result.attempts[0].plan, str)
        assert isinstance(result.attempts[0].code_block, str)

    def test_ensemble_attempt_nested_field_access(self) -> None:
        """EnsembleAttempt allows deep field access through solution."""
        attempt = _make_ensemble_attempt()
        assert isinstance(attempt.solution.content, str)
        assert attempt.solution.phase == SolutionPhase.ENSEMBLE


# ===========================================================================
# Property-based Tests: RefinementAttempt with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestRefinementAttemptPropertyBased:
    """Property-based tests for RefinementAttempt using Hypothesis."""

    @given(
        plan=st.text(min_size=0, max_size=200),
        score=st.one_of(
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        code_block=st.text(min_size=0, max_size=200),
        was_improvement=st.booleans(),
    )
    @settings(max_examples=50)
    def test_any_valid_inputs_produce_valid_attempt(
        self,
        plan: str,
        score: float | None,
        code_block: str,
        was_improvement: bool,
    ) -> None:
        """Property: any combination of valid typed inputs creates a valid attempt."""
        attempt = RefinementAttempt(
            plan=plan,
            score=score,
            code_block=code_block,
            was_improvement=was_improvement,
        )
        assert attempt.plan == plan
        assert attempt.score == score
        assert attempt.code_block == code_block
        assert attempt.was_improvement == was_improvement

    @given(
        plan=st.text(min_size=0, max_size=100),
        score=st.one_of(
            st.none(),
            st.floats(
                min_value=-1e4,
                max_value=1e4,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        code_block=st.text(min_size=0, max_size=100),
        was_improvement=st.booleans(),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_all_fields(
        self,
        plan: str,
        score: float | None,
        code_block: str,
        was_improvement: bool,
    ) -> None:
        """Property: JSON round-trip preserves all RefinementAttempt fields."""
        original = RefinementAttempt(
            plan=plan,
            score=score,
            code_block=code_block,
            was_improvement=was_improvement,
        )
        restored = RefinementAttempt.model_validate_json(original.model_dump_json())
        assert restored == original

    @given(
        was_improvement=st.booleans(),
    )
    @settings(max_examples=20)
    def test_was_improvement_is_always_bool(self, was_improvement: bool) -> None:
        """Property: was_improvement is always stored as a bool."""
        attempt = _make_refinement_attempt(was_improvement=was_improvement)
        assert isinstance(attempt.was_improvement, bool)
        assert attempt.was_improvement is was_improvement


# ===========================================================================
# Property-based Tests: EnsembleAttempt with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestEnsembleAttemptPropertyBased:
    """Property-based tests for EnsembleAttempt using Hypothesis."""

    @given(
        plan=st.text(min_size=0, max_size=200),
        score=st.one_of(
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        content=st.text(min_size=1, max_size=200),
        phase=st.sampled_from(list(SolutionPhase)),
    )
    @settings(max_examples=50)
    def test_any_valid_inputs_produce_valid_attempt(
        self,
        plan: str,
        score: float | None,
        content: str,
        phase: SolutionPhase,
    ) -> None:
        """Property: any combination of valid typed inputs creates a valid attempt."""
        sol = SolutionScript(content=content, phase=phase)
        attempt = EnsembleAttempt(
            plan=plan,
            score=score,
            solution=sol,
        )
        assert attempt.plan == plan
        assert attempt.score == score
        assert attempt.solution.content == content
        assert attempt.solution.phase == phase

    @given(
        plan=st.text(min_size=0, max_size=100),
        score=st.one_of(
            st.none(),
            st.floats(
                min_value=-1e4,
                max_value=1e4,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_plan_and_score(
        self,
        plan: str,
        score: float | None,
    ) -> None:
        """Property: JSON round-trip preserves plan and score fields."""
        original = EnsembleAttempt(
            plan=plan,
            score=score,
            solution=_make_solution_script(phase=SolutionPhase.ENSEMBLE),
        )
        restored = EnsembleAttempt.model_validate_json(original.model_dump_json())
        assert restored.plan == original.plan
        assert restored.score == original.score


# ===========================================================================
# Property-based Tests: InnerLoopResult with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestInnerLoopResultPropertyBased:
    """Property-based tests for InnerLoopResult using Hypothesis."""

    @given(
        best_score=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        improved=st.booleans(),
        num_attempts=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_any_valid_score_and_improved_flag_accepted(
        self,
        best_score: float,
        improved: bool,
        num_attempts: int,
    ) -> None:
        """Property: any valid score with any improved flag creates a valid result."""
        attempts = [
            _make_refinement_attempt(
                plan=f"Plan {i}",
                score=best_score - 0.01 * i,
                was_improvement=i == 0,
            )
            for i in range(num_attempts)
        ]
        result = InnerLoopResult(
            best_solution=_make_solution_script(),
            best_score=best_score,
            attempts=attempts,
            improved=improved,
        )
        assert result.best_score == best_score
        assert result.improved is improved
        assert len(result.attempts) == num_attempts

    @given(
        best_score=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        improved=st.booleans(),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_score_and_improved(
        self,
        best_score: float,
        improved: bool,
    ) -> None:
        """Property: JSON round-trip preserves best_score and improved."""
        original = _make_inner_loop_result(
            best_score=best_score,
            improved=improved,
        )
        restored = InnerLoopResult.model_validate_json(original.model_dump_json())
        assert restored.best_score == original.best_score
        assert restored.improved == original.improved

    @given(
        num_attempts=st.integers(min_value=0, max_value=8),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_attempt_count(
        self,
        num_attempts: int,
    ) -> None:
        """Property: JSON round-trip preserves the number of attempts."""
        attempts = [_make_refinement_attempt() for _ in range(num_attempts)]
        original = _make_inner_loop_result(attempts=attempts)
        restored = InnerLoopResult.model_validate_json(original.model_dump_json())
        assert len(restored.attempts) == num_attempts

    @given(
        improved=st.booleans(),
    )
    @settings(max_examples=20)
    def test_improved_is_always_bool(self, improved: bool) -> None:
        """Property: improved is always stored as a bool."""
        result = _make_inner_loop_result(improved=improved)
        assert isinstance(result.improved, bool)
        assert result.improved is improved

    @given(
        scores=st.lists(
            st.one_of(
                st.none(),
                st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            min_size=0,
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_attempts_with_none_scores_round_trip(
        self,
        scores: list[float | None],
    ) -> None:
        """Property: attempts with None scores survive JSON round-trip."""
        attempts = [
            _make_refinement_attempt(score=s, was_improvement=s is not None)
            for s in scores
        ]
        original = _make_inner_loop_result(attempts=attempts)
        restored = InnerLoopResult.model_validate_json(original.model_dump_json())
        for i, s in enumerate(scores):
            assert restored.attempts[i].score == s
