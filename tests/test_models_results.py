"""Tests for MLE-STAR evaluation and phase result data models (Task 06).

Validates EvaluationResult, Phase1Result, Phase2Result, Phase3Result, and
FinalResult Pydantic models defined in ``src/mle_star/models.py``.  These
tests are written TDD-first -- the implementation does not yet exist.  They
serve as the executable specification for REQ-DM-021 through REQ-DM-025.

Refs:
    SRS 01a (Data Models Results), IMPLEMENTATION_PLAN.md Task 06.
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings, strategies as st
from mle_star.models import (
    CodeBlock,
    CodeBlockCategory,
    DataModality,
    EvaluationResult,
    FinalResult,
    MetricDirection,
    Phase1Result,
    Phase2Result,
    Phase3Result,
    PipelineConfig,
    RetrievedModel,
    SolutionPhase,
    SolutionScript,
    TaskDescription,
    TaskType,
)
from pydantic import ValidationError
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


def _make_retrieved_model(**overrides: Any) -> RetrievedModel:
    """Build a valid RetrievedModel with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed RetrievedModel instance.
    """
    defaults: dict[str, Any] = {
        "model_name": "RandomForestClassifier",
        "example_code": "from sklearn.ensemble import RandomForestClassifier\n",
    }
    defaults.update(overrides)
    return RetrievedModel(**defaults)


def _make_code_block(**overrides: Any) -> CodeBlock:
    """Build a valid CodeBlock with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed CodeBlock instance.
    """
    defaults: dict[str, Any] = {
        "content": "model.fit(X_train, y_train)",
        "category": CodeBlockCategory.TRAINING,
        "outer_step": 1,
    }
    defaults.update(overrides)
    return CodeBlock(**defaults)


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


def _make_phase1_result(**overrides: Any) -> Phase1Result:
    """Build a valid Phase1Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase1Result instance.
    """
    defaults: dict[str, Any] = {
        "retrieved_models": [_make_retrieved_model()],
        "candidate_solutions": [_make_solution_script()],
        "candidate_scores": [0.85],
        "initial_solution": _make_solution_script(phase=SolutionPhase.INIT),
        "initial_score": 0.85,
    }
    defaults.update(overrides)
    return Phase1Result(**defaults)


def _make_phase2_result(**overrides: Any) -> Phase2Result:
    """Build a valid Phase2Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase2Result instance.
    """
    defaults: dict[str, Any] = {
        "ablation_summaries": ["Removed feature X, score dropped by 0.02."],
        "refined_blocks": [_make_code_block()],
        "best_solution": _make_solution_script(phase=SolutionPhase.REFINED),
        "best_score": 0.90,
        "step_history": [{"step": 1, "score": 0.88, "action": "refine_training"}],
    }
    defaults.update(overrides)
    return Phase2Result(**defaults)


def _make_phase3_result(**overrides: Any) -> Phase3Result:
    """Build a valid Phase3Result with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed Phase3Result instance.
    """
    defaults: dict[str, Any] = {
        "input_solutions": [
            _make_solution_script(phase=SolutionPhase.REFINED, score=0.90),
        ],
        "ensemble_plans": ["Weighted average of top-2 solutions."],
        "ensemble_scores": [0.92],
        "best_ensemble": _make_solution_script(
            phase=SolutionPhase.ENSEMBLE, score=0.92
        ),
        "best_ensemble_score": 0.92,
    }
    defaults.update(overrides)
    return Phase3Result(**defaults)


def _make_final_result(**overrides: Any) -> FinalResult:
    """Build a valid FinalResult with sensible defaults.

    Args:
        **overrides: Field values to override.

    Returns:
        A fully constructed FinalResult instance.
    """
    defaults: dict[str, Any] = {
        "task": _make_task_description(),
        "config": PipelineConfig(),
        "phase1": _make_phase1_result(),
        "phase2_results": [_make_phase2_result()],
        "phase3": _make_phase3_result(),
        "final_solution": _make_solution_script(phase=SolutionPhase.FINAL),
        "submission_path": "./final/submission.csv",
        "total_duration_seconds": 3600.0,
    }
    defaults.update(overrides)
    return FinalResult(**defaults)


# ===========================================================================
# REQ-DM-021: EvaluationResult
# ===========================================================================


@pytest.mark.unit
class TestEvaluationResultConstruction:
    """EvaluationResult has correct required and optional fields (REQ-DM-021)."""

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all fields succeeds and stores correct values."""
        result = _make_evaluation_result()
        assert result.score == 0.85
        assert result.stdout == "Training complete."
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.duration_seconds == 42.5
        assert result.is_error is False
        assert result.error_traceback is None

    def test_score_accepts_none(self) -> None:
        """Score field accepts None (parsing failure case)."""
        result = _make_evaluation_result(score=None)
        assert result.score is None

    def test_score_accepts_float(self) -> None:
        """Score field accepts a float value."""
        result = _make_evaluation_result(score=0.99)
        assert result.score == 0.99

    def test_score_accepts_negative_float(self) -> None:
        """Score field accepts negative floats (some metrics are negative)."""
        result = _make_evaluation_result(score=-1.234)
        assert result.score == -1.234

    def test_score_accepts_zero(self) -> None:
        """Score field accepts zero."""
        result = _make_evaluation_result(score=0.0)
        assert result.score == 0.0

    def test_stdout_is_required_string(self) -> None:
        """Stdout field holds a string value."""
        result = _make_evaluation_result(stdout="output line 1\noutput line 2")
        assert isinstance(result.stdout, str)
        assert "output line 1" in result.stdout

    def test_stderr_is_required_string(self) -> None:
        """Stderr field holds a string value."""
        result = _make_evaluation_result(stderr="WARNING: deprecation")
        assert isinstance(result.stderr, str)
        assert result.stderr == "WARNING: deprecation"

    def test_exit_code_is_required_int(self) -> None:
        """exit_code field holds an integer value."""
        result = _make_evaluation_result(exit_code=1)
        assert isinstance(result.exit_code, int)
        assert result.exit_code == 1

    def test_exit_code_accepts_nonzero(self) -> None:
        """exit_code accepts non-zero values (error exit codes)."""
        result = _make_evaluation_result(exit_code=137)
        assert result.exit_code == 137

    def test_duration_seconds_is_required_float(self) -> None:
        """duration_seconds field holds a float value."""
        result = _make_evaluation_result(duration_seconds=120.75)
        assert isinstance(result.duration_seconds, float)
        assert result.duration_seconds == 120.75

    def test_is_error_is_required_bool(self) -> None:
        """is_error field holds a boolean value."""
        result = _make_evaluation_result(is_error=True)
        assert isinstance(result.is_error, bool)
        assert result.is_error is True

    def test_error_traceback_defaults_to_none(self) -> None:
        """error_traceback defaults to None when not provided."""
        result = _make_evaluation_result()
        assert result.error_traceback is None

    def test_error_traceback_accepts_string(self) -> None:
        """error_traceback accepts a string value."""
        tb = "Traceback (most recent call last):\n  File 'x.py', line 1\nError"
        result = _make_evaluation_result(error_traceback=tb)
        assert result.error_traceback == tb

    def test_error_result_with_traceback(self) -> None:
        """An error result can carry score=None, is_error=True, and a traceback."""
        result = _make_evaluation_result(
            score=None,
            exit_code=1,
            is_error=True,
            error_traceback="ImportError: no module named 'xgboost'",
        )
        assert result.score is None
        assert result.is_error is True
        assert result.exit_code == 1
        assert result.error_traceback is not None
        assert "xgboost" in result.error_traceback


@pytest.mark.unit
class TestEvaluationResultRequiredFields:
    """EvaluationResult raises ValidationError when required fields are missing."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "stdout",
            "stderr",
            "exit_code",
            "duration_seconds",
            "is_error",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "score": 0.85,
            "stdout": "ok",
            "stderr": "",
            "exit_code": 0,
            "duration_seconds": 1.0,
            "is_error": False,
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            EvaluationResult(**all_fields)

    def test_score_can_be_omitted(self) -> None:
        """Omitting score should still create a valid result (defaults to None).

        Score is typed as ``float | None`` and should default to None
        when not provided, since parsing can fail.
        """
        result = EvaluationResult(
            stdout="ok",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            is_error=False,
        )
        assert result.score is None


@pytest.mark.unit
class TestEvaluationResultFrozen:
    """EvaluationResult is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_score(self) -> None:
        """Assignment to score raises an error."""
        result = _make_evaluation_result()
        with pytest.raises(ValidationError):
            result.score = 0.99  # type: ignore[misc]

    def test_cannot_mutate_stdout(self) -> None:
        """Assignment to stdout raises an error."""
        result = _make_evaluation_result()
        with pytest.raises(ValidationError):
            result.stdout = "changed"  # type: ignore[misc]

    def test_cannot_mutate_exit_code(self) -> None:
        """Assignment to exit_code raises an error."""
        result = _make_evaluation_result()
        with pytest.raises(ValidationError):
            result.exit_code = 99  # type: ignore[misc]

    def test_cannot_mutate_is_error(self) -> None:
        """Assignment to is_error raises an error."""
        result = _make_evaluation_result()
        with pytest.raises(ValidationError):
            result.is_error = True  # type: ignore[misc]

    def test_cannot_mutate_duration_seconds(self) -> None:
        """Assignment to duration_seconds raises an error."""
        result = _make_evaluation_result()
        with pytest.raises(ValidationError):
            result.duration_seconds = 999.0  # type: ignore[misc]

    def test_cannot_mutate_error_traceback(self) -> None:
        """Assignment to error_traceback raises an error."""
        result = _make_evaluation_result()
        with pytest.raises(ValidationError):
            result.error_traceback = "new traceback"  # type: ignore[misc]


@pytest.mark.unit
class TestEvaluationResultSerialization:
    """EvaluationResult supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_evaluation_result(
            score=0.75,
            stdout="output",
            stderr="warn",
            exit_code=0,
            duration_seconds=10.5,
            is_error=False,
            error_traceback=None,
        )
        json_str = original.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)

        assert restored.score == original.score
        assert restored.stdout == original.stdout
        assert restored.stderr == original.stderr
        assert restored.exit_code == original.exit_code
        assert restored.duration_seconds == original.duration_seconds
        assert restored.is_error == original.is_error
        assert restored.error_traceback == original.error_traceback

    def test_round_trip_with_none_score(self) -> None:
        """Round-trip preserves None score."""
        original = _make_evaluation_result(score=None)
        json_str = original.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)
        assert restored.score is None

    def test_round_trip_with_error_traceback(self) -> None:
        """Round-trip preserves error_traceback string."""
        tb = "Traceback:\n  File 'x.py'\nValueError: bad"
        original = _make_evaluation_result(
            score=None, is_error=True, error_traceback=tb
        )
        json_str = original.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)
        assert restored.error_traceback == tb

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_evaluation_result()
        restored = EvaluationResult.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        result = _make_evaluation_result()
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed, dict)
        assert "score" in parsed
        assert "stdout" in parsed
        assert "stderr" in parsed
        assert "exit_code" in parsed
        assert "duration_seconds" in parsed
        assert "is_error" in parsed


# ===========================================================================
# REQ-DM-022: Phase1Result
# ===========================================================================


@pytest.mark.unit
class TestPhase1ResultConstruction:
    """Phase1Result has correct required fields (REQ-DM-022)."""

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all required fields succeeds."""
        result = _make_phase1_result()
        assert len(result.retrieved_models) == 1
        assert len(result.candidate_solutions) == 1
        assert len(result.candidate_scores) == 1
        assert result.candidate_scores[0] == 0.85
        assert isinstance(result.initial_solution, SolutionScript)
        assert result.initial_score == 0.85

    def test_retrieved_models_contains_retrieved_model_instances(self) -> None:
        """retrieved_models list holds RetrievedModel instances."""
        models = [
            _make_retrieved_model(model_name="RF"),
            _make_retrieved_model(model_name="XGB"),
        ]
        result = _make_phase1_result(retrieved_models=models)
        assert len(result.retrieved_models) == 2
        assert result.retrieved_models[0].model_name == "RF"
        assert result.retrieved_models[1].model_name == "XGB"

    def test_candidate_solutions_contains_solution_script_instances(self) -> None:
        """candidate_solutions list holds SolutionScript instances."""
        solutions = [
            _make_solution_script(content="s1"),
            _make_solution_script(content="s2"),
        ]
        result = _make_phase1_result(candidate_solutions=solutions)
        assert len(result.candidate_solutions) == 2
        assert result.candidate_solutions[0].content == "s1"
        assert result.candidate_solutions[1].content == "s2"

    def test_candidate_scores_accepts_none_values(self) -> None:
        """candidate_scores list accepts None values (failed evaluation)."""
        result = _make_phase1_result(candidate_scores=[0.85, None, 0.72])
        assert result.candidate_scores[0] == 0.85
        assert result.candidate_scores[1] is None
        assert result.candidate_scores[2] == 0.72

    def test_candidate_scores_all_none(self) -> None:
        """candidate_scores list can contain all None values."""
        result = _make_phase1_result(candidate_scores=[None, None])
        assert all(s is None for s in result.candidate_scores)

    def test_initial_solution_is_solution_script(self) -> None:
        """initial_solution holds a SolutionScript instance."""
        sol = _make_solution_script(content="initial", phase=SolutionPhase.INIT)
        result = _make_phase1_result(initial_solution=sol)
        assert isinstance(result.initial_solution, SolutionScript)
        assert result.initial_solution.content == "initial"

    def test_initial_score_is_float(self) -> None:
        """initial_score holds a float value."""
        result = _make_phase1_result(initial_score=0.91)
        assert isinstance(result.initial_score, float)
        assert result.initial_score == 0.91

    def test_empty_lists_accepted(self) -> None:
        """Empty lists for retrieved_models, candidate_solutions, scores are valid."""
        result = _make_phase1_result(
            retrieved_models=[],
            candidate_solutions=[],
            candidate_scores=[],
        )
        assert len(result.retrieved_models) == 0
        assert len(result.candidate_solutions) == 0
        assert len(result.candidate_scores) == 0


@pytest.mark.unit
class TestPhase1ResultRequiredFields:
    """Phase1Result raises ValidationError when required fields are missing."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "retrieved_models",
            "candidate_solutions",
            "candidate_scores",
            "initial_solution",
            "initial_score",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "retrieved_models": [_make_retrieved_model()],
            "candidate_solutions": [_make_solution_script()],
            "candidate_scores": [0.85],
            "initial_solution": _make_solution_script(),
            "initial_score": 0.85,
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            Phase1Result(**all_fields)


@pytest.mark.unit
class TestPhase1ResultFrozen:
    """Phase1Result is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_retrieved_models(self) -> None:
        """Assignment to retrieved_models raises an error."""
        result = _make_phase1_result()
        with pytest.raises(ValidationError):
            result.retrieved_models = []  # type: ignore[misc]

    def test_cannot_mutate_initial_score(self) -> None:
        """Assignment to initial_score raises an error."""
        result = _make_phase1_result()
        with pytest.raises(ValidationError):
            result.initial_score = 0.99  # type: ignore[misc]

    def test_cannot_mutate_initial_solution(self) -> None:
        """Assignment to initial_solution raises an error."""
        result = _make_phase1_result()
        with pytest.raises(ValidationError):
            result.initial_solution = _make_solution_script()  # type: ignore[misc]

    def test_cannot_mutate_candidate_scores(self) -> None:
        """Assignment to candidate_scores raises an error."""
        result = _make_phase1_result()
        with pytest.raises(ValidationError):
            result.candidate_scores = [0.5]  # type: ignore[misc]


@pytest.mark.unit
class TestPhase1ResultSerialization:
    """Phase1Result supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_phase1_result()
        json_str = original.model_dump_json()
        restored = Phase1Result.model_validate_json(json_str)

        assert len(restored.retrieved_models) == len(original.retrieved_models)
        assert (
            restored.retrieved_models[0].model_name
            == original.retrieved_models[0].model_name
        )
        assert len(restored.candidate_solutions) == len(original.candidate_solutions)
        assert len(restored.candidate_scores) == len(original.candidate_scores)
        assert restored.candidate_scores[0] == original.candidate_scores[0]
        assert restored.initial_solution.content == original.initial_solution.content
        assert restored.initial_score == original.initial_score

    def test_round_trip_with_none_scores(self) -> None:
        """Round-trip preserves None values in candidate_scores."""
        original = _make_phase1_result(candidate_scores=[0.85, None, 0.70])
        json_str = original.model_dump_json()
        restored = Phase1Result.model_validate_json(json_str)
        assert restored.candidate_scores == [0.85, None, 0.70]

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_phase1_result()
        restored = Phase1Result.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        result = _make_phase1_result()
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed, dict)
        assert "retrieved_models" in parsed
        assert "candidate_solutions" in parsed
        assert "candidate_scores" in parsed
        assert "initial_solution" in parsed
        assert "initial_score" in parsed

    def test_round_trip_multiple_models_and_solutions(self) -> None:
        """Round-trip with multiple retrieved models and candidate solutions."""
        original = _make_phase1_result(
            retrieved_models=[
                _make_retrieved_model(model_name="RF"),
                _make_retrieved_model(model_name="XGB"),
            ],
            candidate_solutions=[
                _make_solution_script(content="s1"),
                _make_solution_script(content="s2"),
            ],
            candidate_scores=[0.85, 0.90],
        )
        json_str = original.model_dump_json()
        restored = Phase1Result.model_validate_json(json_str)
        assert len(restored.retrieved_models) == 2
        assert len(restored.candidate_solutions) == 2
        assert restored.candidate_scores == [0.85, 0.90]


# ===========================================================================
# REQ-DM-023: Phase2Result
# ===========================================================================


@pytest.mark.unit
class TestPhase2ResultConstruction:
    """Phase2Result has correct required fields (REQ-DM-023)."""

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all required fields succeeds."""
        result = _make_phase2_result()
        assert len(result.ablation_summaries) == 1
        assert len(result.refined_blocks) == 1
        assert isinstance(result.best_solution, SolutionScript)
        assert result.best_score == 0.90
        assert len(result.step_history) == 1

    def test_ablation_summaries_is_list_of_strings(self) -> None:
        """ablation_summaries holds a list of strings."""
        summaries = ["Summary A", "Summary B", "Summary C"]
        result = _make_phase2_result(ablation_summaries=summaries)
        assert result.ablation_summaries == summaries
        assert all(isinstance(s, str) for s in result.ablation_summaries)

    def test_refined_blocks_contains_code_block_instances(self) -> None:
        """refined_blocks list holds CodeBlock instances."""
        blocks = [
            _make_code_block(content="block1"),
            _make_code_block(content="block2"),
        ]
        result = _make_phase2_result(refined_blocks=blocks)
        assert len(result.refined_blocks) == 2
        assert result.refined_blocks[0].content == "block1"
        assert result.refined_blocks[1].content == "block2"

    def test_best_solution_is_solution_script(self) -> None:
        """best_solution holds a SolutionScript instance."""
        sol = _make_solution_script(content="best", phase=SolutionPhase.REFINED)
        result = _make_phase2_result(best_solution=sol)
        assert isinstance(result.best_solution, SolutionScript)
        assert result.best_solution.content == "best"

    def test_best_score_is_float(self) -> None:
        """best_score holds a float value."""
        result = _make_phase2_result(best_score=0.95)
        assert isinstance(result.best_score, float)
        assert result.best_score == 0.95

    def test_step_history_is_list_of_dicts(self) -> None:
        """step_history holds a list of dicts with arbitrary keys."""
        history = [
            {"step": 1, "score": 0.88, "action": "refine"},
            {"step": 2, "score": 0.90, "action": "tune"},
        ]
        result = _make_phase2_result(step_history=history)
        assert len(result.step_history) == 2
        assert result.step_history[0]["step"] == 1
        assert result.step_history[1]["score"] == 0.90

    def test_step_history_dict_with_varied_types(self) -> None:
        """step_history dicts can hold mixed-type values."""
        history = [
            {"step": 1, "improved": True, "delta": 0.02, "notes": None},
        ]
        result = _make_phase2_result(step_history=history)
        assert result.step_history[0]["improved"] is True
        assert result.step_history[0]["delta"] == 0.02
        assert result.step_history[0]["notes"] is None

    def test_empty_lists_accepted(self) -> None:
        """Empty lists for ablation_summaries, refined_blocks, step_history are valid."""
        result = _make_phase2_result(
            ablation_summaries=[],
            refined_blocks=[],
            step_history=[],
        )
        assert len(result.ablation_summaries) == 0
        assert len(result.refined_blocks) == 0
        assert len(result.step_history) == 0


@pytest.mark.unit
class TestPhase2ResultRequiredFields:
    """Phase2Result raises ValidationError when required fields are missing."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "ablation_summaries",
            "refined_blocks",
            "best_solution",
            "best_score",
            "step_history",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "ablation_summaries": ["summary"],
            "refined_blocks": [_make_code_block()],
            "best_solution": _make_solution_script(phase=SolutionPhase.REFINED),
            "best_score": 0.90,
            "step_history": [{"step": 1}],
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            Phase2Result(**all_fields)


@pytest.mark.unit
class TestPhase2ResultFrozen:
    """Phase2Result is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_ablation_summaries(self) -> None:
        """Assignment to ablation_summaries raises an error."""
        result = _make_phase2_result()
        with pytest.raises(ValidationError):
            result.ablation_summaries = []  # type: ignore[misc]

    def test_cannot_mutate_best_score(self) -> None:
        """Assignment to best_score raises an error."""
        result = _make_phase2_result()
        with pytest.raises(ValidationError):
            result.best_score = 0.99  # type: ignore[misc]

    def test_cannot_mutate_best_solution(self) -> None:
        """Assignment to best_solution raises an error."""
        result = _make_phase2_result()
        with pytest.raises(ValidationError):
            result.best_solution = _make_solution_script()  # type: ignore[misc]

    def test_cannot_mutate_step_history(self) -> None:
        """Assignment to step_history raises an error."""
        result = _make_phase2_result()
        with pytest.raises(ValidationError):
            result.step_history = []  # type: ignore[misc]

    def test_cannot_mutate_refined_blocks(self) -> None:
        """Assignment to refined_blocks raises an error."""
        result = _make_phase2_result()
        with pytest.raises(ValidationError):
            result.refined_blocks = []  # type: ignore[misc]


@pytest.mark.unit
class TestPhase2ResultSerialization:
    """Phase2Result supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_phase2_result()
        json_str = original.model_dump_json()
        restored = Phase2Result.model_validate_json(json_str)

        assert restored.ablation_summaries == original.ablation_summaries
        assert len(restored.refined_blocks) == len(original.refined_blocks)
        assert restored.refined_blocks[0].content == original.refined_blocks[0].content
        assert restored.best_solution.content == original.best_solution.content
        assert restored.best_score == original.best_score
        assert len(restored.step_history) == len(original.step_history)

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_phase2_result()
        restored = Phase2Result.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        result = _make_phase2_result()
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed, dict)
        assert "ablation_summaries" in parsed
        assert "refined_blocks" in parsed
        assert "best_solution" in parsed
        assert "best_score" in parsed
        assert "step_history" in parsed

    def test_round_trip_complex_step_history(self) -> None:
        """Round-trip preserves complex step_history with nested data."""
        history = [
            {"step": 1, "score": 0.88, "action": "refine", "improved": True},
            {"step": 2, "score": 0.90, "action": "tune", "improved": True},
            {"step": 3, "score": 0.90, "action": "tune", "improved": False},
        ]
        original = _make_phase2_result(step_history=history)
        json_str = original.model_dump_json()
        restored = Phase2Result.model_validate_json(json_str)
        assert restored.step_history == history


# ===========================================================================
# REQ-DM-024: Phase3Result
# ===========================================================================


@pytest.mark.unit
class TestPhase3ResultConstruction:
    """Phase3Result has correct required fields (REQ-DM-024)."""

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all required fields succeeds."""
        result = _make_phase3_result()
        assert len(result.input_solutions) == 1
        assert len(result.ensemble_plans) == 1
        assert len(result.ensemble_scores) == 1
        assert isinstance(result.best_ensemble, SolutionScript)
        assert result.best_ensemble_score == 0.92

    def test_input_solutions_contains_solution_scripts(self) -> None:
        """input_solutions list holds SolutionScript instances."""
        solutions = [
            _make_solution_script(content="s1", phase=SolutionPhase.REFINED),
            _make_solution_script(content="s2", phase=SolutionPhase.REFINED),
        ]
        result = _make_phase3_result(input_solutions=solutions)
        assert len(result.input_solutions) == 2
        assert result.input_solutions[0].content == "s1"
        assert result.input_solutions[1].content == "s2"

    def test_ensemble_plans_is_list_of_strings(self) -> None:
        """ensemble_plans holds a list of strings."""
        plans = ["Plan A: weighted average", "Plan B: stacking", "Plan C: voting"]
        result = _make_phase3_result(ensemble_plans=plans)
        assert result.ensemble_plans == plans
        assert all(isinstance(p, str) for p in result.ensemble_plans)

    def test_ensemble_scores_accepts_none_values(self) -> None:
        """ensemble_scores list accepts None values (failed ensemble)."""
        result = _make_phase3_result(ensemble_scores=[0.92, None, 0.88])
        assert result.ensemble_scores[0] == 0.92
        assert result.ensemble_scores[1] is None
        assert result.ensemble_scores[2] == 0.88

    def test_ensemble_scores_all_none(self) -> None:
        """ensemble_scores list can contain all None values."""
        result = _make_phase3_result(ensemble_scores=[None, None])
        assert all(s is None for s in result.ensemble_scores)

    def test_best_ensemble_is_solution_script(self) -> None:
        """best_ensemble holds a SolutionScript instance."""
        sol = _make_solution_script(content="ensemble", phase=SolutionPhase.ENSEMBLE)
        result = _make_phase3_result(best_ensemble=sol)
        assert isinstance(result.best_ensemble, SolutionScript)
        assert result.best_ensemble.content == "ensemble"

    def test_best_ensemble_score_is_float(self) -> None:
        """best_ensemble_score holds a float value."""
        result = _make_phase3_result(best_ensemble_score=0.95)
        assert isinstance(result.best_ensemble_score, float)
        assert result.best_ensemble_score == 0.95

    def test_empty_lists_accepted(self) -> None:
        """Empty lists for input_solutions, ensemble_plans, ensemble_scores are valid."""
        result = _make_phase3_result(
            input_solutions=[],
            ensemble_plans=[],
            ensemble_scores=[],
        )
        assert len(result.input_solutions) == 0
        assert len(result.ensemble_plans) == 0
        assert len(result.ensemble_scores) == 0


@pytest.mark.unit
class TestPhase3ResultRequiredFields:
    """Phase3Result raises ValidationError when required fields are missing."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "input_solutions",
            "ensemble_plans",
            "ensemble_scores",
            "best_ensemble",
            "best_ensemble_score",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "input_solutions": [_make_solution_script(phase=SolutionPhase.REFINED)],
            "ensemble_plans": ["Weighted average"],
            "ensemble_scores": [0.92],
            "best_ensemble": _make_solution_script(phase=SolutionPhase.ENSEMBLE),
            "best_ensemble_score": 0.92,
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            Phase3Result(**all_fields)


@pytest.mark.unit
class TestPhase3ResultFrozen:
    """Phase3Result is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_input_solutions(self) -> None:
        """Assignment to input_solutions raises an error."""
        result = _make_phase3_result()
        with pytest.raises(ValidationError):
            result.input_solutions = []  # type: ignore[misc]

    def test_cannot_mutate_best_ensemble_score(self) -> None:
        """Assignment to best_ensemble_score raises an error."""
        result = _make_phase3_result()
        with pytest.raises(ValidationError):
            result.best_ensemble_score = 0.99  # type: ignore[misc]

    def test_cannot_mutate_best_ensemble(self) -> None:
        """Assignment to best_ensemble raises an error."""
        result = _make_phase3_result()
        with pytest.raises(ValidationError):
            result.best_ensemble = _make_solution_script()  # type: ignore[misc]

    def test_cannot_mutate_ensemble_plans(self) -> None:
        """Assignment to ensemble_plans raises an error."""
        result = _make_phase3_result()
        with pytest.raises(ValidationError):
            result.ensemble_plans = []  # type: ignore[misc]

    def test_cannot_mutate_ensemble_scores(self) -> None:
        """Assignment to ensemble_scores raises an error."""
        result = _make_phase3_result()
        with pytest.raises(ValidationError):
            result.ensemble_scores = []  # type: ignore[misc]


@pytest.mark.unit
class TestPhase3ResultSerialization:
    """Phase3Result supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_phase3_result()
        json_str = original.model_dump_json()
        restored = Phase3Result.model_validate_json(json_str)

        assert len(restored.input_solutions) == len(original.input_solutions)
        assert restored.ensemble_plans == original.ensemble_plans
        assert restored.ensemble_scores == original.ensemble_scores
        assert restored.best_ensemble.content == original.best_ensemble.content
        assert restored.best_ensemble_score == original.best_ensemble_score

    def test_round_trip_with_none_scores(self) -> None:
        """Round-trip preserves None values in ensemble_scores."""
        original = _make_phase3_result(ensemble_scores=[0.92, None, 0.88])
        json_str = original.model_dump_json()
        restored = Phase3Result.model_validate_json(json_str)
        assert restored.ensemble_scores == [0.92, None, 0.88]

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_phase3_result()
        restored = Phase3Result.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        result = _make_phase3_result()
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed, dict)
        assert "input_solutions" in parsed
        assert "ensemble_plans" in parsed
        assert "ensemble_scores" in parsed
        assert "best_ensemble" in parsed
        assert "best_ensemble_score" in parsed


# ===========================================================================
# REQ-DM-025: FinalResult
# ===========================================================================


@pytest.mark.unit
class TestFinalResultConstruction:
    """FinalResult has correct required and optional fields (REQ-DM-025)."""

    def test_valid_construction_with_all_fields(self) -> None:
        """Constructing with all fields succeeds."""
        result = _make_final_result()
        assert isinstance(result.task, TaskDescription)
        assert isinstance(result.config, PipelineConfig)
        assert isinstance(result.phase1, Phase1Result)
        assert len(result.phase2_results) == 1
        assert isinstance(result.phase2_results[0], Phase2Result)
        assert isinstance(result.phase3, Phase3Result)
        assert isinstance(result.final_solution, SolutionScript)
        assert result.submission_path == "./final/submission.csv"
        assert result.total_duration_seconds == 3600.0

    def test_task_holds_task_description(self) -> None:
        """The task field holds a TaskDescription instance."""
        td = _make_task_description(competition_id="my-comp")
        result = _make_final_result(task=td)
        assert result.task.competition_id == "my-comp"

    def test_config_holds_pipeline_config(self) -> None:
        """The config field holds a PipelineConfig instance."""
        cfg = PipelineConfig(num_retrieved_models=8)
        result = _make_final_result(config=cfg)
        assert result.config.num_retrieved_models == 8

    def test_phase1_holds_phase1_result(self) -> None:
        """phase1 field holds a Phase1Result instance."""
        p1 = _make_phase1_result(initial_score=0.77)
        result = _make_final_result(phase1=p1)
        assert result.phase1.initial_score == 0.77

    def test_phase2_results_is_list_of_phase2_result(self) -> None:
        """phase2_results holds a list of Phase2Result instances."""
        p2a = _make_phase2_result(best_score=0.88)
        p2b = _make_phase2_result(best_score=0.91)
        result = _make_final_result(phase2_results=[p2a, p2b])
        assert len(result.phase2_results) == 2
        assert result.phase2_results[0].best_score == 0.88
        assert result.phase2_results[1].best_score == 0.91

    def test_phase3_accepts_none(self) -> None:
        """phase3 field accepts None (ensemble phase may be skipped)."""
        result = _make_final_result(phase3=None)
        assert result.phase3 is None

    def test_phase3_accepts_phase3_result(self) -> None:
        """phase3 field accepts a Phase3Result instance."""
        p3 = _make_phase3_result(best_ensemble_score=0.95)
        result = _make_final_result(phase3=p3)
        assert result.phase3 is not None
        assert result.phase3.best_ensemble_score == 0.95

    def test_final_solution_is_solution_script(self) -> None:
        """final_solution holds a SolutionScript instance."""
        sol = _make_solution_script(content="final code", phase=SolutionPhase.FINAL)
        result = _make_final_result(final_solution=sol)
        assert result.final_solution.content == "final code"

    def test_submission_path_is_string(self) -> None:
        """submission_path holds a string value."""
        result = _make_final_result(submission_path="/output/submission.csv")
        assert isinstance(result.submission_path, str)
        assert result.submission_path == "/output/submission.csv"

    def test_total_duration_seconds_is_float(self) -> None:
        """total_duration_seconds holds a float value."""
        result = _make_final_result(total_duration_seconds=7200.5)
        assert isinstance(result.total_duration_seconds, float)
        assert result.total_duration_seconds == 7200.5

    def test_phase2_results_empty_list_accepted(self) -> None:
        """Empty list for phase2_results is valid."""
        result = _make_final_result(phase2_results=[])
        assert len(result.phase2_results) == 0


@pytest.mark.unit
class TestFinalResultRequiredFields:
    """FinalResult raises ValidationError when required fields are missing."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            "task",
            "config",
            "phase1",
            "phase2_results",
            "final_solution",
            "submission_path",
            "total_duration_seconds",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        """Omitting any required field raises ValidationError."""
        all_fields: dict[str, Any] = {
            "task": _make_task_description(),
            "config": PipelineConfig(),
            "phase1": _make_phase1_result(),
            "phase2_results": [_make_phase2_result()],
            "phase3": _make_phase3_result(),
            "final_solution": _make_solution_script(phase=SolutionPhase.FINAL),
            "submission_path": "./final/submission.csv",
            "total_duration_seconds": 3600.0,
        }
        del all_fields[missing_field]
        with pytest.raises(ValidationError):
            FinalResult(**all_fields)

    def test_phase3_can_be_omitted_defaults_to_none(self) -> None:
        """Omitting phase3 should default to None (optional field)."""
        result = FinalResult(
            task=_make_task_description(),
            config=PipelineConfig(),
            phase1=_make_phase1_result(),
            phase2_results=[_make_phase2_result()],
            final_solution=_make_solution_script(phase=SolutionPhase.FINAL),
            submission_path="./final/submission.csv",
            total_duration_seconds=3600.0,
        )
        assert result.phase3 is None


@pytest.mark.unit
class TestFinalResultFrozen:
    """FinalResult is frozen (immutable) per REQ-DM-039."""

    def test_cannot_mutate_task(self) -> None:
        """Assignment to task raises an error."""
        result = _make_final_result()
        with pytest.raises(ValidationError):
            result.task = _make_task_description()  # type: ignore[misc]

    def test_cannot_mutate_config(self) -> None:
        """Assignment to config raises an error."""
        result = _make_final_result()
        with pytest.raises(ValidationError):
            result.config = PipelineConfig()  # type: ignore[misc]

    def test_cannot_mutate_phase1(self) -> None:
        """Assignment to phase1 raises an error."""
        result = _make_final_result()
        with pytest.raises(ValidationError):
            result.phase1 = _make_phase1_result()  # type: ignore[misc]

    def test_cannot_mutate_submission_path(self) -> None:
        """Assignment to submission_path raises an error."""
        result = _make_final_result()
        with pytest.raises(ValidationError):
            result.submission_path = "/new/path.csv"  # type: ignore[misc]

    def test_cannot_mutate_total_duration_seconds(self) -> None:
        """Assignment to total_duration_seconds raises an error."""
        result = _make_final_result()
        with pytest.raises(ValidationError):
            result.total_duration_seconds = 9999.0  # type: ignore[misc]

    def test_cannot_mutate_final_solution(self) -> None:
        """Assignment to final_solution raises an error."""
        result = _make_final_result()
        with pytest.raises(ValidationError):
            result.final_solution = _make_solution_script()  # type: ignore[misc]

    def test_cannot_mutate_phase2_results(self) -> None:
        """Assignment to phase2_results raises an error."""
        result = _make_final_result()
        with pytest.raises(ValidationError):
            result.phase2_results = []  # type: ignore[misc]

    def test_cannot_mutate_phase3(self) -> None:
        """Assignment to phase3 raises an error."""
        result = _make_final_result()
        with pytest.raises(ValidationError):
            result.phase3 = None  # type: ignore[misc]


@pytest.mark.unit
class TestFinalResultSerialization:
    """FinalResult supports JSON round-trip serialization."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """Serialize and deserialize; all fields preserved."""
        original = _make_final_result()
        json_str = original.model_dump_json()
        restored = FinalResult.model_validate_json(json_str)

        assert restored.task.competition_id == original.task.competition_id
        assert (
            restored.config.num_retrieved_models == original.config.num_retrieved_models
        )
        assert restored.phase1.initial_score == original.phase1.initial_score
        assert len(restored.phase2_results) == len(original.phase2_results)
        assert (
            restored.phase2_results[0].best_score
            == original.phase2_results[0].best_score
        )
        assert restored.phase3 is not None
        assert original.phase3 is not None
        assert (
            restored.phase3.best_ensemble_score == original.phase3.best_ensemble_score
        )
        assert restored.final_solution.content == original.final_solution.content
        assert restored.submission_path == original.submission_path
        assert restored.total_duration_seconds == original.total_duration_seconds

    def test_round_trip_with_none_phase3(self) -> None:
        """Round-trip preserves None phase3."""
        original = _make_final_result(phase3=None)
        json_str = original.model_dump_json()
        restored = FinalResult.model_validate_json(json_str)
        assert restored.phase3 is None

    def test_round_trip_equality(self) -> None:
        """Serialized then deserialized object equals the original."""
        original = _make_final_result()
        restored = FinalResult.model_validate_json(original.model_dump_json())
        assert original == restored

    def test_model_dump_json_returns_valid_json(self) -> None:
        """model_dump_json() produces parseable JSON."""
        result = _make_final_result()
        parsed = json.loads(result.model_dump_json())
        assert isinstance(parsed, dict)
        assert "task" in parsed
        assert "config" in parsed
        assert "phase1" in parsed
        assert "phase2_results" in parsed
        assert "phase3" in parsed
        assert "final_solution" in parsed
        assert "submission_path" in parsed
        assert "total_duration_seconds" in parsed

    def test_round_trip_multiple_phase2_results(self) -> None:
        """Round-trip with multiple Phase2Result entries."""
        original = _make_final_result(
            phase2_results=[
                _make_phase2_result(best_score=0.88),
                _make_phase2_result(best_score=0.91),
                _make_phase2_result(best_score=0.93),
            ]
        )
        json_str = original.model_dump_json()
        restored = FinalResult.model_validate_json(json_str)
        assert len(restored.phase2_results) == 3
        assert restored.phase2_results[0].best_score == 0.88
        assert restored.phase2_results[1].best_score == 0.91
        assert restored.phase2_results[2].best_score == 0.93


# ===========================================================================
# Composition tests: models containing other models
# ===========================================================================


@pytest.mark.unit
class TestModelComposition:
    """Result models correctly compose nested Pydantic models."""

    def test_phase1_result_contains_retrieved_models(self) -> None:
        """Phase1Result.retrieved_models are RetrievedModel instances."""
        result = _make_phase1_result()
        for model in result.retrieved_models:
            assert isinstance(model, RetrievedModel)

    def test_phase1_result_contains_solution_scripts(self) -> None:
        """Phase1Result.candidate_solutions are SolutionScript instances."""
        result = _make_phase1_result()
        for sol in result.candidate_solutions:
            assert isinstance(sol, SolutionScript)

    def test_phase2_result_contains_code_blocks(self) -> None:
        """Phase2Result.refined_blocks are CodeBlock instances."""
        result = _make_phase2_result()
        for block in result.refined_blocks:
            assert isinstance(block, CodeBlock)

    def test_phase3_result_contains_solution_scripts(self) -> None:
        """Phase3Result.input_solutions are SolutionScript instances."""
        result = _make_phase3_result()
        for sol in result.input_solutions:
            assert isinstance(sol, SolutionScript)

    def test_final_result_composes_all_phases(self) -> None:
        """FinalResult correctly composes TaskDescription, PipelineConfig, and all phases."""
        result = _make_final_result()
        assert isinstance(result.task, TaskDescription)
        assert isinstance(result.config, PipelineConfig)
        assert isinstance(result.phase1, Phase1Result)
        assert all(isinstance(p2, Phase2Result) for p2 in result.phase2_results)
        assert isinstance(result.phase3, Phase3Result)
        assert isinstance(result.final_solution, SolutionScript)

    def test_final_result_nested_model_field_access(self) -> None:
        """FinalResult allows deep field access through nested models."""
        result = _make_final_result()
        # Access through task
        assert result.task.task_type == TaskType.CLASSIFICATION
        # Access through config
        assert result.config.num_retrieved_models == 4
        # Access through phase1
        assert len(result.phase1.retrieved_models) >= 0
        # Access through phase2_results
        assert result.phase2_results[0].best_score >= 0
        # Access through phase3
        assert result.phase3 is not None
        assert result.phase3.best_ensemble_score >= 0


# ===========================================================================
# Property-based tests: EvaluationResult with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestEvaluationResultPropertyBased:
    """Property-based tests for EvaluationResult using Hypothesis."""

    @given(
        score=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
        stdout=st.text(min_size=0, max_size=200),
        stderr=st.text(min_size=0, max_size=200),
        exit_code=st.integers(min_value=-128, max_value=255),
        duration_seconds=st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        is_error=st.booleans(),
        error_traceback=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
    )
    @settings(max_examples=50)
    def test_any_valid_inputs_produce_valid_result(
        self,
        score: float | None,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_seconds: float,
        is_error: bool,
        error_traceback: str | None,
    ) -> None:
        """Property: any combination of valid typed inputs creates a valid result."""
        result = EvaluationResult(
            score=score,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            is_error=is_error,
            error_traceback=error_traceback,
        )
        assert result.score == score
        assert result.stdout == stdout
        assert result.stderr == stderr
        assert result.exit_code == exit_code
        assert result.duration_seconds == duration_seconds
        assert result.is_error == is_error
        assert result.error_traceback == error_traceback

    @given(
        score=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
        exit_code=st.integers(min_value=0, max_value=255),
        duration=st.floats(
            min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_key_fields(
        self, score: float | None, exit_code: int, duration: float
    ) -> None:
        """Property: JSON round-trip preserves score, exit_code, and duration."""
        original = EvaluationResult(
            score=score,
            stdout="out",
            stderr="err",
            exit_code=exit_code,
            duration_seconds=duration,
            is_error=False,
        )
        restored = EvaluationResult.model_validate_json(original.model_dump_json())
        assert restored.score == original.score
        assert restored.exit_code == original.exit_code
        assert restored.duration_seconds == original.duration_seconds

    @given(
        score=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
        stdout=st.text(min_size=0, max_size=100),
        stderr=st.text(min_size=0, max_size=100),
        exit_code=st.integers(min_value=0, max_value=255),
        duration=st.floats(
            min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False
        ),
        is_error=st.booleans(),
    )
    @settings(max_examples=30)
    def test_round_trip_equality(
        self,
        score: float | None,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration: float,
        is_error: bool,
    ) -> None:
        """Property: JSON round-trip produces an equal object."""
        original = EvaluationResult(
            score=score,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration,
            is_error=is_error,
        )
        restored = EvaluationResult.model_validate_json(original.model_dump_json())
        assert restored == original


# ===========================================================================
# Property-based tests: Phase1Result with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestPhase1ResultPropertyBased:
    """Property-based tests for Phase1Result using Hypothesis."""

    @given(
        num_models=st.integers(min_value=0, max_value=5),
        initial_score=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_any_model_count_with_valid_score_is_accepted(
        self, num_models: int, initial_score: float
    ) -> None:
        """Property: any number of retrieved models with a valid score is accepted."""
        models = [
            _make_retrieved_model(model_name=f"Model_{i}") for i in range(num_models)
        ]
        solutions = [
            _make_solution_script(content=f"solution_{i}") for i in range(num_models)
        ]
        scores: list[float | None] = [
            initial_score + i * 0.01 for i in range(num_models)
        ]
        result = Phase1Result(
            retrieved_models=models,
            candidate_solutions=solutions,
            candidate_scores=scores,
            initial_solution=_make_solution_script(),
            initial_score=initial_score,
        )
        assert len(result.retrieved_models) == num_models
        assert len(result.candidate_solutions) == num_models
        assert len(result.candidate_scores) == num_models
        assert result.initial_score == initial_score

    @given(
        initial_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_initial_score(self, initial_score: float) -> None:
        """Property: JSON round-trip preserves initial_score."""
        original = _make_phase1_result(initial_score=initial_score)
        restored = Phase1Result.model_validate_json(original.model_dump_json())
        assert restored.initial_score == original.initial_score


# ===========================================================================
# Property-based tests: Phase2Result with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestPhase2ResultPropertyBased:
    """Property-based tests for Phase2Result using Hypothesis."""

    @given(
        num_summaries=st.integers(min_value=0, max_value=5),
        best_score=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_any_summary_count_with_valid_score_is_accepted(
        self, num_summaries: int, best_score: float
    ) -> None:
        """Property: any number of ablation summaries with a valid score is accepted."""
        summaries = [f"Summary {i}" for i in range(num_summaries)]
        result = _make_phase2_result(
            ablation_summaries=summaries,
            best_score=best_score,
        )
        assert len(result.ablation_summaries) == num_summaries
        assert result.best_score == best_score

    @given(
        best_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_best_score(self, best_score: float) -> None:
        """Property: JSON round-trip preserves best_score."""
        original = _make_phase2_result(best_score=best_score)
        restored = Phase2Result.model_validate_json(original.model_dump_json())
        assert restored.best_score == original.best_score


# ===========================================================================
# Property-based tests: Phase3Result with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestPhase3ResultPropertyBased:
    """Property-based tests for Phase3Result using Hypothesis."""

    @given(
        num_plans=st.integers(min_value=0, max_value=5),
        best_score=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_any_plan_count_with_valid_score_is_accepted(
        self, num_plans: int, best_score: float
    ) -> None:
        """Property: any number of ensemble plans with a valid score is accepted."""
        plans = [f"Plan {i}" for i in range(num_plans)]
        scores: list[float | None] = [best_score + i * 0.01 for i in range(num_plans)]
        result = _make_phase3_result(
            ensemble_plans=plans,
            ensemble_scores=scores,
            best_ensemble_score=best_score,
        )
        assert len(result.ensemble_plans) == num_plans
        assert len(result.ensemble_scores) == num_plans
        assert result.best_ensemble_score == best_score

    @given(
        best_score=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_round_trip_preserves_best_ensemble_score(self, best_score: float) -> None:
        """Property: JSON round-trip preserves best_ensemble_score."""
        original = _make_phase3_result(best_ensemble_score=best_score)
        restored = Phase3Result.model_validate_json(original.model_dump_json())
        assert restored.best_ensemble_score == original.best_ensemble_score

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
    def test_ensemble_scores_with_none_values_round_trip(
        self, scores: list[float | None]
    ) -> None:
        """Property: ensemble_scores with None values survive round-trip."""
        original = _make_phase3_result(ensemble_scores=scores)
        restored = Phase3Result.model_validate_json(original.model_dump_json())
        assert restored.ensemble_scores == original.ensemble_scores


# ===========================================================================
# Property-based tests: FinalResult with Hypothesis
# ===========================================================================


@pytest.mark.unit
class TestFinalResultPropertyBased:
    """Property-based tests for FinalResult using Hypothesis."""

    @given(
        duration=st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30)
    def test_any_valid_duration_accepted(self, duration: float) -> None:
        """Property: any valid duration creates a valid FinalResult."""
        result = _make_final_result(
            total_duration_seconds=duration,
        )
        assert result.total_duration_seconds == duration

    @given(
        num_phase2=st.integers(min_value=0, max_value=4),
    )
    @settings(max_examples=20)
    def test_any_number_of_phase2_results_accepted(self, num_phase2: int) -> None:
        """Property: any number of Phase2Result entries is accepted."""
        p2_results = [_make_phase2_result() for _ in range(num_phase2)]
        result = _make_final_result(phase2_results=p2_results)
        assert len(result.phase2_results) == num_phase2

    @given(
        duration=st.floats(
            min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=20)
    def test_round_trip_preserves_duration(self, duration: float) -> None:
        """Property: JSON round-trip preserves total_duration_seconds."""
        original = _make_final_result(
            total_duration_seconds=duration,
        )
        restored = FinalResult.model_validate_json(original.model_dump_json())
        assert restored.total_duration_seconds == original.total_duration_seconds

    @given(
        with_phase3=st.booleans(),
    )
    @settings(max_examples=20)
    def test_round_trip_preserves_phase3_presence(self, with_phase3: bool) -> None:
        """Property: JSON round-trip preserves whether phase3 is None or not."""
        p3 = _make_phase3_result() if with_phase3 else None
        original = _make_final_result(phase3=p3)
        restored = FinalResult.model_validate_json(original.model_dump_json())
        if with_phase3:
            assert restored.phase3 is not None
        else:
            assert restored.phase3 is None
